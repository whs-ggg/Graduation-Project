import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model.MTMF import MTMFTransformer  # 导入你自己的模型
from skorch import NeuralNetClassifier
from skorch.dataset import ValidSplit
from skorch.callbacks import EarlyStopping, Callback
from utils import setup_seed, evaluate
from collections import OrderedDict

# 设置路径
data_path = '/hdc/yzq_python/MSFT/datas/'
ko_path = os.path.join(data_path, 'ko_abundance.csv')
species_path = os.path.join(data_path, 'species_abundance.csv')

# 假设你传入的疾病名称为 'EW-T2D'
disease = 'CRC'

# 加载数据
ko_df = pd.read_csv(ko_path)
species_df = pd.read_csv(species_path)

# 数据预处理（提取特征和标签）
def preprocess_data(df):
    """将读取到的原始数据框（DataFrame）拆分为模型可以识别的特征矩阵和标签向量"""
    features = df.iloc[:, 2:].values  # 从第三列开始是特征
    labels = df.iloc[:, 1].values     # 第二列是标签
    return features, labels

ko_features, ko_labels = preprocess_data(ko_df)
species_features, species_labels = preprocess_data(species_df)

# 数据标准化
scaler = StandardScaler()
ko_features = scaler.fit_transform(ko_features)
species_features = scaler.fit_transform(species_features)

# 划分数据集：80% 训练，20% 测试
X_train_ko, X_test_ko, y_train_ko, y_test_ko = train_test_split(ko_features, ko_labels, test_size=0.2, random_state=42)
X_train_species, X_test_species, y_train_species, y_test_species = train_test_split(species_features, species_labels, test_size=0.2, random_state=42)

# 定义自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).view(-1, 1)  # 确保标签是二维的

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# 创建数据加载器
train_dataset_ko = CustomDataset(X_train_ko, y_train_ko)  # 实例化数据集：把切分好的训练集特征和标签，丢进刚才定义的 CustomDataset 中打包
test_dataset_ko = CustomDataset(X_test_ko, y_test_ko)
train_loader_ko = DataLoader(train_dataset_ko, batch_size=1, shuffle=True)  # 创建训练集加载器：负责在训练时喂数据,每次只拿1个病人的数据喂给模型,每个训练轮次（Epoch）开始前，把病人的顺序完全打乱
test_loader_ko = DataLoader(test_dataset_ko, batch_size=1, shuffle=False)  # 测试时不需要打乱顺序，按序评估即可

train_dataset_species = CustomDataset(X_train_species, y_train_species)
test_dataset_species = CustomDataset(X_test_species, y_test_species)
train_loader_species = DataLoader(train_dataset_species, batch_size=1, shuffle=True)
test_loader_species = DataLoader(test_dataset_species, batch_size=1, shuffle=False)

# 模型定义
model = MTMFTransformer(
    n_layers=2,                       # Transformer 的层数
    num_bottleneck=16,                # 瓶颈（Bottleneck）Token 的数量
    use_bottleneck=True,              # 是否启用瓶颈机制
    btn_init='embed',                 # 瓶颈 Token 的初始化方式
    use_cross_atn=True,               # 是否启用交叉注意力机制（Cross-Attention）
    inputs_dim={'ko': (1, ko_features.shape[1]), 'species': (1, species_features.shape[1])}  # 模型输入的特征维度究竟有多大
).to('cuda')

'''
model = MTMFTransformer(
    n_layers=2, num_bottleneck=16, use_bottleneck=True,
    btn_init='embed', use_cross_atn=True, inputs_dim=inputs_dim,
    use_feat_compress=True,          # ← 打开列压缩
    compress_n_queries=1024,           # K，首跑建议 64
    compress_n_heads=8,
    compress_dropout=0.0
).to(device)
'''

# 损失函数和优化器
criterion = torch.nn.BCEWithLogitsLoss()  # 二分类问题
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 定义回调函数：保存最佳模型
class SaveModel(Callback):
    def __init__(self, disease: str, seed: int):
        # 将疾病名称包含在模型保存路径中
        self.output_dir = f"./Checkpoints/{disease}/evaluate/{seed}"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def initialize(self):
        self.critical_epoch_ = -1

    def on_epoch_end(self, net, **kwargs):
        # 保存最佳模型
        if net.history[-1, 'valid_acc_best']:
            self.save_best_model(net)

    def save_best_model(self, net):
        epoch = len(net.history)
        params_path = os.path.join(self.output_dir, "model_best.pkl")
        optim_path = os.path.join(self.output_dir, "optim_best.pkl")
        history_path = os.path.join(self.output_dir, "history_best.json")
        net.save_params(f_params=params_path,
                        f_optimizer=optim_path,
                        f_history=history_path)

# 训练函数
def train_model(model, train_loaders, optimizer, criterion, epochs=10):
    model.train()
    for epoch in range(epochs):
        for i, (ko_data, ko_labels) in enumerate(train_loaders['ko']):
            species_data, species_labels = next(iter(train_loaders['species']))  # 假设两个数据集的训练同步

            ko_data, ko_labels = ko_data.to('cuda'), ko_labels.to('cuda')
            species_data, species_labels = species_data.to('cuda'), species_labels.to('cuda')

            optimizer.zero_grad()

            # 模型的前向传播
            outputs_ko = model(ko=ko_data, species=species_data)
            outputs_species = model(ko=ko_data, species=species_data)

            # 计算损失
            loss_ko = criterion(outputs_ko, ko_labels)
            loss_species = criterion(outputs_species, species_labels)

            # 总损失
            loss = (loss_ko + loss_species) / 2

            # 反向传播
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}], Loss: {loss.item():.4f}')

# 将数据加载器传递给训练函数
train_model(model, {'ko': train_loader_ko, 'species': train_loader_species}, optimizer, criterion, epochs=10)

# 保存模型
torch.save(model.state_dict(), 'mtmf_model.pth')
print('模型已保存！')

# 评估模型
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to('cuda'), labels.to('cuda')
            outputs = model(data)
            predicted = (outputs > 0.5).float()  # 二分类
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test dataset: {accuracy:.2f}%')

# 评估模型
evaluate_model(model, test_loader_ko)
evaluate_model(model, test_loader_species)
