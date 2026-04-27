# -*- coding: utf-8 -*-
from collections import OrderedDict
import os
import numpy as np
import yaml
import pandas as pd
import torch

from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping, Callback
from skorch.helper import predefined_split as sk_predefined_split
from skorch.dataset import Dataset as SkorchDataset

from model.MTMF import MTMFTransformer, FT_Vote
from model.FT_transformer import FTTransformer
from model.MBT import MBT

from utils import setup_seed, evaluate, check_record
from dateset import load_single_features, load_full_features, load_multi_features


# ------------------ 工具：保存 best ------------------ #
def save_best_model(net, output_dir: str):
    """在模型训练过程中，将表现最好的模型状态持久化保存到本地硬盘中"""
    os.makedirs(output_dir, exist_ok=True)
    net.save_params(
        f_params=os.path.join(output_dir, "model_best.pkl"),
        f_optimizer=os.path.join(output_dir, "optim_best.pkl"),
        f_history=os.path.join(output_dir, "history_best.json"),
    )


class SaveModel(Callback):
    """
    每当 valid_acc_best 刷新，就把 best 写到：
      ./Checkpoints/{disease}/evaluate/{phase_tag}/
    注：不按折/seed 分目录，满足你“只按阶段保存”的需求。
    """
    def __init__(self, disease: str, seed: int, phase_tag: str = "baseline"):
        self.output_dir = f"./Checkpoints/{disease}/evaluate/seed{seed}/{phase_tag}"
        os.makedirs(self.output_dir, exist_ok=True)

    def initialize(self):
        self.critical_epoch_ = -1

    def on_epoch_end(self, net, **kwargs):
        if net.history[-1, 'valid_acc_best']:
            save_best_model(net, self.output_dir)
# --------------------------------------------------- #


def train(
    disease: str,
    feature: str,
    seed: int,
    model_type: str,              # 模型架构名称
    *,
    predefined_split=None,        # (train_ids, val_ids) -> sample_id 字符串，接收预先划分好的训练集和验证集的样本 ID 索引
    fold_name: str | None = None, # 仅用于结果文件名区分折次
    use_config: bool = False,
    mode: int = 0,                # 当前的训练阶段，0=baseline, 1/2/3 = phase1/2/3
    use_bottleneck: bool = True,  # 是否使用瓶颈层（Bottleneck）结构
    btn_init: str = "embed",      # 瓶颈层的初始化方式
    use_cross_atn: bool = True,   # 是否使用交叉注意力机制（Cross-Attention）
    noise: float = 0,             # 数据中添加的噪声比例
    **kwargs
):
    # 读取/准备超参
    if use_config:
        config_path = f"Config/{disease}.yaml"
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            modelconfig = dict(config[model_type][feature])
    else:
        modelconfig = dict(kwargs)  # 防侧写

    # 固定随机种子
    setup_seed(seed)

    # 将传入的字符串格式的特征参数解析成特征列表（List）
    feature_list = [s.strip() for s in feature.split(",") if s.strip()]

    # -------------------- 数据加载（透传本折划分） -------------------- #
    """根据模型类型和特征模态的数量，动态地路由到不同的数据加载函数，同时提取出输入数据的维度信息"""
    if model_type == "FT" and len(feature_list) == 1:  # 单模态
        x_train, x_val, y_train, y_val = load_single_features(
            seed=seed, disease=disease, feature=feature_list[0],
            predefined_split=predefined_split
        )
        inputs_dim = OrderedDict({"f1_input": x_train['f1_input'].shape})  # 记录模态输入维度
    else:
        if len(feature_list) == 2:  # 双模态
            x_train, x_val, y_train, y_val = load_full_features(
                seed=seed, disease=disease, feature=feature_list, noise=noise,
                predefined_split=predefined_split
            )
            inputs_dim = OrderedDict({  # 记录各个模态输入维度
                "f1_input": x_train['f1_input'].shape,
                "f2_input": x_train['f2_input'].shape
            })
        else:  # 多模态（大于2个）
            x_train, x_val, y_train, y_val = load_multi_features(
                seed=seed, disease=disease, features=feature_list, sort=False,
                predefined_split=predefined_split
            )
            inputs_dim = OrderedDict({k: x_train[k].shape for k in x_train.keys()})  # 记录各个模态输入维度

    # 非多输入模型（FT / FT-Concat）拼接所有模态的特征
    if model_type not in ['MTMFTransformer', 'FT-Vote', 'MBT']:
        x_train = np.concatenate(list(x_train.values()), axis=1).astype(np.float32)
        x_val   = np.concatenate(list(x_val.values()), axis=1).astype(np.float32)

    # 设备与优化超参
    device = "cuda"
    modelconfig['lr'] = float(modelconfig['lr'])
    lr = modelconfig['lr']  # 学习率
    batch_size = int(modelconfig['batch_size'])  # 批大小

    # -------------------- 模型构造与记录 -------------------- #
    """根据不同的模型类型（model_type），动态地向模型配置字典（modelconfig）中注入专属的架构参数"""
    if model_type == "MTMFTransformer":
        modelconfig['inputs_dim'] = inputs_dim
        modelconfig['use_bottleneck'] = use_bottleneck
        modelconfig['btn_init'] = btn_init
        modelconfig['use_cross_atn'] = use_cross_atn
        record = OrderedDict(modelconfig)
        record.pop('inputs_dim')
    elif "FT" in model_type:
        record = OrderedDict(modelconfig)
        modelconfig['last_layer_query_idx'] = [-1]
        modelconfig['d_out'] = 1
        modelconfig['cat_cardinalities'] = None
        if model_type != "FT-Vote":
            modelconfig['n_num_features'] = x_train.shape[1]
        else:
            modelconfig['n_num_features'] = inputs_dim
    elif model_type == "MBT":
        modelconfig['inputs_dim'] = inputs_dim
        modelconfig['use_bottleneck'] = use_bottleneck
        record = OrderedDict(modelconfig)
        record.pop('inputs_dim')
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # 记录实验参数
    phase_tag = f"phase{mode}" if isinstance(mode, int) and mode > 0 else "baseline"
    record['seed'] = seed
    record['mode'] = phase_tag            # 可读的阶段名
    record['feature'] = ','.join(feature_list)
    if fold_name:
        record['fold'] = fold_name

    # 结果文件命名（带 fold_name，避免覆盖）
    logdir = f"./results/{disease}"
    os.makedirs(logdir, exist_ok=True)
    if use_config:
        base_log = f"{logdir}/results-{model_type}_res.csv"
    elif noise:
        base_log = f"{logdir}/results-{model_type}_noise_{noise}.csv"
    else:
        base_log = f"{logdir}/{model_type}.csv"
    if fold_name:
        if base_log.endswith("_res.csv"):
            logpath = base_log.replace("_res.csv", f"_{fold_name}_res.csv")
        elif base_log.endswith(".csv"):
            logpath = base_log.replace(".csv", f"_{fold_name}.csv")
        else:
            logpath = base_log + f"_{fold_name}"
    else:
        logpath = base_log

    print(logpath)
    if not check_record(record, logpath):
        print("paras has trained.")
        return

    # 真正构建模型
    modelconfig.pop('lr')
    modelconfig.pop('batch_size')

    if model_type == "MTMFTransformer":
        model = MTMFTransformer(**modelconfig).cuda()
    elif model_type in ("FT-Concat", "FT"):
        model = FTTransformer.make_default(**modelconfig).cuda()
    elif model_type == "FT-Vote":
        model = FT_Vote(**modelconfig).cuda()
    elif model_type == "MBT":
        model = MBT(**modelconfig).cuda()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # 多卡
    if torch.cuda.device_count() > 1:
        print(f"[Multi-GPU] Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    # 损失函数
    if disease == 'Obesity':
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([0.5]))
    else:
        criterion = torch.nn.BCEWithLogitsLoss

    # 验证集（skorch 的 predefined_split）
    y_val_for_skorch = np.expand_dims(np.asarray(y_val, dtype=np.float32), axis=1)
    valid_ds = SkorchDataset(x_val, y_val_for_skorch)
    splitter = sk_predefined_split(valid_ds)

    # 回调：早停 + 保存最优（按阶段目录）
    callbacks = [
        EarlyStopping(patience=15),
        SaveModel(disease=disease, seed=seed, phase_tag=phase_tag),
    ]

    net = NeuralNetClassifier(
        model,
        max_epochs=200,
        criterion=criterion,
        lr=lr,
        iterator_train__shuffle=True,
        train_split=splitter,
        device=device,
        optimizer=torch.optim.AdamW,
        optimizer__weight_decay=1e-4,
        batch_size=batch_size,
        callbacks=callbacks
    )

    # 训练
    net.fit(x_train, y_train)

    # 从该阶段目录加载 best（若存在）
    ckpt_dir = f"./Checkpoints/{disease}/evaluate/seed{seed}/{phase_tag}"
    try:
        net.load_params(
            f_params=os.path.join(ckpt_dir, "model_best.pkl"),
            f_optimizer=os.path.join(ckpt_dir, "optim_best.pkl"),
            f_history=os.path.join(ckpt_dir, "history_best.json"),
        )
    except Exception as e:
        print(f"[WARN] load_params 失败：{e}；使用当前权重继续评估。")

    # 评估
    scores, df = evaluate(net, x_val, y_val)
    record.update(scores)

    # 落盘
    try:
        res_df = pd.read_csv(logpath)
        record_df = pd.DataFrame(record, index=[0])
        res_df = pd.concat([res_df, record_df], ignore_index=True)
    except Exception:
        res_df = pd.DataFrame(record, index=[0])
    res_df.to_csv(logpath, index=False)

    return scores
