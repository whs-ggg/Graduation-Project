import sys
import os

gpu = sys.argv[1]
disease = sys.argv[2]
feature = "ko,species"
from datetime import datetime
os.environ['CUDA_VISIBLE_DEVICES'] = gpu


a=datetime.now()

seeds = [392, 412, 432, 452, 472]

n_layerss = [1, 2]
num_bottlenecks = [4, 8, 16]
lrs = [1e-4, 1e-5, 1e-3, 5e-5]
batch_sizes = [4, 8, 16]

from old_train import train

# for n_layers in n_layerss:
#     for num_bottleneck in num_bottlenecks:
#         for lr in lrs:
#             for batch_size in batch_sizes:
#                 for seed in seeds:
#                     paras = {
#                         'n_layers': n_layers,
#                         'num_bottleneck': num_bottleneck,
#                         'use_bottleneck': True,
#                         'lr': lr,
#                         'batch_size': batch_size
#                     }
#                     train(seed=seed, disease=disease, feature=feature,
#                           model_type="MTMFTransformer", use_config=False,
#                           use_cross_atn=True, btn_init="embed", mode=0, noise=0.3,  **paras
#                           )

n_layerss = [1, 2]
m_layers = [1, 2]
num_bottlenecks = [4]
hidden_sizes = [32, 0]
lrs = [1e-4, 1e-3, 6e-05]
batch_sizes = [8, 16]

for n_layers in n_layerss:
    for m_layer in m_layers:
        for num_bottleneck in num_bottlenecks:
            for hidden_size in hidden_sizes:
                for lr in lrs:
                    for batch_size in batch_sizes:
                        for seed in seeds:
                            paras = {
                                'n_layers': n_layers,
                                'm_layers': m_layer,
                                'num_bottleneck': num_bottleneck,
                                'hidden_size':hidden_size,
                                'lr': lr,
                                'batch_size': batch_size
                            }
                            train(disease, feature, seed, 'MBT',
                                  False, 0, True, 'embed', True, noise=0.05,
                                  **paras)

b=datetime.now()
# 计算运行时间
execution_time = b - a

# 打印运行时间
print(f"代码运行时间为: {execution_time}")
