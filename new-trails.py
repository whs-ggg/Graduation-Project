import sys
import os

gpu = sys.argv[1]
disease = sys.argv[2]
feature = sys.argv[3]
from datetime import datetime

os.environ['CUDA_VISIBLE_DEVICES'] = gpu
print("使用的GPPU为： ", gpu)

from MBT_new_train import train

a=datetime.now()

seeds = [392, 412, 432, 452, 472]

n_layerss = [1, 2]
num_bottlenecks = [4, 8, 16]
lrs = [1e-4, 1e-5, 1e-3, 5e-5]
batch_sizes = [4, 8, 16]

# paras = {
#     'n_layers': 3,
#     'num_bottleneck': 8,
#     'use_bottleneck': True,
#     'lr': 1e-4,
#     'batch_size': 4
# }

for n_layers in n_layerss:
    for num_bottleneck in num_bottlenecks:
        for lr in lrs:
            for batch_size in batch_sizes:
                for seed in seeds:
                    paras = {
                        'n_layers': n_layers,
                        'num_bottleneck': num_bottleneck,
                        'use_bottleneck': True,
                        'lr': lr,
                        'batch_size': batch_size
                    }
                    train(seed=seed, disease=disease, feature=feature,
                          model_type="Ours", sort=True, use_config=False,
                          use_cross_atn=True, btn_init="embed", mode=0, **paras
                          )
                    # MBT_train2(seed=seed, disease='LC', feature=feature,
                    #            use_config=False, mode=0, **paras)
                    # MBT_train2(seed=seed, disease='C-T2D', feature=feature,
                    #            use_config=False, mode=0, **paras)
                    # MBT_train2(seed=seed, disease='IBD', feature=feature,
                    #            use_config=False, mode=0, **paras)
                    # MBT_train2(seed=seed, disease='Obesity', feature=feature,
                    #            use_config=False, mode=0, **paras)
b=datetime.now()
# 计算运行时间
execution_time = b - a

# 打印运行时间
print(f"代码运行时间为: {execution_time}")
