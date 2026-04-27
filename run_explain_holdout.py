"""
run_explain_holdout.py
在“hold-out 瘦身目录（phase0/1/2/3）”上做 SHAP 解释：
  Data/{DISEASE}_holdout/phaseX/
    ko_abundance.csv
    species_abundance.csv
    untarget_pos_abundance.csv
    untarget_neg_abundance.csv
    train_ids.txt
    val_ids.txt
    feature_order.json (可选)

要求：
  - 训练时 best 已保存在 ./Checkpoints/{DISEASE}/evaluate/seed{SEED}/phase{X}/model_best.pkl
  - Config/{DISEASE}.yaml 中有 {MODEL}[FEATURES] 的模型超参区段（与训练一致）
"""

import os
from pathlib import Path
from explainable import explain  # 用你现有的极省显存 SHAP 核心

DISEASE  = "CRC4"   # 训练用的疾病代号
SEED     = 418      # 训练用的 seed
MODEL    = "MTMFTransformer"
FEATURES = "ko,species,untarget_pos,untarget_neg"   # 必须与 YAML 键一致
DEVICE   = "cuda"   # "cuda" 或 "cpu"
HOLDOUT_ROOT = f"./Data/{DISEASE}_holdout"
MAX_BG   = 25       # 背景样本上限（建议 4 起步；OOM 再降）
BG_REP   = 20        # 背景重复次数
ONLY_MODS = None    # 仅解释某些模态，例："ko" 或 "ko,species"
SHUFFLE_BG = True   # 背景是否洗牌

def main():
    phases = []
    for p in [0, 1, 2, 3]:
        d = Path(HOLDOUT_ROOT) / f"phase{p}"
        if d.is_dir():
            phases.append((p, str(d)))

    if not phases:
        raise FileNotFoundError(f"没有发现任何相位目录：{HOLDOUT_ROOT}/phase0..3")

    print(f"[INFO] 将在 {len(phases)} 个相位上做解释：{[f'phase{p}' for p,_ in phases]}")

    for phase, fold_dir in phases:
        print(f"\n========== 解释 phase{phase} ==========")
        # 这里的 explain() 会：
        # 1) 从 fold_dir 读四张“瘦身”CSV + train/val ids
        # 2) 在 train 拟合各模态 scaler；val 作为解释样本
        # 3) 自动搜索 Checkpoints/evaluate/seed{SEED}/phase{phase}/best
        explain(
            disease=DISEASE,
            model=MODEL,
            feature=FEATURES,
            seed=SEED,
            device=DEVICE,
            max_background=MAX_BG,
            bg_repeats=BG_REP,
            only_mods=ONLY_MODS,
            shuffle_background=SHUFFLE_BG,
            phase=phase,
            fold_dir=fold_dir,
        )

if __name__ == "__main__":
    main()
