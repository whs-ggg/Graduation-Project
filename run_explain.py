#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
from explainable import explain, _glob_ckpt_candidates  # 复用其搜索逻辑


GPU = 0                           # -1 => 强制CPU
DISEASE = "CRC4"
MODEL = "MTMFTransformer"
FEATURES = "ko,species,untarget_pos,untarget_neg"
SEEDS = [418]                     # 也可: [392, 412, 432, 452, 472]
PHASES = [3]             # 需要解释哪些阶段

# SHAP 超参
MAX_BACKGROUND = 8                # 背景样本量
BG_REPEATS = 10                   # 重复次数
ONLY_MODS = "ko"                  # 仅导出某些模态；全模态设为 None
SHUFFLE_BACKGROUND = True

# 设备
DEVICE = "cuda" if GPU >= 0 else "cpu"
if GPU >= 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)


def _find_fold_dirs(disease: str):
    pattern = os.path.join("./Data", disease, "fold*")
    dirs = sorted([d for d in glob.glob(pattern) if os.path.isdir(d)])
    if not dirs:
        raise FileNotFoundError(f"未找到任何折目录：{pattern}\n请先运行 make_cv_ttest_slim.py 生成每折瘦身数据。")
    return dirs


def main():
    fold_dirs = _find_fold_dirs(DISEASE)

    for seed in SEEDS:
        for phase in PHASES:
            # 先筛一下：只跑“有对应ckpt”的fold
            usable_folds = []
            for fold_dir in fold_dirs:
                fold_name = os.path.basename(fold_dir.rstrip("/"))
                cands = _glob_ckpt_candidates(DISEASE, MODEL, seed, phase, fold_name)
                if cands:
                    usable_folds.append(fold_dir)
                else:
                    print(f"[SKIP] 无ckpt -> seed={seed}, phase={phase}, fold={fold_name}")

            if not usable_folds:
                print(f"[WARN] 该组合无可用fold -> seed={seed}, phase={phase}")
                continue

            for fold_dir in usable_folds:
                print(f"\n[INFO] 解释 -> disease={DISEASE}, model={MODEL}, "
                      f"features={FEATURES}, seed={seed}, phase={phase}, fold={os.path.basename(fold_dir)}, device={DEVICE}")
                explain(
                    disease=DISEASE,
                    model=MODEL,
                    feature=FEATURES,
                    seed=seed,
                    device=DEVICE,
                    max_background=MAX_BACKGROUND,
                    bg_repeats=BG_REPEATS,
                    only_mods=ONLY_MODS,
                    shuffle_background=SHUFFLE_BACKGROUND,
                    phase=phase,
                    fold_dir=fold_dir
                )

    print("\n[DONE] 全部解释完成。输出在 ./explain/{DISEASE}/phaseX/{MODEL}/fold*/ 下。")


if __name__ == "__main__":
    main()
