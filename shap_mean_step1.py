#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import pandas as pd

########################################
# 配置区
ROOT_EXPLAIN = "/hdc/yzq_python/MSFT/explain/CRC4"
MODEL_NAME   = "MTMFTransformer"
SEED_TAG     = "418"   # 如果文件名长成 418_rep1.csv 这种；如果不是就设成""空字符串

PHASES = ["phase1", "phase2", "phase3"]

OUT_DIR = "./shap_mean_per_phase"
os.makedirs(OUT_DIR, exist_ok=True)
########################################


def read_single_shap_csv(path: str) -> pd.DataFrame:
    """
    假设格式:
        行: sample_id
        列: feature
        值: shap

    我们按原样读，不设 index_col，这样第一列如果是 sample_id 就会进来当普通列。
    之后我们要判断一下第一列是不是sample_id并设成index。
    """
    df = pd.read_csv(path)
    # 常见两种情况：
    # 1) CSV第一列本来就是无名列(样本id) + 其他列是特征
    # 2) CSV已经有列名，比如 'sample_id', 'KO_0001', ...
    # 我们要确保 sample_id 变成行索引

    # 如果第一列看起来像样本id（非全数字列名 or 叫 sample_id 之类），我们就把它设成index
    first_col = df.columns[0]
    # 只要这一列不是全数值的特征（比如 "KO_0001" 不太会是第一列），
    # 我们就假设它是sample_id
    if first_col.lower() in ["sample_id", "id", "sample", "sid"]:
        df = df.set_index(first_col)
    else:
        # 如果第一列不是上面这些名字，它还是很可能是sample_id这一列
        # 我们直接把第一列当index
        df = df.set_index(first_col)

    return df


def get_phase_mean_vector(phase_code: str) -> pd.Series:
    """
    对单个 phase:
      1. 找到该phase目录下所有 *_rep*.csv
      2. 对每个rep:
           df_rep: 行=sample, 列=feature
           mean_this_rep = df_rep.mean(axis=0)  # 对样本求平均 -> 每个特征一个值
      3. 拼起来再平均 -> 最终 feature -> mean shap across reps

    返回:
      final_mean: pd.Series(index=feature, value=平均shap(带正负))
    """

    phase_dir = os.path.join(ROOT_EXPLAIN, phase_code, MODEL_NAME, phase_code)
    pattern = os.path.join(phase_dir, f"{SEED_TAG}_rep*.csv") if SEED_TAG else os.path.join(phase_dir, "*_rep*.csv")
    paths = sorted(glob.glob(pattern))
    if len(paths) == 0:
        raise FileNotFoundError(f"[{phase_code}] 没找到匹配文件: {pattern}")

    rep_feature_means = []
    for p in paths:
        df_rep = read_single_shap_csv(p)
        # axis=0 → 沿着行求平均，也就是对所有样本平均，得到每个特征的平均shap
        mean_this_rep = df_rep.mean(axis=0)
        rep_feature_means.append(mean_this_rep)

    # 把每个rep的特征均值拼成DataFrame: 行=feature, 列=rep
    mat = pd.concat(rep_feature_means, axis=1)
    # 在列方向再平均，得到最终跨rep平均
    final_mean = mat.mean(axis=1)
    final_mean.name = "mean_shap_across_reps"

    return final_mean


def main():
    for phase_code in PHASES:
        final_mean = get_phase_mean_vector(phase_code)

        out_csv = os.path.join(OUT_DIR, f"{phase_code}_mean_shap_across_reps.csv")
        # 保存成两列：feature, mean_shap_across_reps
        final_mean.to_csv(out_csv, header=True)
        print(f"[WRITE] {out_csv} ({len(final_mean)} features)")


if __name__ == "__main__":
    main()
