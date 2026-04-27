#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
相对丰度(行归一化) + 非零比例过滤(≥0.2) + 丰度过滤(相对丰度阈值0.001)
输出：
  - 过滤后的 CSV（保留 id_col, label_col + 过滤后的特征；特征为相对丰度）
  - *_keep.txt（保留特征名一行一个）
  - *_filter_report.json（包含每一步统计与参数）
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Optional, Dict

# 配置字典，提供默认运行参数
DEFAULTS = {
    "ko_csv": "./datas/ko_abundance.csv",
    "species_csv": "./datas/species_abundance.csv",

    # 基础列
    "id_col": "sample_id",
    "label_col": None,     # None -> 自动取第2列

    # 固定阈值
    "nonzero_ratio": 0.2, # 非零比例过滤阈值
    "abundance_min": 0.001 # 相对丰度阈值；整列最大值≤此阈值则丢弃
}

def _row_normalize(df_features: pd.DataFrame) -> pd.DataFrame:
    """样本归一化：每行÷行和得到相对丰度；行和为0则整行置0以避免 NaN/inf。"""
    row_sums = df_features.sum(axis=1).astype(float)
    norm = df_features.div(row_sums.replace(0.0, np.nan), axis=0)
    norm = norm.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return norm

def filter_one(csv_path: str,
               out_dir: str,
               id_col: str,
               label_col_in: Optional[str],
               nonzero_ratio: float,
               abundance_min: float) -> Dict:
    """
    流程：
      0) 行归一化 -> 相对丰度
      1) 非零比例过滤：非零占比 ≥ nonzero_ratio，非零值比例小于 nonzero_ratio 则删除该列
      2) 丰度过滤（相对丰度）：整列最大值 ≤ abundance_min 则删除
    """
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.basename(csv_path)
    out_csv = os.path.join(out_dir, fname)
    out_list = os.path.join(out_dir, fname.replace(".csv", "_keep.txt"))
    report_path = os.path.join(out_dir, fname.replace(".csv", "_filter_report.json"))

    df = pd.read_csv(csv_path)
    label_col = df.columns[1] if label_col_in is None else label_col_in
    # 确保 ID 列和标签列在表格中确实存在
    assert id_col in df.columns, f"Cannot find id_col='{id_col}' in {csv_path}"
    assert label_col in df.columns, f"Cannot find label_col='{label_col}' in {csv_path}"

    # 提取特征：剔除 ID 和 Label 列，剩下的全是 KO 或物种
    feat_raw = df.drop(columns=[id_col, label_col])
    feat_cols = feat_raw.columns.tolist()

    # Step 0: 归一化处理，得到相对丰度
    feat_rel = _row_normalize(feat_raw)

    report = {
        "input_csv": csv_path,
        "n_samples": int(len(df)),
        "n_features_original": int(len(feat_cols)),
        "params": {
            "nonzero_ratio": float(nonzero_ratio),
            "abundance_min": float(abundance_min),
            "id_col": id_col,
            "label_col": label_col
        },
        "steps": []
    }

    # Step 1: 非零比例过滤
    if feat_rel.shape[1] == 0:
        keep_mask_prev = pd.Series([], dtype=bool, index=feat_rel.columns)
    else:
        nonzero = (feat_rel != 0).sum(axis=0) / len(feat_rel)  # 计算每一列（特征）非零值占总样本数的比例
        keep_mask_prev = nonzero >= float(nonzero_ratio)

    kept_after_prev = feat_rel.loc[:, keep_mask_prev]
    # 记录该步骤删掉了多少特征
    report["steps"].append({
        "step": "nonzero_ratio",
        "threshold": float(nonzero_ratio),
        "kept": int(keep_mask_prev.sum()),
        "dropped": int((~keep_mask_prev).sum())
    })

    # Step 2: 丰度过滤（相对丰度）：整列最大值 ≤ abundance_min 的丢弃
    if kept_after_prev.shape[1] > 0:
        col_max = kept_after_prev.max(axis=0)
        keep_mask_ab = col_max > float(abundance_min)  # 等价于：若全样本都≤阈值，则删
        kept_cols = kept_after_prev.columns[keep_mask_ab].tolist()

        report["steps"].append({
            "step": "abundance_relative",
            "mode": "col_max>threshold",
            "threshold": float(abundance_min),
            "kept": int(keep_mask_ab.sum()),
            "dropped": int((~keep_mask_ab).sum())
        })
    else:
        kept_cols = []
        report["steps"].append({
            "step": "abundance_relative",
            "mode": "col_max>threshold",
            "threshold": float(abundance_min),
            "kept": 0,
            "dropped": 0
        })

    # 输出
    kept_features = feat_rel[kept_cols]
    out_df = pd.concat([df[[id_col, label_col]], kept_features], axis=1)
    out_df.to_csv(out_csv, index=False, encoding="utf-8")

    with open(out_list, "w", encoding="utf-8") as f:
        for c in kept_features.columns:
            f.write(str(c) + "\n")

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    report.update({
        "n_features_final": int(out_df.shape[1] - 2),
        "out_csv": out_csv,
        "out_list": out_list,
        "report": report_path
    })
    return report


if __name__ == "__main__":
    OUT_DIR = "./Data/CRC4"  # 输出路径

    rep_ko = filter_one(
        csv_path=DEFAULTS["ko_csv"],
        out_dir=OUT_DIR,
        id_col=DEFAULTS["id_col"],
        label_col_in=DEFAULTS["label_col"],
        nonzero_ratio=DEFAULTS["nonzero_ratio"],
        abundance_min=DEFAULTS["abundance_min"],
    )
    print("[KO] done:", json.dumps(rep_ko, ensure_ascii=False, indent=2))

    rep_sp = filter_one(
        csv_path=DEFAULTS["species_csv"],
        out_dir=OUT_DIR,
        id_col=DEFAULTS["id_col"],
        label_col_in=DEFAULTS["label_col"],
        nonzero_ratio=DEFAULTS["nonzero_ratio"],
        abundance_min=DEFAULTS["abundance_min"],
    )
    print("[species] done:", json.dumps(rep_sp, ensure_ascii=False, indent=2))
