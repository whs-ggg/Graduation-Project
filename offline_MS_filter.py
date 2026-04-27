# -*- coding: utf-8 -*-

"""
只保留方差最大的前 30%
"""

import os
import re
import pandas as pd
import numpy as np
from typing import Optional, Tuple

CFG = {
    "pos_file": r"./4data/pos.AllSample.Quant.Nor.xlsx",
    "neg_file": r"./4data/neg.AllSample.Quant.Nor.xlsx",
    "excel_sheet": 0,
    "label_map_path": r"./4data/Sample.Info.xlsx",
    "label_sheet": 0,
    "label_col_name": "label",
    "drop_qc_regex": r"^(QC|Pool|POOL|qc)",
    "train_id_file": r"./splits/train_ids.txt",
    "var_quantile": 0.70,  # 方差分位阈值（保留方差最大的 30%）
    "disease": "CRC4",
    "out_pos_name": "untarget_pos_abundance.csv",
    "out_neg_name": "untarget_neg_abundance.csv",
}

def _read_any(path: str, sheet=0) -> pd.DataFrame:
    """自动识别文件格式（Excel 或 CSV）并处理常见的编码报错问题，将数据加载为 Pandas 的 DataFrame 格式"""
    lower = path.lower()
    if lower.endswith(".xlsx") or lower.endswith(".xls"):
        return pd.read_excel(path, sheet_name=sheet)
    if lower.endswith(".csv"):
        for enc in ("utf-8", "gbk", "latin1"):
            try:
                return pd.read_csv(path, encoding=enc)
            except Exception:
                continue
        return pd.read_csv(path, encoding="latin1", errors="ignore")
    raise ValueError(f"不支持的文件类型: {path}")

def _transpose_feature_rows(df: pd.DataFrame) -> pd.DataFrame:
    """转置为“样本为行、特征为列”的格式"""
    if df.shape[1] < 2:
        raise ValueError("输入表至少需要两列")
    feat_col = df.columns[0]
    df = df.set_index(feat_col)
    df_t = df.T
    df_t.index = df_t.index.astype(str).str.strip()
    df_t.index.name = "sample_id"
    return df_t.reset_index()

def _normalize_sample_id_col(df: pd.DataFrame) -> pd.DataFrame:
    """标准化样本 ID"""
    df["sample_id"] = df["sample_id"].astype(str).str.strip()
    if df["sample_id"].duplicated().any():
        df = df.drop_duplicates(subset=["sample_id"], keep="first")
    return df.reset_index(drop=True)

def _read_label_table(path: str, sheet=0) -> pd.DataFrame:
    return _read_any(path, sheet=sheet)

def _merge_label(df: pd.DataFrame,
                 label_map_path: Optional[str],
                 label_col: str = "label",
                 label_sheet=0) -> pd.DataFrame:
    """将已经转置好的代谢物丰度数据表与外部的样本分组标签表根据样本 ID 横向合并"""
    if label_map_path is None:
        if label_col not in df.columns:
            df.insert(1, label_col, np.nan)
        return df
    lab = _read_label_table(label_map_path, sheet=label_sheet)
    lab = lab.rename(columns={c: str(c).strip().lower() for c in lab.columns})
    id_alias = ["sample_id", "sampleid", "id", "sample"]
    lbl_alias = ["label", "y", "group", "class"]
    cand_id = next((c for c in id_alias if c in lab.columns), None)
    cand_label = next((c for c in lbl_alias if c in lab.columns), None)
    if cand_id is None:
        raise ValueError("标签文件缺少 sample_id 列")
    keep_cols = [cand_id] + ([cand_label] if cand_label else [])
    lab = lab[keep_cols].copy().rename(columns={cand_id: "sample_id"})
    if cand_label:
        lab = lab.rename(columns={cand_label: label_col})
    lab["sample_id"] = lab["sample_id"].astype(str).str.strip()
    lab = lab.drop_duplicates(subset=["sample_id"], keep="first")
    out = df.merge(lab, on="sample_id", how="left")
    cols = list(out.columns)
    cols.remove("sample_id")
    if label_col in cols:
        cols.remove(label_col)
        out = out[["sample_id", label_col] + cols]
    else:
        out.insert(1, label_col, np.nan)
    return out

def _drop_qc(df: pd.DataFrame, regex: Optional[str]) -> pd.DataFrame:
    """剔除质控样本（Quality Control, QC）或混合样本（Pool）"""
    if not regex:
        return df
    pat = re.compile(regex)
    mask = df["sample_id"].astype(str).apply(lambda x: not bool(pat.search(x)))
    return df.loc[mask].reset_index(drop=True)

def _read_train_ids(path: Optional[str]) -> Optional[set]:
    """从外部文件中读取预先设定好的“训练集样本 ID 列表”"""
    if not path or not os.path.exists(path):
        return None
    ids = []
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
        ids = df.iloc[:, 0].astype(str).str.strip().tolist()
    else:
        with open(path, "r", encoding="utf-8") as f:
            ids = [line.strip() for line in f if line.strip()]
    return set(ids)

def _filter_by_variance_quantile(df: pd.DataFrame,
                                 start_col_idx: int = 2,
                                 train_ids: Optional[set] = None,
                                 q: float = 0.75) -> Tuple[pd.DataFrame, int, int]:
    """基于方差的分位数特征筛选:计算每一列特征的方差，把那些“没什么变化”或者“变化太小”的代谢物（≤ var_quantile）剔除掉，只保留波动最明显的特征"""
    feats = df.iloc[:, start_col_idx:].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    n_before = feats.shape[1]
    if train_ids is not None:
        mask = df["sample_id"].astype(str).isin(train_ids)
        feats_fit = feats.loc[mask]
        if feats_fit.shape[0] == 0:
            feats_fit = feats
    else:
        feats_fit = feats
    var = feats_fit.var(axis=0).astype(float).fillna(0.0)
    non_const = var > 0.0
    if non_const.any():
        tau = float(var[non_const].quantile(q))
        keep_cols = var.index[var >= tau]
    else:
        keep_cols = feats.columns
    filtered = pd.concat([df.iloc[:, :start_col_idx], feats.loc[:, keep_cols]], axis=1)
    n_after = len(keep_cols)
    return filtered, n_before, n_after

def process_one(input_path: str,
                label_map_path: Optional[str],
                out_path: str,
                excel_sheet=0,
                label_sheet=0,
                label_col: str = "label",
                drop_qc_regex: Optional[str] = None,
                train_id_file: Optional[str] = None,
                var_quantile: float = 0.75) -> Tuple[int, int]:

    raw = _read_any(input_path, sheet=excel_sheet)  # 读取
    df = _transpose_feature_rows(raw)  # 转置
    df = _normalize_sample_id_col(df)  # 样本 ID 标准化
    df = _drop_qc(df, drop_qc_regex)  # 去除质控样本
    df = _merge_label(df, label_map_path, label_col=label_col, label_sheet=label_sheet)  # 对齐标签
    df.iloc[:, 2:] = df.iloc[:, 2:].apply(pd.to_numeric, errors="coerce").fillna(0.0)  # 数值类型转化
    train_ids = _read_train_ids(train_id_file)
    # 特征筛选
    df, n_before, n_after = _filter_by_variance_quantile(
        df, start_col_idx=2, train_ids=train_ids, q=var_quantile
    )
    # 防重检查
    dup_cols = df.columns[df.columns.duplicated()].tolist()
    if dup_cols:
        raise ValueError(f"发现重复列名：{dup_cols[:5]}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[{os.path.basename(out_path)}] kept {n_after}/{n_before} features (var_quantile={var_quantile})")
    return n_before, n_after

def main():
    disease = CFG["disease"]
    out_dir = os.path.join("./Data", disease)  # 结果会被存放在 ./Data/CRC4/ 文件夹下
    pos_out = os.path.join(out_dir, CFG["out_pos_name"])
    neg_out = os.path.join(out_dir, CFG["out_neg_name"])
    common_kwargs = dict(
        label_map_path=CFG["label_map_path"],
        excel_sheet=CFG["excel_sheet"],
        label_sheet=CFG["label_sheet"],
        label_col=CFG["label_col_name"],
        drop_qc_regex=CFG["drop_qc_regex"],
        train_id_file=CFG["train_id_file"],
        var_quantile=CFG["var_quantile"],
    )
    pb, pa = process_one(input_path=CFG["pos_file"], out_path=pos_out, **common_kwargs)  # 处理正离子数据
    nb, na = process_one(input_path=CFG["neg_file"], out_path=neg_out, **common_kwargs)  # 处理负离子数据
    print(f"[SUMMARY] POS: {pa}/{pb} kept | NEG: {na}/{nb} kept")

if __name__ == "__main__":
    main()
