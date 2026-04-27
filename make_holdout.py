#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_holdout_phase_split.py  （仅按 phase 划分，不做任何有监督过滤）
流程：
  1) 读四模态原始表，按 sample_id 交集对齐与一致性校验（标签顺序一致）。
  2) 分别对 phase=0/1/2/3 生成子集：
        - phase0：不过滤分期（健康 ∪ 全部病例）
        - phase1/2/3：健康全保留 + 对应分期的病例
     在每个 phase 子集上做 subject 分组三划分 train/val/test。
  3) 落盘到 ./Data/{DISEASE}_holdout/phase{p}/：
        - {mod}_abundance.csv（包含当前 phase 的 train/val/test 三组全部样本行，保留全部特征列）
        - train_ids.txt / val_ids.txt / test_ids.txt
        - subject_ids_all.txt / train_subjects.txt / val_subjects.txt / test_subjects.txt
        - feature_order.json（记录各模态列顺序；与输入一致）
说明：
  * 不进行 t 检验/MWU/FDR/TopK 等任何“有监督过滤”，仅做划分与落盘。
"""

import os
import json
import argparse
from collections import OrderedDict
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# 依赖你项目里已有逻辑
from dateset import build_subject_series
try:
    from dateset import phase_vs_healthy_index as _phase_index
except Exception:
    _phase_index = None


# ----------------- 可配置默认值 -----------------
DISEASE   = "CRC4"
MODS      = ["ko", "species", "untarget_pos", "untarget_neg"]
SEED      = 418
SPLITS    = (0.64, 0.16, 0.20)    # train/val/test 比例
OUT_ROOT  = f"./Data/{DISEASE}_holdout"
# -----------------------------------------------


def _read_mod(disease: str, mod: str) -> pd.DataFrame:
    path = f"./Data/{disease}/{mod}_abundance.csv"
    if not os.path.isfile(path):
        raise FileNotFoundError(f"缺少 {path}")
    df = pd.read_csv(path)
    if "sample_id" not in df.columns or "label" not in df.columns:
        raise AssertionError(f"{path} 需要包含 sample_id 与 label 列")
    df["sample_id"] = df["sample_id"].astype(str).str.strip()
    # 不强制转 int，先保留原值，再在对齐时做一致性校验
    return df


def _align_by_sample_id(dfs: Dict[str, pd.DataFrame]) -> Tuple[List[str], np.ndarray, Dict[str, List[str]]]:
    """
    返回：
      sample_ids：四模态样本交集并排序后的列表
      labels_ref：交集样本的标签（np.ndarray, int）
      feat_cols：各模态的特征列名列表
    """
    sets = [set(df["sample_id"]) for df in dfs.values()]
    inter = set.intersection(*sets)
    if not inter:
        raise RuntimeError("四模态样本交集为空，请检查 sample_id 是否一致。")
    sample_ids = sorted(list(inter))

    labels_ref = None
    for mod, df in dfs.items():
        sub = df[df["sample_id"].isin(sample_ids)].sort_values("sample_id")
        # 尝试把 label 转 int；若失败，报错以便你修正源表
        y = sub["label"]
        try:
            y = y.astype(int).to_numpy()
        except Exception:
            raise AssertionError(f"模态 {mod} 的 label 列无法转为 int，请检查：\n{sub[['sample_id','label']].head()}")
        if labels_ref is None:
            labels_ref = y
        else:
            if not np.array_equal(labels_ref, y):
                raise AssertionError(f"模态 {mod} 的标签顺序与其他模态不一致。")

    feat_cols = {m: [c for c in dfs[m].columns if c not in ("sample_id", "label")] for m in dfs.keys()}
    return sample_ids, labels_ref, feat_cols


def _phase_subset_index(sample_ids: List[str], labels: np.ndarray, phase: int) -> np.ndarray:
    """健康全保留；病例仅该分期。phase=0 表示不过滤。"""
    if phase in (1, 2, 3):
        if _phase_index is not None:
            return _phase_index(sample_ids, labels, phase=phase)
        # 简易猜测：sample_id 中含 'phase{1|2|3}'
        idx = []
        for i, sid in enumerate(sample_ids):
            if labels[i] == 0:              # 健康全保留
                idx.append(i)
            elif f"phase{phase}" in str(sid).lower():
                idx.append(i)
        if not idx:
            # 若完全匹配不到分期，仅保留健康（最保守）
            idx = [i for i, y in enumerate(labels) if y == 0]
        return np.asarray(idx, dtype=int)
    else:
        return np.arange(len(sample_ids), dtype=int)


def _three_way_split(sample_ids: List[str], y: np.ndarray, seed: int,
                     ratios=(0.64, 0.16, 0.20)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    基于 subject 分组三划分（GroupShuffleSplit 两段式），返回 train/val/test 的行索引。
    """
    groups = np.asarray(build_subject_series(sample_ids))
    y = np.asarray(y).astype(int)

    r_train, r_val, r_test = ratios
    assert abs(r_train + r_val + r_test - 1.0) < 1e-6

    gss1 = GroupShuffleSplit(n_splits=1, test_size=r_test, random_state=seed)
    idx_all = np.arange(len(sample_ids))
    trva_idx, te_idx = next(gss1.split(idx_all, y, groups))

    remain_groups = groups[trva_idx]
    remain_y = y[trva_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=r_val/(r_train+r_val), random_state=seed)
    tr_idx_rel, va_idx_rel = next(gss2.split(trva_idx, remain_y, remain_groups))
    tr_idx = trva_idx[tr_idx_rel]
    va_idx = trva_idx[va_idx_rel]
    return tr_idx, va_idx, te_idx


def _write_lines(path: str, lines: List[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in lines:
            f.write(str(s) + "\n")


def _write_split_ids(out_dir: str, train_ids: List[str], val_ids: List[str], test_ids: List[str]):
    _write_lines(os.path.join(out_dir, "train_ids.txt"), train_ids)
    _write_lines(os.path.join(out_dir, "val_ids.txt"),   val_ids)
    _write_lines(os.path.join(out_dir, "test_ids.txt"),  test_ids)


def _write_subject_ids(out_dir: str, sample_ids: List[str],
                       tr_ids: List[str], va_ids: List[str], te_ids: List[str]):
    subs_all = list(dict.fromkeys(build_subject_series(sample_ids)))
    subs_tr  = sorted(set(build_subject_series(tr_ids)))
    subs_va  = sorted(set(build_subject_series(va_ids)))
    subs_te  = sorted(set(build_subject_series(te_ids)))
    _write_lines(os.path.join(out_dir, "subject_ids_all.txt"), subs_all)
    _write_lines(os.path.join(out_dir, "train_subjects.txt"),  subs_tr)
    _write_lines(os.path.join(out_dir, "val_subjects.txt"),    subs_va)
    _write_lines(os.path.join(out_dir, "test_subjects.txt"),   subs_te)


def _save_phase_csv(full_df: pd.DataFrame, out_csv: str,
                    train_ids: List[str], val_ids: List[str], test_ids: List[str]):
    """
    在一个 CSV 中输出当前 phase 的 train/val/test 三组样本行（按 train→val→test 顺序拼接），
    列为 [sample_id, label] + 原始全部特征列（不做瘦身）。
    """
    wanted = set(train_ids) | set(val_ids) | set(test_ids)
    sub = full_df[full_df["sample_id"].isin(wanted)].copy()
    # 按 train→val→test 顺序排序
    order = list(train_ids) + list(val_ids) + list(test_ids)
    order_map = {sid: i for i, sid in enumerate(order)}
    sub["__ord__"] = sub["sample_id"].map(order_map)
    sub = sub.sort_values("__ord__").drop(columns="__ord__")
    sub.to_csv(out_csv, index=False, encoding="utf-8")


def _apply_phase(phase: int,
                 dfs: Dict[str, pd.DataFrame],
                 sample_ids_all: List[str],
                 labels_all: np.ndarray,
                 feat_cols: Dict[str, List[str]]):
    """
    在给定 phase 子集上做 subject 三划分，并为每个模态写出 CSV 与辅助文件。
    """
    idx_phase = _phase_subset_index(sample_ids_all, labels_all, phase)
    sample_ids = [sample_ids_all[i] for i in idx_phase]
    labels = labels_all[idx_phase]

    # 三划分
    tr_idx, va_idx, te_idx = _three_way_split(sample_ids, labels, seed=SEED, ratios=SPLITS)
    train_ids = [sample_ids[i] for i in tr_idx]
    val_ids   = [sample_ids[i] for i in va_idx]
    test_ids  = [sample_ids[i] for i in te_idx]

    out_dir = os.path.join(OUT_ROOT, f"phase{phase}")
    os.makedirs(out_dir, exist_ok=True)

    # 写出样本清单 / subject 清单
    _write_split_ids(out_dir, train_ids, val_ids, test_ids)
    _write_subject_ids(out_dir, sample_ids, train_ids, val_ids, test_ids)

    # 写出各模态 CSV 与列顺序
    feature_order = OrderedDict()
    for m in MODS:
        out_csv = os.path.join(out_dir, f"{m}_abundance.csv")
        _save_phase_csv(dfs[m], out_csv, train_ids, val_ids, test_ids)
        feature_order[m] = feat_cols[m]

    with open(os.path.join(out_dir, "feature_order.json"), "w", encoding="utf-8") as f:
        json.dump(feature_order, f, ensure_ascii=False, indent=2)

    print(f"[phase{phase}] 划分完成：train/val/test = {len(train_ids)}/{len(val_ids)}/{len(test_ids)}")


def main():
    global DISEASE, MODS, SEED, SPLITS, OUT_ROOT

    parser = argparse.ArgumentParser(description="仅按 phase 做 subject 三划分（无任何筛列）")
    parser.add_argument("--disease", default=DISEASE, type=str)
    parser.add_argument("--mods", default=",".join(MODS), type=str,
                        help="模态列表（逗号分隔），如：ko,species,untarget_pos,untarget_neg")
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument("--splits", default="0.64,0.16,0.20", type=str,
                        help="train,val,test 比例，如 0.64,0.16,0.20")
    args = parser.parse_args()

    DISEASE = args.disease
    MODS = [s.strip() for s in args.mods.split(",") if s.strip()]
    SEED = int(args.seed)
    SPLITS = tuple(float(x) for x in args.splits.split(","))
    assert len(SPLITS) == 3 and abs(sum(SPLITS) - 1.0) < 1e-6, "splits 必须为三元组且和为1"
    OUT_ROOT = f"./Data/{DISEASE}_holdout"
    os.makedirs(OUT_ROOT, exist_ok=True)

    # 1) 读取四模态与对齐
    dfs = {m: _read_mod(DISEASE, m) for m in MODS}
    sample_ids_all, labels_all, feat_cols = _align_by_sample_id(dfs)

    print(f"[INFO] 仅划分，不筛列 -> disease={DISEASE}, seed={SEED}, splits={SPLITS}")
    print(f"       样本数（四模态交集）= {len(sample_ids_all)}；label 分布："
          f"{np.bincount(labels_all) if labels_all.size else 'N/A'}")

    # 2) 依次处理 phase0/1/2/3
    for ph in (0, 1, 2, 3):
        _apply_phase(ph, dfs, sample_ids_all, labels_all, feat_cols)

    print(f"\n[DONE] 输出目录：{OUT_ROOT}")
    print("      每个 phase 目录包含：各模态 CSV（保留全部特征列）+ 样本清单 + subject 清单 + feature_order.json")


if __name__ == "__main__":
    main()
