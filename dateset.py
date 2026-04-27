from collections import Counter
from typing import Tuple, Dict, Any, List, Optional, Iterable, Union
import os
from os.path import join as join
import re
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from skorch.helper import SliceDict

from utils import setup_seed, check_sample_order  # noqa: F401

# sklearn>=1.1 可用 StratifiedGroupKFold；否则置为 None
try:
    from sklearn.model_selection import StratifiedGroupKFold  # type: ignore
except Exception:
    StratifiedGroupKFold = None

# -----------------------------------------------------------------------------------------
# 基础 Dataset（按文件读取一次）
class dataset(Dataset):
    """
    读取单张 CSV 的 Dataset：
    - 第一列：sample_id
    - 第二列：label（int）
    - 后续列：特征
    - [NEW] 生成 subject_id（若原文件没有该列）
    - [CHANGED] 默认不标准化（normalize=False），避免信息泄露；标准化在切分后做。
    """
    def __init__(self,
                 path: str,
                 use_cols: Optional[List[str]] = None,
                 normalize: bool = False,           # [CHANGED] 默认 False
                 sort: bool = False,
                 derive_subject: Optional[callable] = None):
        super().__init__()
        if use_cols is None:
            use_cols = []
        print(f'Load dataset from {path}')

        df = pd.read_csv(path) if not use_cols else pd.read_csv(path)[use_cols]
        if "sample_id" not in df.columns or "label" not in df.columns:
            raise ValueError(f"{path} 需要包含 sample_id 和 label 列")
        if sort:
            print("开香槟啦，要重排序 sample_id")
            df["sample_id"] = df["sample_id"].astype(str).str.strip()
            df.sort_values('sample_id', inplace=True)

        # 基础字段
        self.sample_id = df['sample_id'].astype(str).values
        self.label = df['label'].astype(int).values.squeeze()
        X = df.iloc[:, 2:].values

        #  subject_id：优先使用文件中已有列；否则由规则推断；也支持自定义函数
        if 'subject_id' in df.columns:
            self.subject_id = df['subject_id'].astype(str).values
        else:
            rule = derive_subject if derive_subject is not None else derive_subject_id
            self.subject_id = np.array([rule(s) for s in self.sample_id])

        #  是否在此处规范化（默认 False；我们会在切分后再做）
        if normalize:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        self.data = X

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

# -----------------------------------------------------------------------------------------
#  subject 工具函数
def derive_subject_id(sample_id: str) -> str:
    """
    经验规则（可按你实际命名改动）：从"样本编号（Sample ID）"中推导出"患者编号（Subject ID）"
    - CRC 病人: 如 CRA011/CRA012/CRA013 -> 取 'CRA01' 作为 subject（字母+前两位数字）
    - 正常人: 如 HP88 -> 就用全串 'HP88'
    - 如不匹配，直接返回原 sample_id
    """
    s = str(sample_id).strip()
    m = re.match(r"^([A-Za-z]+)(\d+)$", s)
    if not m:
        return s
    prefix, digits = m.groups()
    if len(digits) >= 3:
        return f"{prefix}{digits[:2]}"  # 'CRA01' / 'CRD02' ...
    return s  # 如 HP88 -> 原样返回


def build_subject_series(sample_ids: Iterable[str],
                         external_map: Optional[Dict[str, str]] = None) -> List[str]:
    """
    若提供 external_map（如 { 'CRA011':'CRA01', ... }），优先用映射；否则走规则 derive_subject_id。
    """
    out: List[str] = []
    for sid in sample_ids:
        sid = str(sid)
        if external_map and sid in external_map:
            out.append(external_map[sid])
        else:
            out.append(derive_subject_id(sid))
    return out


# ------------------------------------------------------------------------------------
#  选择行工具（兼容 ndarray / dict）
def select_rows(X: Union[np.ndarray, Dict[str, np.ndarray]],
                mask_or_idx: Union[np.ndarray, List[bool], List[int]]) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """
    统一按行筛选：
    - 若 X 是 ndarray：返回 X[mask_or_idx]
    - 若 X 是 dict[name -> ndarray]：对每个 value 做同样筛选
    """
    idx = np.asarray(mask_or_idx)
    if isinstance(X, dict):
        return {k: v[idx] for k, v in X.items()}
    else:
        X = np.asarray(X)
        return X[idx]


# ----------------------------------------------------------------------------------
#  统一的“按 subject 划分”函数
def _split_by_subject(
    X: np.ndarray,
    y: np.ndarray,
    groups: Iterable[str],
    test_size: float = 0.2,
    seed: int = 0,
    stratify: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回 (train_idx, val_idx)，保证组（subject）不交叉。
    - 优先用 StratifiedGroupKFold（可保标签平衡），否则退化为 GroupShuffleSplit。
    """
    y = np.asarray(y)
    groups = np.asarray(list(groups))

    if stratify and StratifiedGroupKFold is not None:
        n_splits = max(2, int(round(1 / test_size)))
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        # 取第一折作为 (train, val)
        train_idx, val_idx = next(cv.split(X, y, groups=groups))
        return train_idx, val_idx

    # 回退方案：GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, val_idx = next(gss.split(X, y, groups=groups))
    return train_idx, val_idx


# ---------------------------------------------------------------------------------------------
# 基础读取工具（对齐多模态）
def _read_one_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "sample_id" not in df.columns:
        raise ValueError(f"{path} 缺少 sample_id 列")
    if "label" not in df.columns:
        raise ValueError(f"{path} 缺少 label 列")
    df["sample_id"] = df["sample_id"].astype(str).str.strip()
    return df


def _align_by_sample_id(paths: List[str]) -> Tuple[pd.DataFrame, List[str], np.ndarray, List[List[str]]]:
    """
    将多张表按 sample_id 对齐；返回：
      - base_df：对齐后的 (sample_id, label, 以及所有特征列)
      - sample_ids：对齐后的样本顺序
      - y：标签
      - feat_cols_all：每张表对应的特征列名列表
    """
    dfs = []
    feat_cols_all: List[List[str]] = []
    for p in paths:
        df = _read_one_table(p)
        df = df.sort_values("sample_id")
        feat_cols = [c for c in df.columns if c not in ("sample_id", "label")]
        dfs.append(df[["sample_id", "label"] + feat_cols].copy())
        feat_cols_all.append(feat_cols)

    base = dfs[0][["sample_id", "label"]].copy()
    for i in range(1, len(dfs)):
        base = base.merge(dfs[i][["sample_id", "label"]], on=["sample_id", "label"], how="inner")

    base = base.sort_values("sample_id").reset_index(drop=True)
    sample_ids = base["sample_id"].tolist()
    y = base["label"].astype(int).to_numpy()
    return base, sample_ids, y, feat_cols_all


# -----------------------------------------------------------------------------------------------
# 单模态：支持 subject 划分 + 预定义划分
def load_single_features(seed: int,
                         disease: str,
                         feature: str,
                         *,
                         predefined_split: Optional[Tuple[List[str], List[str]]] = None,  # [NEW]
                         sort: bool = False,
                         test_size: float = 0.2
                         ) -> Tuple[SliceDict, SliceDict, np.ndarray, np.ndarray]:
    """
    单模态装载：
      - feature: 'ko' 这样的字符串
      - 若 predefined_split 提供，则使用 (train_ids, val_ids) 的 sample_id 列表直接切分
      - 否则使用 subject 分组划分（优先 StratifiedGroupKFold）
    返回：
      x_train: SliceDict(f1_input=ndarray)
      x_val  : SliceDict(f1_input=ndarray)
      y_train(np.float32, N,1), y_val(np.int)
    """
    path = f"./Data/{disease}/{feature}_abundance.csv"
    d = dataset(path, normalize=False, sort=sort)

    #  划分索引
    if predefined_split is not None:  # [NEW]
        train_ids, val_ids = predefined_split
        sid_to_idx = {sid: i for i, sid in enumerate(d.sample_id)}
        train_idx = np.array([sid_to_idx[s] for s in train_ids if s in sid_to_idx], dtype=int)
        val_idx = np.array([sid_to_idx[s] for s in val_ids if s in sid_to_idx], dtype=int)
    else:
        train_idx, val_idx = _split_by_subject(
            X=d.data,
            y=d.label.astype(int),
            groups=d.subject_id,
            test_size=test_size,
            seed=seed,
            stratify=True
        )

    #  无泄露标准化
    sc = StandardScaler().fit(d.data[train_idx])
    x_train_arr = sc.transform(d.data[train_idx]).astype(np.float32)
    x_val_arr   = sc.transform(d.data[val_idx]).astype(np.float32)

    x_train = SliceDict(f1_input=x_train_arr)
    x_val   = SliceDict(f1_input=x_val_arr)

    y_train = d.label[train_idx].astype(int)
    y_val   = d.label[val_idx].astype(int)
    print(Counter(y_train), Counter(y_val))

    y_train = np.expand_dims(y_train, axis=1).astype(np.float32)
    return x_train, x_val, y_train, y_val


# -----------------------------------------------------------------------------------------
# 双模态：支持 subject 划分 + 预定义划分
def load_full_features(seed: int,
                       disease: str,
                       feature: List[str],
                       sort: bool = False,
                       noise: float = 0.0,
                       *,
                       predefined_split: Optional[Tuple[List[str], List[str]]] = None  # [NEW]
                       ) -> Tuple[SliceDict, SliceDict, np.ndarray, np.ndarray]:
    """
    双模态（例如 ['ko','species']）：
      - 路径：./Data/{disease}/{f}_abundance.csv
      - 先对齐样本顺序；按 subject 统一划分一次索引；两模态共用
      - 各模态的标准化都严格在训练集拟合
    """
    assert len(feature) == 2, "load_full_features 仅用于两个模态"
    if noise:
        f1_path = f"./Data/{disease}/{feature[0]}_noisy_{noise}_abundance.csv"
        f2_path = f"./Data/{disease}/{feature[1]}_noisy_{noise}_abundance.csv"
    else:
        f1_path = f"./Data/{disease}/{feature[0]}_abundance.csv"
        f2_path = f"./Data/{disease}/{feature[1]}_abundance.csv"

    check_sample_order([f1_path, f2_path])  # 确保输入文件样本顺序一致（如你已有该校验）

    d1 = dataset(f1_path, normalize=False, sort=sort)
    d2 = dataset(f2_path, normalize=False, sort=sort)

    # 一致性校验（标签应一致）
    if not np.array_equal(d1.label.astype(int), d2.label.astype(int)):
        raise AssertionError("两个模态的标签不一致，请检查数据")

    #  划分索引
    if predefined_split is not None:  # [NEW]
        train_ids, val_ids = predefined_split
        sid_to_idx = {sid: i for i, sid in enumerate(d1.sample_id)}
        train_idx = np.array([sid_to_idx[s] for s in train_ids if s in sid_to_idx], dtype=int)
        val_idx   = np.array([sid_to_idx[s] for s in val_ids if s in sid_to_idx], dtype=int)
    else:
        train_idx, val_idx = _split_by_subject(
            X=d1.data,
            y=d1.label.astype(int),
            groups=d1.subject_id,
            test_size=0.2,
            seed=seed,
            stratify=True
        )

    #  保障不交叉
    assert set(d1.subject_id[train_idx]).isdisjoint(set(d1.subject_id[val_idx]))

    # 各模态各自拟合 scaler（仅在 train）
    sc1 = StandardScaler().fit(d1.data[train_idx])
    sc2 = StandardScaler().fit(d2.data[train_idx])

    x_train_1 = sc1.transform(d1.data[train_idx]).astype(np.float32)
    x_val_1   = sc1.transform(d1.data[val_idx]).astype(np.float32)
    x_train_2 = sc2.transform(d2.data[train_idx]).astype(np.float32)
    x_val_2   = sc2.transform(d2.data[val_idx]).astype(np.float32)

    x_train = SliceDict(f1_input=x_train_1, f2_input=x_train_2)
    x_val   = SliceDict(f1_input=x_val_1,   f2_input=x_val_2)

    y_train = d1.label[train_idx].astype(int)
    y_val   = d1.label[val_idx].astype(int)
    print(Counter(y_train), Counter(y_val))

    y_train = np.expand_dims(y_train, axis=1).astype(np.float32)
    return x_train, x_val, y_train, y_val


# -----------------------------------------------------------------------------------
# 多模态 N>=1：支持 subject 划分 + 预定义划分
def load_multi_features(seed: int,
                        disease: str,
                        features: List[str],
                        sort: bool = False,
                        *,
                        predefined_split: Optional[Tuple[List[str], List[str]]] = None,  # [NEW]
                        test_size: float = 0.2
                        ) -> Tuple[SliceDict, SliceDict, np.ndarray, np.ndarray]:
    """
    多模态读取与一次性划分（支持任意数量 features）。
    要求每个 CSV 结构一致：第一列 sample_id，第二列 label，后面全是特征列。
    路径：./Data/{disease}/{f}_abundance.csv
    """
    paths = [f"./Data/{disease}/{f}_abundance.csv" for f in features]
    # 若想严格检查原文件顺序一致（而非对齐），也可继续用 check_sample_order(paths)
    base_df, sample_ids, y, feat_cols_all = _align_by_sample_id(paths)

    # 为每个模态取出对齐后的特征矩阵
    X_mats: List[np.ndarray] = []
    for pi, p in enumerate(paths):
        df = _read_one_table(p).sort_values("sample_id")
        df = df[df["sample_id"].isin(sample_ids)].sort_values("sample_id")
        X_mats.append(df[feat_cols_all[pi]].to_numpy(dtype=np.float32))

    # 构造 subject_id（按对齐后的顺序）
    subjects = build_subject_series(sample_ids)

    # 划分索引
    if predefined_split is not None:  # [NEW]
        train_ids, val_ids = predefined_split
        sid_to_idx = {sid: i for i, sid in enumerate(sample_ids)}
        train_idx = np.array([sid_to_idx[s] for s in train_ids if s in sid_to_idx], dtype=int)
        val_idx   = np.array([sid_to_idx[s] for s in val_ids if s in sid_to_idx], dtype=int)
    else:
        # 仅用第一模态的 X 作为占位传入；分割只依赖 y 和 subjects
        X_dummy = X_mats[0]
        train_idx, val_idx = _split_by_subject(
            X=X_dummy, y=y, groups=subjects, test_size=test_size, seed=seed, stratify=True
        )

    # 组装 SliceDict，并对每个模态做“仅在训练集拟合”的标准化
    X_train = SliceDict()
    X_val = SliceDict()
    for i, X in enumerate(X_mats, start=1):
        sc = StandardScaler().fit(X[train_idx])
        X_train[f"f{i}_input"] = sc.transform(X[train_idx]).astype(np.float32)
        X_val[f"f{i}_input"]   = sc.transform(X[val_idx]).astype(np.float32)

    y_train = y[train_idx].astype(int)
    y_val   = y[val_idx].astype(int)
    print(Counter(y_train), Counter(y_val))

    y_train = np.expand_dims(y_train, axis=1).astype(np.float32)
    return X_train, X_val, y_train, y_val


# 兼容旧接口：一次性把多模态作为 X 返回（不划分）
def load_features(disease: str, features: List[str]) -> Tuple[SliceDict, np.ndarray]:
    """
    与原函数保持相近：直接读入并返回 X(各模态) + y，不做划分与标准化。
    """
    fps = [f"./Data/{disease}/{f}_abundance.csv" for f in features]
    base_df, sample_ids, y, feat_cols_all = _align_by_sample_id(fps)

    X = SliceDict()
    for i, p in enumerate(fps, start=1):
        df = _read_one_table(p).sort_values("sample_id")
        df = df[df["sample_id"].isin(sample_ids)].sort_values("sample_id")
        X[f"f{i}_input"] = df[feat_cols_all[i - 1]].to_numpy(dtype=np.float32)
    return X, y

# --------------------新增三个时期分别 vs H------------------------------------
# === dateset.py: 阶段解析 & 索引 ===
import re
import numpy as np
import pandas as pd

def derive_phase(sample_id: str) -> int:
    """
    阶段编码：
      - 病例多时点样本：最后一位是 1/2/3 -> 返回 1/2/3（pre/mid/post）
      - 健康或无阶段标记样本（如 HP88）：返回 0
    """
    s = str(sample_id).strip()
    if s and s[-1] in {'1', '2', '3'}:
        return int(s[-1])
    return 0

def phase_vs_healthy_index(sample_ids, labels, phase: int):
    """
    生成用于 “phase(1/2/3) vs H(0)” 的样本索引。
    保留：
      - label==1 且 derive_phase(sample_id)==phase 的病例样本
      - 所有 label==0 的健康样本
    返回：np.ndarray 升序索引
    """
    sample_ids = pd.Series(sample_ids, dtype=str)
    labels = pd.Series(labels)
    ph = sample_ids.map(derive_phase)
    keep = ((labels == 1) & (ph == phase)) | (labels == 0)
    idx = np.where(keep.values)[0]
    return np.sort(idx)
