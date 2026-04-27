#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
- 递归搜索 ./Checkpoints/ 下“本fold”的 model_best.pkl（适配多种目录布局）
- 形状自检：确保 ckpt 中各模态 Tokenizer 的列数 == 本fold特征列数
- 构造与训练一致的 PyTorch 模型（不走 skorch），自行 load_state_dict
- 每模态用“训练子集”拟合 StandardScaler（无信息泄露）
- DeepExplainer 优先；OOM -> 背景减半；仍不行 -> GradientExplainer
- 流式写盘：每个repeat 生成一份CSV；导出模态重要性与meta
"""

import os
import json
import glob
from collections import OrderedDict
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
import shap
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# 限制底层并行，稳内存
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
try:
    torch.set_num_threads(1)
except Exception:
    pass

# === 你的模型类（按需补充别的模型映射）===
from model.MTMF import MTMFTransformer

MODEL_CLASS_MAP = {
    "MTMFTransformer": MTMFTransformer,
}

# ========================= 默认参数 =========================
DEFAULTS = dict(
    disease="CRC4",
    model="MTMFTransformer",
    feature="ko,species,untarget_pos,untarget_neg",
    seed=418,
    device="cpu",
    max_background=4,
    bg_repeats=5,
    only_mods=None,
    shuffle_background=True,
    phase=0,
    fold_dir=None,
)
# ============================================================


# ------------------- 小工具 -------------------
def _list_mods_from_features(feature: str) -> List[str]:
    return [m.strip() for m in feature.split(",") if m.strip()]

def _read_ids(txt_path: str) -> List[str]:
    with open(txt_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def _read_fold_tables(fold_dir: str, mods: List[str]) -> Dict[str, pd.DataFrame]:
    """读取各模态下的CSV，进行格式检查和数据清洗"""
    data = {}
    for mod in mods:
        csv = os.path.join(fold_dir, f"{mod}_abundance.csv")
        if not os.path.isfile(csv):
            raise FileNotFoundError(f"{csv} 不存在（请先生成每折瘦身数据）")
        df = pd.read_csv(csv)
        assert "sample_id" in df.columns and "label" in df.columns, f"{csv} 缺 sample_id/label"
        df["sample_id"] = df["sample_id"].astype(str).str.strip()
        df["label"] = df["label"].astype(int)
        data[mod] = df
    return data

def _build_numpy_by_ids(df: pd.DataFrame, ids: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    feat_cols = [c for c in df.columns if c not in ("sample_id", "label")]
    sub = df[df["sample_id"].isin(ids)].set_index("sample_id").loc[ids].reset_index()
    X = sub[feat_cols].to_numpy(dtype=np.float32)
    y = sub["label"].astype(np.int64).to_numpy()
    return X, y, feat_cols

def _concat_dict_features(x: OrderedDict) -> np.ndarray:
    return np.concatenate([x[k] for k in x.keys()], axis=1).astype(np.float32)

def _read_feature_names_from_fold(fold_dir: str, mods: List[str]) -> Tuple[List[str], Dict[str, Tuple[int,int]]]:
    feat_names: List[str] = []
    per_mod_index: Dict[str, Tuple[int,int]] = {}
    cursor = 0
    for mod in mods:
        csv = os.path.join(fold_dir, f"{mod}_abundance.csv")
        cols = list(pd.read_csv(csv, nrows=1).columns)[2:]  # 跳过 sample_id, label
        feat_names.extend(cols)
        per_mod_index[mod] = (cursor, cursor + len(cols))
        cursor += len(cols)
    return feat_names, per_mod_index
# ------------------------------------------------


# ------------------- Wrapper：拼接->按模态切回 -------------------
class ConcatWrapper(torch.nn.Module):
    def __init__(self, base_module: torch.nn.Module, per_mod_index: Dict[str, Tuple[int,int]]):
        super().__init__()
        self.base = base_module
        self.ranges = per_mod_index

    # 对列进行切分，同一模态的列放一起
    def forward(self, x_concat: torch.Tensor):
        feed = {mod: x_concat[:, s:e] for mod, (s, e) in self.ranges.items()}
        return self.base(**feed)

# ---------------------------------------------------------------


# ------------------- 背景与解释器构造 -------------------
def _build_explainer(kind: str, net_module, background):
    """DeepExplainer解释器和GradientExplainer（内存爆炸备用）"""
    if kind == "deep":
        explainer = shap.DeepExplainer(net_module, background)
        def fwd(xb):
            sv = explainer.shap_values(xb, check_additivity=False)
            if isinstance(sv, list): sv = sv[-1]
            sv = np.asarray(sv).squeeze()
            if sv.ndim == 1: sv = sv[None, :]
            return sv
        return explainer, fwd
    elif kind == "grad":
        explainer = shap.GradientExplainer(net_module, background)
        def fwd(xb):
            sv = explainer.shap_values(xb, check_additivity=False)
            if isinstance(sv, list): sv = sv[-1]
            sv = np.asarray(sv).squeeze()
            if sv.ndim == 1: sv = sv[None, :]
            return sv
        return explainer, fwd
    else:
        raise ValueError(kind)
# -----------------------------------------------------------


# ------------------- 载入每折数据（训练模态名） -------------------
def _load_fold_for_explain(feature: str, fold_dir: str):
    mods = _list_mods_from_features(feature)
    data = _read_fold_tables(fold_dir, mods)
    train_ids = _read_ids(os.path.join(fold_dir, "train_ids.txt"))
    val_ids   = _read_ids(os.path.join(fold_dir, "val_ids.txt"))

    Xtr_dict, Xte_dict, inputs_dim = OrderedDict(), OrderedDict(), OrderedDict()
    y_tr = y_te = None

    for mod in mods:
        Xtr, ytr, _ = _build_numpy_by_ids(data[mod], train_ids)
        Xte, yte, _ = _build_numpy_by_ids(data[mod], val_ids)
        if y_tr is None:
            y_tr, y_te = ytr, yte
        else:
            assert np.array_equal(y_tr, ytr), f"{mod} 训练标签不一致"
            assert np.array_equal(y_te, yte), f"{mod} 验证标签不一致"
        sc = StandardScaler().fit(Xtr)
        Xtr_dict[mod] = sc.transform(Xtr).astype(np.float32)
        Xte_dict[mod] = sc.transform(Xte).astype(np.float32)
        inputs_dim[mod] = Xtr.shape  # (N_tr, d_m)

    feature_names, per_mod_index = _read_feature_names_from_fold(fold_dir, mods)
    Xtr_concat = _concat_dict_features(Xtr_dict)
    Xte_concat = _concat_dict_features(Xte_dict)
    return Xtr_concat, Xte_concat, y_tr, y_te, inputs_dim, feature_names, per_mod_index, val_ids
# -----------------------------------------------------------------


# ------------------- Checkpoint 查找 & 加载 -------------------
def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_sd[k[7:]] = v
        else:
            new_sd[k] = v
    return new_sd

def _check_shapes_match(sd: Dict[str, torch.Tensor], inputs_dim: OrderedDict):
    name_map = {
        "ko": "feature_tokenizers.ko_Tokenizer.weight",
        "species": "feature_tokenizers.species_Tokenizer.weight",
        "untarget_pos": "feature_tokenizers.untarget_pos_Tokenizer.weight",
        "untarget_neg": "feature_tokenizers.untarget_neg_Tokenizer.weight",
    }
    prob = []
    for mod, shape in inputs_dim.items():
        d_mod = shape[1]
        key = name_map.get(mod)
        if key in sd:
            got = int(sd[key].shape[0])
            if got != d_mod:
                prob.append((mod, d_mod, got, key))
    return prob

# 仅支持5fold的四模态函数
'''
def _glob_ckpt_candidates(disease: str, model: str, seed: int, phase: int, fold_name: str) -> List[str]:
    """广谱搜索：适配 evaluate/seed418/phase0/foldX/ 这种布局；优先匹配 model_best.pkl"""
    base = "./Checkpoints"
    seed_tags = {str(seed), f"seed{seed}"}
    phase_tag = f"phase{phase}"

    # 先列出所有包含 fold_name 的 pkl
    all_pkl = glob.glob(os.path.join(base, disease, "**", fold_name, "**", "*.pkl"), recursive=True)

    def score(path: str) -> Tuple[int,int,int,int,int]:
        # 打分：是否包含 model_best、是否同 phase、是否同 seed、是否包含模型名、路径越短越优
        name = os.path.basename(path)
        s1 = int("model_best" in name or "best_model" in name)
        s2 = int(phase_tag in path.replace("\\","/"))
        s3 = int(any(tag in path.replace("\\","/") for tag in seed_tags))
        s4 = int(model in path.replace("\\","/"))  # 可能为0（你现在就是这种）
        s5 = -len(path)  # 越短越好
        return (s1, s2, s3, s4, s5)

    cands = sorted(all_pkl, key=score, reverse=True)
    # 只保留前若干个最相关的
    return cands[:5]
'''


def _glob_ckpt_candidates(disease: str, model: str, seed: int, phase: int, fold_name: str | None):
    """
    通用 Checkpoint 搜索：
      兼容路径：
        ./Checkpoints/{disease}_holdout/phase{X}/evaluate/seed{SEED}/baseline/model_best.pkl
        ./Checkpoints/{disease}/evaluate/seed{SEED}/phase{X}/model_best.pkl
        等
    """
    base = "./Checkpoints"
    seed_tags = {str(seed), f"seed{seed}"}
    phase_tag = f"phase{phase}"

    # 收集所有 pkl
    all_pkl = glob.glob(os.path.join(base, "**", "*.pkl"), recursive=True)

    def score(path: str):
        P = path.replace("\\", "/")
        name = os.path.basename(path)
        s_best = int("model_best" in name)
        s_disease = int(disease in P)
        s_holdout = int(f"{disease}_holdout" in P)
        s_phase = int(phase_tag in P)
        s_seed = int(any(tag in P for tag in seed_tags))
        s_baseline = int("baseline" in P)
        s_eval = int("evaluate" in P)
        s_len = -len(P)
        return (s_best, s_holdout, s_disease, s_phase, s_seed, s_baseline, s_eval, s_len)

    cands = sorted(all_pkl, key=score, reverse=True)
    return cands[:8]

def _build_model_and_load(model_name: str, modelconfig: dict, ckpt_path: str, inputs_dim: OrderedDict, device: str):
    if model_name not in MODEL_CLASS_MAP:
        raise ValueError(f"未知模型：{model_name}，请在 MODEL_CLASS_MAP 里登记对应类。")
    ModelCls = MODEL_CLASS_MAP[model_name]

    cfg = dict(modelconfig)
    cfg['inputs_dim'] = inputs_dim
    cfg.setdefault('use_bottleneck', True)
    cfg.setdefault('btn_init', 'embed')
    cfg.setdefault('use_cross_atn', True)
    for k in [
        'lr', 'batch_size', 'max_epochs', 'epochs',
        'optimizer', 'optimizer__weight_decay', 'weight_decay', 'wd',
        'momentum', 'scheduler', 'patience', 'gamma', 'step_size',
        'warmup_steps', 'warmup_ratio', 'clip_grad_norm', 'label_smoothing'
    ]:
        cfg.pop(k, None)
    model = ModelCls(**cfg).to(device)
    raw = torch.load(ckpt_path, map_location=device)
    if isinstance(raw, dict) and all(isinstance(k, str) for k in raw.keys()):
        sd = raw
    elif isinstance(raw, dict) and 'state_dict' in raw:
        sd = raw['state_dict']
    else:
        sd = raw
    sd = _strip_module_prefix(sd)

    mism = _check_shapes_match(sd, inputs_dim)
    if mism:
        msg = ["[Checkpoint 与本fold列数不匹配]:"]
        for mod, need, got, key in mism:
            msg.append(f"  - {mod}: 需要 {need}, ckpt中 {got} (key={key})")
        msg.append(f"  ckpt = {ckpt_path}")
        raise RuntimeError("\n".join(msg))

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:   print("[WARN] missing keys:", missing)
    if unexpected:print("[WARN] unexpected keys:", unexpected)
    model.eval()
    return model
# -----------------------------------------------------------------

'''
def explain(
    disease: str = DEFAULTS["disease"],
    model: str = DEFAULTS["model"],
    feature: str = DEFAULTS["feature"],
    seed: int = DEFAULTS["seed"],
    device: str = DEFAULTS["device"],
    max_background: int = DEFAULTS["max_background"],
    bg_repeats: int = DEFAULTS["bg_repeats"],
    only_mods: Optional[str] = DEFAULTS["only_mods"],
    shuffle_background: bool = DEFAULTS["shuffle_background"],
    phase: int = DEFAULTS["phase"],
    fold_dir: Optional[str] = DEFAULTS["fold_dir"],
):
    if fold_dir is None or not os.path.isdir(fold_dir):
        raise FileNotFoundError("请提供有效的 fold_dir，例如 ./Data/CRC4/fold0")

    # 1) 读取 YAML，构造与训练一致的 modelconfig
    cfg_path = f"Config/{disease.split('/')[0]}.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    modelconfig = dict(config[model][feature])
'''
def explain(
    disease: str = DEFAULTS["disease"],
    model: str = DEFAULTS["model"],
    feature: str = DEFAULTS["feature"],
    seed: int = 418,    # 加默认 seed 参数
    device: str = DEFAULTS["device"],
    max_background: int = DEFAULTS["max_background"],
    bg_repeats: int = DEFAULTS["bg_repeats"],
    only_mods: Optional[str] = DEFAULTS["only_mods"],
    shuffle_background: bool = DEFAULTS["shuffle_background"],
    phase: int = DEFAULTS["phase"],
    fold_dir: Optional[str] = DEFAULTS["fold_dir"],  # 待解释
):
    if fold_dir is None or not os.path.isdir(fold_dir):
        raise FileNotFoundError("请提供有效的 fold_dir，例如 ./Data/CRC4/fold0")

    # =======================================================
    # 固定随机种子，确保 SHAP 解释完全可复现
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[INFO] 固定随机种子 seed={seed}")
    # =======================================================

    # 1) 读取 YAML，构造与训练一致的 modelconfig
    cfg_path = f"Config/{disease.split('/')[0]}.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    modelconfig = dict(config[model][feature])

    # 2) 加载该折数据（训练拟合scaler）
    Xtr, Xte, y_tr, y_te, inputs_dim, feature_names, per_mod_index, val_ids = \
        _load_fold_for_explain(feature, fold_dir)

    # 3) 找到“本fold”的 ckpt 并构建训练同款模型
    fold_name = os.path.basename(os.path.normpath(fold_dir))
    cands = _glob_ckpt_candidates(disease.split('/')[0], model, seed, phase, fold_name)
    if not cands:
        looked_root = os.path.join("./Checkpoints", disease.split('/')[0])
        raise FileNotFoundError(
            "未找到 checkpoint：\n"
            f"  - disease = {disease}, phase = {phase}, seed = {seed}, fold = {fold_name}\n"
            f"  - 搜索根目录：{looked_root}\n"
            "  - 请确认该 fold 确实训练完成并保存了 model_best.pkl（或 best_model.pkl）。"
        )
    ckpt_path = cands[0]
    print(f"[CKPT] using: {ckpt_path}")

    base = _build_model_and_load(model, modelconfig, ckpt_path, inputs_dim, device)

    # 4) 用 ConcatWrapper 将拼接矩阵切回模态，并喂原模型
    wrapper = ConcatWrapper(base, per_mod_index)

    # 5) “仅导出某些模态”的列选择（不影响前向计算）
    keep_idx = None
    keep_tag = ""
    keep_colnames = feature_names
    if only_mods:
        wanted = [m.strip() for m in only_mods.split(",") if m.strip()]
        idxs = []
        for m in wanted:
            if m not in per_mod_index:
                raise ValueError(f"[only_mods] 模态 {m} 不在本次特征里，可选：{list(per_mod_index.keys())}")
            s, e = per_mod_index[m]
            idxs.extend(range(s, e))
        keep_idx = np.array(idxs, dtype=int)
        keep_tag = "_" + "_".join(wanted)
        keep_colnames = [feature_names[i] for i in keep_idx]

    # 6) 输出目录
    out_dir = f"./explain/{disease.split('/')[0]}/phase{phase}/{model}/{fold_name}"
    os.makedirs(out_dir, exist_ok=True)

    # 7) 元信息
    meta = dict(
        disease=disease.split('/')[0],
        model=model,
        seed=seed,
        phase=phase,
        fold=fold_name,
        feature_list=_list_mods_from_features(feature),
        per_mod_index=per_mod_index,
        bg_repeats=int(bg_repeats),
        bg_size=int(max_background),
        device=device,
        stream_csv=True,
        only_mods=only_mods,
        ckpt=ckpt_path,
    )

    # 8) 统计量积累
    N_te, D_total = Xte.shape
    abs_mean_accum = np.zeros(D_total, dtype=np.float64)  # 记录每一个测试患者所有特征列的SHAP值
    count_accum = 0  # 记录测试患者

    rng = np.random.default_rng(seed + 2025)
    full_idx = np.arange(len(Xtr))

    # 9) 逐 repeat
    for r in tqdm(range(1, bg_repeats + 1), desc="Repeats", ncols=100):
        # 背景样本的抽取
        if shuffle_background and len(full_idx) > 0:
            bg_idx = rng.choice(full_idx, size=min(max_background, len(full_idx)), replace=False)
        else:
            bg_idx = np.arange(min(max_background, len(Xtr)))
        background = torch.tensor(Xtr[bg_idx], device=device)  # 背景样本

        # 解释器（OOM 兜底）
        kind_order = ["deep", "grad"]
        explainer = None
        for kind in kind_order:
            try:
                explainer, forward = _build_explainer(kind, wrapper, background)
                _ = forward(torch.tensor(Xte[:1], device=device))
                cur_kind = kind
                break
            except RuntimeError as e:  # 如果遇到OOM，将背景样本砍掉一半
                if "out of memory" in str(e).lower():
                    if background.shape[0] > 1:
                        background = background[: max(1, background.shape[0] // 2)]
                        continue
                continue
        if explainer is None:
            raise RuntimeError("无法构建 SHAP 解释器（Deep/Grad 都失败）")

        # 在空表上先写好 ID 列和所有特征列名
        out_csv = os.path.join(out_dir, f"{seed}{keep_tag}_rep{r}.csv")
        with open(out_csv, "w", encoding="utf-8") as f:
            f.write(",".join(["sample_id"] + keep_colnames) + "\n")

        """挨个提取每一个测试患者的数据，计算出他们各自的特征重要性（SHAP 值）"""
        for i in range(N_te):
            xb = torch.tensor(Xte[i:i+1], device=device)
            sv = forward(xb)  # (1, D_total)
            row = sv[0]
            row_out = row[keep_idx] if keep_idx is not None else row
            sid = val_ids[i] if i < len(val_ids) else f"idx{i}"
            with open(out_csv, "a", encoding="utf-8") as f:
                f.write(f"{sid}," + ",".join([f"{float(x):.8g}" for x in row_out]) + "\n")

            abs_mean_accum += np.abs(row)
            count_accum += 1  # 患者数+1

        print(f"[repeat {r}] kind={cur_kind}, background={len(background)}, saved -> {out_csv}")

    abs_mean_avg = abs_mean_accum / max(1, count_accum)  # 算完所有测试患者的各个特征SHAP值的和后，取平均
    # 以模态为主，计算各个相应模态的所有特征的平均值（该模态的所有特征的SHAP和 / 该模态的特征数）
    rows = []
    for mod, (s, e) in per_mod_index.items():
        rows.append(dict(modality=mod, mean_abs_shap=float(abs_mean_avg[s:e].mean())))
    pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False)\
        .to_csv(os.path.join(out_dir, f"{seed}_modality_importance.csv"), index=False)

    with open(os.path.join(out_dir, f"{seed}_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[DONE] 输出：\n  {out_dir}\n  - *_rep*.csv（每个repeat一份）\n  - {seed}_modality_importance.csv\n  - {seed}_meta.json")
