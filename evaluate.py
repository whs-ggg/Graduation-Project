# evaluate.py
import os
from collections import OrderedDict
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit

from skorch import NeuralNetClassifier
from skorch.dataset import ValidSplit
from skorch.helper import SliceDict

from utils import setup_seed, evaluate as eval_once, check_sample_order
from dateset import (
    dataset as _Dataset,
    build_subject_series,
    _split_by_subject,          # 直接复用 dateset.py 里的分组切分
    _align_by_sample_id,        # 复用多模态对齐逻辑（≥3 模态时与训练完全一致）
)

from model.MTMF import MTMFTransformer, FT_Vote
from model.MBT import MBT
from model.MVIB import MVIB
from model.MDL4Microbiome import MDL4Microbiome
from model.FT_transformer import FTTransformer
from model.TabularMLP import TabularMLP
from model.MSFT_explainable import MTMFTransformer_explainable


def _eval_checkpoint_candidates(
    disease: str,
    seed: int,
    phase,
    model_name: str,
    fold_name: str | None,
) -> list:
    """与 get_trained_model 内原逻辑一致：候选 checkpoint 目录列表。"""
    candidates = []
    try:
        _ph = int(phase)
    except (TypeError, ValueError):
        _ph = 0
    _phase_sub = "baseline" if _ph == 0 else f"phase{_ph}"
    candidates.append(f"./Checkpoints/{disease}/evaluate/{model_name}/seed{seed}/{_phase_sub}")
    base_phase = f"./Checkpoints/{disease}/phase{phase}/{model_name}/{seed}"
    if fold_name:
        candidates.append(os.path.join(base_phase, fold_name))
    candidates.append(base_phase)
    if str(phase) == "0" or phase == 0:
        candidates.append(f"./Checkpoints/{disease}/evaluate/seed{seed}/baseline")
    parts = str(disease).split("/")
    if len(parts) >= 2:
        candidates.append(f"./Checkpoints/{disease}/evaluate/seed{seed}/baseline")
    candidates.append(f"./Checkpoints/{disease}/evaluate/seed{seed}/phase{phase}")
    candidates.append(f"./Checkpoints/{disease}/evaluate/{seed}")
    candidates.append(f"./Checkpoints/{disease}/{model_name}/{seed}")
    return candidates


def _find_checkpoint_dir_with_file(
    disease: str,
    seed: int,
    phase,
    model_name: str,
    fold_name: str | None,
    filename: str,
) -> str:
    import glob

    candidates = _eval_checkpoint_candidates(disease, seed, phase, model_name, fold_name)
    for c in candidates:
        if os.path.isdir(c) and os.path.isfile(os.path.join(c, filename)):
            return c
    hits = []
    for c in candidates:
        if os.path.isdir(c):
            hits += glob.glob(os.path.join(c, "**", filename), recursive=True)
    if hits:
        hits.sort(key=len)
        return os.path.dirname(hits[0])
    raise FileNotFoundError(f"No {filename} found. Tried under: {candidates}")


# 1) 载入已训练模型（best.ckpt）
def get_trained_model(disease: str,
                      seed: int,
                      modelconfig: dict,
                      explainable: bool = False,
                      phase: int = 0,
                      model_name: str | None = None,
                      fold_name: str | None = None):
    import torch, os
    from skorch import NeuralNetClassifier
    from skorch.dataset import ValidSplit

    eff_arch = model_name or modelconfig.get("model_name") or modelconfig.get("model_type") or "MTMFTransformer"

    if not explainable and eff_arch == "XGBoost":
        _mn = model_name or "XGBoost"
        ckpt_dir = _find_checkpoint_dir_with_file(
            disease, seed, phase, _mn, fold_name, "xgb_classifier.pkl"
        )
        import joblib

        path = os.path.join(ckpt_dir, "xgb_classifier.pkl")
        return joblib.load(path)

    lr = float(modelconfig['lr'])
    batch_size = int(modelconfig['batch_size'])
    modelconfig = dict(modelconfig)
    modelconfig.pop('lr'); modelconfig.pop('batch_size')

    _FT_MAKE_KEYS = (
        "n_num_features",
        "cat_cardinalities",
        "n_blocks",
        "last_layer_query_idx",
        "d_out",
        "kv_compression_ratio",
        "kv_compression_sharing",
    )

    # 1) 构建模型（解释/非解释）
    # 【原逻辑】非可解释时一律：model = MTMFTransformer(**modelconfig).to("cuda")
    if not explainable:
        if eff_arch == "MBT":
            cfg = dict(modelconfig)
            cfg.pop('btn_init', None)
            cfg.pop('use_cross_atn', None)
            model = MBT(**cfg).to("cuda")
        elif eff_arch == "MVIB":
            cfg = dict(modelconfig)
            for k in ("btn_init", "use_cross_atn", "use_bottleneck"):
                cfg.pop(k, None)
            model = MVIB(**cfg).to("cuda")
        elif eff_arch == "MDL4Microbiome":
            cfg = dict(modelconfig)
            for k in ("btn_init", "use_cross_atn", "use_bottleneck"):
                cfg.pop(k, None)
            model = MDL4Microbiome(**cfg).to("cuda")
        elif eff_arch in ("FT", "FT-Concat"):
            cfg = {k: modelconfig[k] for k in _FT_MAKE_KEYS if k in modelconfig}
            if "cat_cardinalities" not in cfg:
                cfg["cat_cardinalities"] = None
            model = FTTransformer.make_default(**cfg).to("cuda")
        elif eff_arch == "FT-Vote":
            model = FT_Vote(**modelconfig).to("cuda")
        elif eff_arch == "MLP":
            cfg = {
                "in_dim": int(modelconfig["in_dim"]),
                "hidden_dims": modelconfig.get("hidden_dims"),
                "dropout": float(modelconfig.get("dropout", 0.1)),
            }
            model = TabularMLP(**cfg).to("cuda")
        else:
            model = MTMFTransformer(**modelconfig).to("cuda")
    else:
        model = MTMFTransformer_explainable(**modelconfig).to("cuda")

    if disease == 'Obesity':
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([0.5]))
    else:
        criterion = torch.nn.BCEWithLogitsLoss

    net = NeuralNetClassifier(
        model,
        max_epochs=200,
        criterion=criterion,
        lr=lr,
        iterator_train__shuffle=True,
        train_split=ValidSplit(0.2, random_state=42),
        device="cuda",
        optimizer=torch.optim.AdamW,
        optimizer__weight_decay=1e-4,
        batch_size=batch_size,
    )

    # 2) 推断模型名（用于路径）
    if model_name is None:
        model_name = eff_arch

    ckpt_dir = _find_checkpoint_dir_with_file(
        disease, seed, phase, model_name, fold_name, "model_best.pkl"
    )

    # 4) 加载：优先用 skorch 原生；失败则手动兼容（剥 module. 前缀，strict=False）
    net.initialize()
    f_params = os.path.join(ckpt_dir, "model_best.pkl")
    f_optim  = os.path.join(ckpt_dir, "optim_best.pkl")
    f_hist   = os.path.join(ckpt_dir, "history_best.json")

    load_kwargs = {}
    if os.path.isfile(f_params):
        load_kwargs["f_params"] = f_params
    if os.path.isfile(f_optim):
        load_kwargs["f_optimizer"] = f_optim
    if os.path.isfile(f_hist):
        load_kwargs["f_history"] = f_hist

    if "f_params" not in load_kwargs:
        raise FileNotFoundError(f"Missing model_best.pkl under {ckpt_dir}")

    try:
        # 常规路径（适配 skorch 保存出来的 f_params）
        net.load_params(**load_kwargs)
    except RuntimeError as e:
        # 兼容：权重里键名带 "module." 或模型前后缀不一致
        sd = torch.load(f_params, map_location="cpu")
        # 常见容器字段名适配
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
            sd = sd["model"]

        if isinstance(sd, dict):
            # 剥掉前缀 "module."
            new_sd = {}
            for k, v in sd.items():
                nk = k[7:] if k.startswith("module.") else k
                new_sd[nk] = v
            # 松加载：忽略不匹配的键
            missing, unexpected = net.module_.load_state_dict(new_sd, strict=False)
            print(f"[WARN] Fallback strict=False load. missing={len(missing)}, unexpected={len(unexpected)}")
            allow_loose = os.environ.get("MSFT_ALLOW_LOOSE_CKPT", "").lower() in ("1", "true", "yes")
            if not allow_loose and len(missing) > 30:
                raise RuntimeError(
                    f"Checkpoint 与当前模型结构严重不匹配（目录: {ckpt_dir}）。"
                    "常见原因：旧版把各 model_type 写在同一 evaluate/seed 下，后被其它模型覆盖。"
                    "请删除该目录下 model_best.pkl 并清空 results 中对应 fold 记录后重训本模型；"
                    "或设置 MSFT_ALLOW_LOOSE_CKPT=1 强制加载（不推荐，指标可能无效）。"
                ) from e
        else:
            # 若文件并非纯 state_dict，仍退回 skorch 接口（抛原异常更清晰）
            raise e

    return net


# 2) 工具：按行筛选（兼容 dict/SliceDict）
def _select_rows(X: Dict[str, np.ndarray], idx: np.ndarray) -> SliceDict:
    out = SliceDict()
    for k, v in X.items():
        out[k] = v[idx]
    return out



# 3) 工具：加载(并与训练保持一致的)切分 + 标准化
#    - 两模态：严格复用 load_full_features 的做法（不重排样本）
#    - N≥3 模态：严格复用 load_multi_features 的“对齐+排序”逻辑
#    输出：x_tr/x_te/y_tr/y_te/te_sample_ids
def _load_for_eval(
    disease: str,
    feature_list: List[str],
    seed: int,
) -> Tuple[SliceDict, SliceDict, np.ndarray, np.ndarray, List[str]]:
    # ---------- 两模态：与 dateset.load_full_features 完全一致 ----------
    if len(feature_list) == 2:
        f1_path = f"./Data/{disease}/{feature_list[0]}_abundance.csv"
        f2_path = f"./Data/{disease}/{feature_list[1]}_abundance.csv"
        # 确保原文件样本顺序一致（训练时也是这样）
        check_sample_order([f1_path, f2_path])

        d1 = _Dataset(f1_path, normalize=False, sort=False)
        d2 = _Dataset(f2_path, normalize=False, sort=False)

        # 标签一致性校验
        if not np.array_equal(d1.label.astype(int), d2.label.astype(int)):
            raise AssertionError("两个模态的标签不一致，请检查数据")

        # 与 dateset._split_by_subject 同步
        train_idx, val_idx = _split_by_subject(
            X=d1.data,
            y=d1.label.astype(int),
            groups=d1.subject_id,
            test_size=0.2,
            seed=seed,
            stratify=True
        )

        sc1 = StandardScaler().fit(d1.data[train_idx])
        sc2 = StandardScaler().fit(d2.data[train_idx])

        xtr = SliceDict(
            f1_input=sc1.transform(d1.data[train_idx]).astype(np.float32),
            f2_input=sc2.transform(d2.data[train_idx]).astype(np.float32),
        )
        xte = SliceDict(
            f1_input=sc1.transform(d1.data[val_idx]).astype(np.float32),
            f2_input=sc2.transform(d2.data[val_idx]).astype(np.float32),
        )

        y_tr = d1.label[train_idx].astype(int)
        y_te = d1.label[val_idx].astype(int)
        te_ids = d1.sample_id[val_idx].tolist()
        return xtr, xte, y_tr, y_te, te_ids

    # ---------- N≥3 模态：与 dateset.load_multi_features 完全一致 ----------
    paths = [f"./Data/{disease}/{f}_abundance.csv" for f in feature_list]
    base_df, sample_ids, y_all, feat_cols_all = _align_by_sample_id(paths)  # 已对齐&按 sample_id 排序
    subjects = build_subject_series(sample_ids)

    # 切分索引（与 dateset.load_multi_features 相同）
    X_dummy = pd.read_csv(paths[0]).sort_values("sample_id")
    X_dummy = X_dummy[X_dummy["sample_id"].isin(sample_ids)].sort_values("sample_id")
    X_dummy = X_dummy[feat_cols_all[0]].to_numpy(dtype=np.float32)

    tr_idx, va_idx = _split_by_subject(
        X=X_dummy, y=y_all, groups=subjects, test_size=0.2, seed=seed, stratify=True
    )

    # 各模态独立 scaler（仅在训练集拟合）
    X_train = SliceDict()
    X_test = SliceDict()
    for i, p in enumerate(paths, start=1):
        df = pd.read_csv(p).sort_values("sample_id")
        df = df[df["sample_id"].isin(sample_ids)].sort_values("sample_id")
        X = df[feat_cols_all[i - 1]].to_numpy(dtype=np.float32)

        sc = StandardScaler().fit(X[tr_idx])
        X_train[f"f{i}_input"] = sc.transform(X[tr_idx]).astype(np.float32)
        X_test[f"f{i}_input"] = sc.transform(X[va_idx]).astype(np.float32)

    y_tr = y_all[tr_idx].astype(int)
    y_te = y_all[va_idx].astype(int)
    te_ids = list(np.array(sample_ids)[va_idx])
    return X_train, X_test, y_tr, y_te, te_ids



# 4) 工具：阶段索引（phase-vs-healthy）
#     phase in {1,2,3} => 取“健康(label=0) ∪ 目标期别(=phase)”的样本
#     数据来源优先：Data/{disease}/phase_map.csv -> 解析 sample_id 的 phaseX
def _load_phase_map(disease: str) -> Dict[str, int]:
    mapping = {}
    csv_path = f"./Data/{disease}/phase_map.csv"
    if os.path.isfile(csv_path):
        try:
            df = pd.read_csv(csv_path)
            # 兼容大小写
            cols = {c.lower(): c for c in df.columns}
            sid_col = cols.get("sample_id") or cols.get("sampleid") or "sample_id"
            ph_col = cols.get("phase") or "phase"
            for _, row in df.iterrows():
                sid = str(row[sid_col]).strip()
                ph = row[ph_col]
                if pd.isna(ph):
                    continue
                try:
                    mapping[sid] = int(ph)
                except Exception:
                    continue
        except Exception:
            pass
    return mapping


def _infer_phase_from_id(sample_id: str) -> Optional[int]:
    s = str(sample_id).lower()
    for k in (1, 2, 3):
        if f"phase{k}" in s:
            return k
    return None


def phase_vs_healthy_index(
    sample_ids: List[str],
    labels: np.ndarray,
    phase: int,
    disease: str
) -> np.ndarray:
    """返回满足(healthy OR phase==k)的测试集索引；若无法识别，则回退为全量"""
    if phase not in (1, 2, 3):
        return np.arange(len(sample_ids), dtype=int)

    phase_map = _load_phase_map(disease)
    idx = []
    for i, sid in enumerate(sample_ids):
        lab = int(labels[i])
        if lab == 0:  # 健康
            idx.append(i)
            continue
        # 病例：查 mapping；没有则从 sample_id 猜
        ph = phase_map.get(str(sid).strip())
        if ph is None:
            ph = _infer_phase_from_id(sid)
        if ph == phase:
            idx.append(i)

    if not idx:
        # 找不到任何标注，则回退全量，避免空集
        return np.arange(len(sample_ids), dtype=int)
    return np.asarray(idx, dtype=int)



# 5) 评测入口：一次性评 “all / phase1 / phase2 / phase3”
def eval(model: str, disease: str, feature: str, seed: int):
    """
    写入 ./results/{disease}/{model}_evaluate.csv
    列含：模型超参（去掉 inputs_dim）、seed、feature、subset（all/phase1/2/3）、AUC/ACC/Recall/Precision/F1
    """
    # ---- 读取超参（与训练一致的 YAML 键）----
    config_path = f"Config/{disease}.yaml"
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        modelconfig = dict(config[model][feature])  # 防副作用

    setup_seed(seed)
    feature_list = [x.strip() for x in feature.split(",") if x.strip()]

    # ---- 数据：与训练保持一致的切分 & 标准化 ----
    x_tr, x_te, y_tr, y_te, te_ids = _load_for_eval(disease, feature_list, seed)

    # ---- inputs_dim & 模型构建参数 ----
    inputs_dim = OrderedDict({k: x_tr[k].shape for k in x_tr.keys()})
    modelconfig['inputs_dim'] = inputs_dim
    modelconfig['use_bottleneck'] = True
    modelconfig['btn_init'] = "embed"
    modelconfig['use_cross_atn'] = True

    # ---- 记录超参（去掉 inputs_dim）----
    record_base = OrderedDict(modelconfig)
    record_base.pop('inputs_dim')
    record_base['seed'] = seed
    record_base['mode'] = 0
    record_base['feature'] = ','.join(feature_list)

    # ---- 载入 best 模型 ----
    net = get_trained_model(disease=disease, seed=seed, modelconfig=modelconfig)

    # ---- 结果路径 ----
    logdir = f"./results/{disease}"
    os.makedirs(logdir, exist_ok=True)
    logpath = f"{logdir}/{model}_evaluate.csv"

    # ---- 定义四个子集：all / phase1 / phase2 / phase3 ----
    subsets = [('all', None), ('phase1', 1), ('phase2', 2), ('phase3', 3)]

    rows = []
    for name, ph in subsets:
        if ph is None:
            X = x_te
            y = y_te
            n_used = len(y_te)
        else:
            idx = phase_vs_healthy_index(te_ids, y_te, ph, disease)
            if len(idx) == 0:
                # 理论不会到这（上面做了回退），但兜底一下
                continue
            X = _select_rows(x_te, idx)
            y = y_te[idx]
            n_used = len(y)

        scores, _ = eval_once(net, X, y)
        rec = OrderedDict(record_base)
        rec.update(scores)
        rec['subset'] = name
        rec['n_test'] = int(n_used)
        rows.append(rec)

    # ---- 追加/落盘 ----
    try:
        res_df = pd.read_csv(logpath)
        res_df = pd.concat([res_df, pd.DataFrame(rows)], ignore_index=True)
    except Exception:
        res_df = pd.DataFrame(rows)
    res_df.to_csv(logpath, index=False)

    # 也打印一眼
    for r in rows:
        print(f"[{r['subset']}] AUC={r['AUC']:.4f}  ACC={r['ACC']:.4f}  "
              f"Recall={r['Recall']:.4f}  Precision={r['Precision']:.4f}  F1={r['F1']:.4f}  (n={r['n_test']})")
