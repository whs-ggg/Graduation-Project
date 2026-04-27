"""
main_holdout.py
- 目录结构假设：
    ./Data/CRC4_holdout/phase0/
        ko_abundance.csv
        species_abundance.csv
        untarget_pos_abundance.csv
        untarget_neg_abundance.csv
        train_ids.txt
        val_ids.txt
        test_ids.txt
        feature_order.json (可选)
    ./Data/CRC4_holdout/phase1/ ... （同上）
    ./Data/CRC4_holdout/phase2/ ... （同上）
    ./Data/CRC4_holdout/phase3/ ... （同上）
"""

import os
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

import yaml
import torch

from dateset import load_multi_features, load_full_features, load_single_features  # 复用你的加载逻辑
from train import train
from evaluate import get_trained_model  # 复用写好的加载 best.ckpt 的工具


# 可见 GPU（字符串，逗号分隔、无空格，例如 "0,1,2,3" 或 "0"）
GPU = "2,3"

# “数据疾病名（根目录名）”——指向已划分好的三划分目录所在的根
# 例如：./Data/CRC4_holdout/phase1/...  -> HOLDOUT_ROOT = "CRC4_holdout"
HOLDOUT_ROOT = "CRC4_holdout"

# YAML 的疾病名（用于读取 Config/{YAML_DISEASE}.yaml）
# 通常仍然是原来的 "CRC4"
YAML_DISEASE = "CRC4"

# 训练的模型与特征（必须与 YAML 中的键一致）
MODEL_TYPE = "MTMFTransformer"
FEATURES = "ko,species,untarget_pos,untarget_neg"

# 哪些 phase 要跑（存在的才会跑；不存在会自动跳过）
PHASES = [0,1,2,3]

# 随机种子（可一次跑多个）
SEEDS = [388,403,418,433,448]

# 是否从 YAML 读取超参；若 False，则使用 KWARGS_FOR_TRAIN
USE_CONFIG = True
KWARGS_FOR_TRAIN = {
    "lr": 1e-5,
    "batch_size": 4,
    "n_layers": 1,
    "num_bottleneck": 8,
}
# 是否使用瓶颈 / 交叉注意力 / 初始化方式（仅 MTMF/MBT 有效）
USE_BOTTLENECK = True
USE_CROSS_ATN = True
BTN_INIT = "embed"

def _read_ids(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _phase_dir(phase: int) -> Path:
    return Path("./Data") / HOLDOUT_ROOT / f"phase{phase}"


def _exists_phase(phase: int) -> bool:
    d = _phase_dir(phase)
    return d.is_dir() and (d / "ko_abundance.csv").exists()


def _load_yaml_modelconfig(model_name: str, features: str) -> dict:
    cfg_path = Path("Config") / f"{YAML_DISEASE}.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return dict(cfg[model_name][features])


def _eval_on_ids(
    disease_tag: str,
    phase: int,
    seed: int,
    features: str,
    train_ids: list[str],
    test_ids: list[str],
    model_name: str,
) -> dict:
    """
    使用与训练一致的“训练ID拟合/测试ID变换”策略来评估 test_ids。
    - 从 YAML 读取模型超参（lr/batch_size 等）
    - 利用 dateset.*_features(predefined_split=(train_ids, test_ids)) 构造 (x_tr, x_te)，
      并基于 x_tr 构建 inputs_dim（字典：{输入键: 形状}）
    - 载入 best checkpoint，直接对 x_te 做预测并计算指标
    """
    # 读取 YAML
    modelconfig = _load_yaml_modelconfig(model_name, features)
    lr = float(modelconfig["lr"])
    batch_size = int(modelconfig["batch_size"])

    # 解析模态列表
    feat_list = [s.strip() for s in features.split(",") if s.strip()]

    # ---------- 一次性加载划分后的数据 ----------
    # 注意：这几个加载函数会在内部完成“仅用 train 拟合标准化、再变换 val/test”
    if len(feat_list) == 1:
        x_tr, x_te, _, y_te = load_single_features(
            seed=seed, disease=disease_tag, feature=feat_list[0],
            predefined_split=(train_ids, test_ids)
        )
        # 单输入：保持 ndarray 形式（skorch 对单输入期望 2D 数组）
        inputs_dim = OrderedDict([("f1_input", x_tr.shape)])  # 和训练时保持“字典形式的 shape 映射”
        X_te_for_pred = x_te  # ndarray
    elif len(feat_list) == 2:
        x_tr, x_te, _, y_te = load_full_features(
            seed=seed, disease=disease_tag, feature=feat_list,
            predefined_split=(train_ids, test_ids)
        )
        # x_tr/x_te 是 dict 或 SliceDict：键一般是 'f1_input','f2_input'
        inputs_dim = OrderedDict((k, v.shape) for k, v in x_tr.items())
        X_te_for_pred = x_te  # dict/SliceDict
    else:
        x_tr, x_te, _, y_te = load_multi_features(
            seed=seed, disease=disease_tag, features=feat_list,
            predefined_split=(train_ids, test_ids)
        )
        # x_tr/x_te 是 dict 或 SliceDict：键可能是各模态名（ko/species/...）或 'f{i}_input'
        inputs_dim = OrderedDict((k, v.shape) for k, v in x_tr.items())
        X_te_for_pred = x_te  # dict/SliceDict

    # 将必须参数补齐（若 YAML 没写）
    modelconfig2 = dict(modelconfig)
    modelconfig2.setdefault("use_bottleneck", USE_BOTTLENECK)
    modelconfig2.setdefault("btn_init", BTN_INIT)
    modelconfig2.setdefault("use_cross_atn", USE_CROSS_ATN)
    modelconfig2["inputs_dim"] = inputs_dim  # 关键：MTMFTransformer 需要字典形式

    # 载入 best checkpoint（evaluate.get_trained_model 内部已适配多种目录布局）
    net = get_trained_model(
        disease=disease_tag,
        seed=seed,
        modelconfig=modelconfig2,
        explainable=False,
        phase=phase,
        model_name=model_name,
        fold_name=None,
    )

    # 预测（正类概率）
    y_proba = net.predict_proba(X_te_for_pred)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    # 指标
    out = OrderedDict()
    try:
        out["auc"] = float(roc_auc_score(y_te, y_proba))
    except Exception:
        out["auc"] = float("nan")
    out["acc"] = float(accuracy_score(y_te, y_pred))
    out["f1"] = float(f1_score(y_te, y_pred))
    out["n_test"] = int(len(y_te))
    out["seed"] = seed
    out["phase"] = phase
    return out


def main():
    # 设定可见 GPU（与原来一致）
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU

    results = []

    for phase in PHASES:
        phase_path = _phase_dir(phase)
        if not _exists_phase(phase):
            print(f"[Skip] {phase_path} 不存在或缺少 CSV，已跳过。")
            continue

        # 训练所用的“疾病名” —— 让 train() 直接在该目录下找 CSV
        disease_tag = f"{HOLDOUT_ROOT}/phase{phase}"
        train_txt = phase_path / "train_ids.txt"
        val_txt   = phase_path / "val_ids.txt"
        test_txt  = phase_path / "test_ids.txt"

        if not (train_txt.exists() and val_txt.exists() and test_txt.exists()):
            print(f"[Skip] {phase_path} 缺少 train/val/test 的 ID 列表，已跳过。")
            continue

        train_ids = _read_ids(str(train_txt))
        val_ids   = _read_ids(str(val_txt))
        test_ids  = _read_ids(str(test_txt))

        print(f"\n========== Phase {phase} | N(train)={len(train_ids)}, N(val)={len(val_ids)}, N(test)={len(test_ids)} ==========")

        for seed in SEEDS:
            print(f"\n[Seed={seed}] 开始训练 (phase{phase}) ...")

            # —— 训练（与原先一致；关键是传入 predefined_split 与 mode=phase）
            scores = train(
                disease=disease_tag,
                feature=FEATURES,
                seed=seed,
                model_type=MODEL_TYPE,
                predefined_split=(train_ids, val_ids),
                fold_name=f"holdout-phase{phase}-seed{seed}",   # 仅用于结果文件名后缀
                use_config=USE_CONFIG,
                mode=phase,
                use_bottleneck=USE_BOTTLENECK,
                btn_init=BTN_INIT,
                use_cross_atn=USE_CROSS_ATN,
                **({} if USE_CONFIG else KWARGS_FOR_TRAIN)
            )

            # —— Test 评估：加载 best.ckpt，在 test_ids 上评估
            print(f"[Seed={seed}] 载入最优权重，在 TEST 上评估 ...")
            test_metrics = _eval_on_ids(
                disease_tag=disease_tag,
                phase=phase,
                seed=seed,
                features=FEATURES,
                train_ids=train_ids,
                test_ids=test_ids,
                model_name=MODEL_TYPE
            )
            print(f"[TEST] AUC={test_metrics['auc']:.4f} | ACC={test_metrics['acc']:.4f} | F1={test_metrics['f1']:.4f} (N={test_metrics['n_test']})")

            # —— 汇总保存
            row = OrderedDict()
            row["disease"] = disease_tag
            row["model"] = MODEL_TYPE
            row["features"] = FEATURES
            row["seed"] = seed
            row["phase"] = phase
            # 训练阶段可能已返回 val 的最佳指标（若 train() 有返回的话）；我们统一补充 test 指标
            if isinstance(scores, dict):
                for k, v in scores.items():
                    row[k] = v
            for k, v in test_metrics.items():
                row[f"test_{k}"] = v
            results.append(row)

    # 写出汇总 CSV
    if results:
        out_dir = Path("./results") / HOLDOUT_ROOT
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / f"holdout_summary.csv"
        pd.DataFrame(results).to_csv(out_csv, index=False, encoding="utf-8")
        print(f"\n[Done] 汇总结果已写入：{out_csv}")
    else:
        print("\n[Warn] 没有产生任何结果，请检查上面的跳过原因。")


if __name__ == "__main__":
    main()
