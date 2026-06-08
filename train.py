# -*- coding: utf-8 -*-
from collections import OrderedDict
import json
import os
import sys
import numpy as np
import yaml
import pandas as pd
import torch

from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping, Callback
from skorch.helper import predefined_split as sk_predefined_split
from skorch.dataset import Dataset as SkorchDataset

from model.MTMF import MTMFTransformer, FT_Vote
from model.FT_transformer import FTTransformer
from model.MBT import MBT
from model.MVIB import MVIB
from model.MDL4Microbiome import MDL4Microbiome
from model.TabularMLP import TabularMLP

from utils import setup_seed, evaluate, check_record
from dateset import load_single_features, load_full_features, load_multi_features


class MVIBNeuralNet(NeuralNetClassifier):
    """
    MVIB 总损失 = BCEWithLogits + beta * KL(q(z|x) || N(0,I))；
    KL 在 MVIB.forward 中写入 module.last_kl。
    """

    def get_loss(self, y_pred, y_true, *args, **kwargs):
        loss_cls = super().get_loss(y_pred, y_true, *args, **kwargs)
        m = self.module_
        if isinstance(m, torch.nn.DataParallel):
            m = m.module
        kl = getattr(m, "last_kl", None)
        if kl is None or not isinstance(kl, torch.Tensor):
            return loss_cls
        return loss_cls + float(getattr(m, "beta", 0.0)) * kl


def _normalize_scalar_for_csv_compare(v):
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if v is None:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        if f.is_integer():
            return int(f)
        return f
    if hasattr(v, "item") and not isinstance(v, (bytes, str, dict, list)):
        try:
            return _normalize_scalar_for_csv_compare(v.item())
        except Exception:
            pass
    return v


def _remove_matching_rows_from_results_csv(logpath: str, record_for_log: dict) -> int:
    """
    从 results CSV 中删除与 record_for_log 在相同列上完全一致的行（与 check_record 一致）。
    若删完后无行，则删除该 CSV 文件。
    返回删除行数。
    """
    if not os.path.isfile(logpath):
        return 0
    df = pd.read_csv(logpath)
    cols = list(record_for_log.keys())
    if not cols or not all(c in df.columns for c in cols):
        return 0
    target = {k: _normalize_scalar_for_csv_compare(v) for k, v in record_for_log.items()}
    drop_positions = []
    for i in range(len(df)):
        row = {c: _normalize_scalar_for_csv_compare(df.iloc[i][c]) for c in cols}
        if row == target:
            drop_positions.append(i)
    if not drop_positions:
        return 0
    df2 = df.drop(index=df.index[drop_positions]).reset_index(drop=True)
    if len(df2) == 0:
        os.remove(logpath)
    else:
        df2.to_csv(logpath, index=False)
    return len(drop_positions)


def _record_row_for_results_csv(record: OrderedDict) -> dict:
    """将 record 转为可写入单行 CSV 的纯标量/字符串（列表、元组、嵌套 dict 等 JSON 化）。"""
    row = {}
    for k, v in record.items():
        if isinstance(v, (list, tuple)):
            row[k] = json.dumps(list(v), ensure_ascii=False)
        elif isinstance(v, dict):
            row[k] = json.dumps(v, default=str, ensure_ascii=False)
        elif hasattr(v, "item") and not isinstance(v, (bytes, str)):
            try:
                row[k] = v.item()
            except Exception:
                row[k] = v
        else:
            row[k] = v
    return row


def eval_checkpoint_dir(disease: str, model_type: str, seed: int, phase_tag: str) -> str:
    """与 SaveModel 一致：按 model_type 分目录，避免 MBT/FT/MSFT 共用路径互相覆盖。"""
    return f"./Checkpoints/{disease}/evaluate/{model_type}/seed{seed}/{phase_tag}"


def legacy_eval_checkpoint_dir(disease: str, seed: int, phase_tag: str) -> str:
    """旧版路径（无 model_type）；仅作跳过训练时导出 history 的兜底。"""
    return f"./Checkpoints/{disease}/evaluate/seed{seed}/{phase_tag}"


def _net_history_to_rows(net) -> list:
    """将 skorch history 转为按 epoch 的标量行列表。"""
    hist = getattr(net, "history_", None) or getattr(net, "history", None)
    if hist is None or len(hist) == 0:
        return []
    cols = getattr(hist, "columns_", None) or getattr(hist, "columns", None)
    rows = []
    if cols:
        for i in range(len(hist)):
            row = {}
            for c in cols:
                try:
                    v = hist[i, c]
                except Exception:
                    continue
                if isinstance(v, (list, tuple, dict)):
                    continue
                row[c] = v
            rows.append(row)
    if not rows:
        for i in range(len(hist)):
            try:
                raw = dict(hist[i])
            except Exception:
                continue
            row = {k: v for k, v in raw.items() if not isinstance(v, (list, tuple, dict))}
            if row:
                rows.append(row)
    return rows


def export_checkpoint_history_to_csv(ckpt_dir: str, logpath: str, seed: int, model_type: str) -> bool:
    """
    当跳过训练（check_record 命中）时，从 SaveModel 写入的 history_best.json 恢复 epoch 曲线，
    写入与 save_training_history_csv 相同路径的 history_{model_type}_seed{seed}.csv，供 holdout 汇总图使用。
    """
    hist_path = os.path.join(ckpt_dir, "history_best.json")
    if not os.path.isfile(hist_path):
        return False
    try:
        with open(hist_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:
        print(f"[WARN] 读取 {hist_path} 失败: {e}")
        return False
    if not isinstance(raw, list):
        return False
    scalar_keys = ("train_loss", "valid_loss", "train_acc", "valid_acc")
    rows = []
    for ep in raw:
        if not isinstance(ep, dict):
            continue
        row = {}
        for k in scalar_keys:
            v = ep.get(k)
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                row[k] = v
        if "train_loss" in row or "valid_loss" in row:
            rows.append(row)
    if not rows:
        return False
    df = pd.DataFrame(rows)
    keep = [c for c in ("train_loss", "valid_loss") if c in df.columns]
    if not keep:
        return False
    df_out = df[keep].copy()
    df_out.insert(0, "epoch", np.arange(1, len(df_out) + 1))
    plot_dir = os.path.join(os.path.dirname(logpath), "plots")
    os.makedirs(plot_dir, exist_ok=True)
    out_csv = os.path.join(plot_dir, f"history_{model_type}_seed{seed}.csv")
    df_out.to_csv(out_csv, index=False)
    print(f"[History] 从 checkpoint 恢复: {out_csv}")
    return True


def save_training_history_csv(net, logpath: str, seed: int, model_type: str) -> str | None:
    """
    将本 run 的 epoch 曲线存为 CSV（供 holdout 按 phase 汇总画图）。
    须在 net.fit() 之后、load_params() 之前调用。
    """
    rows = _net_history_to_rows(net)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    keep = [c for c in ("train_loss", "valid_loss") if c in df.columns]
    if not keep:
        return None
    df_out = df[keep].copy()
    df_out.insert(0, "epoch", np.arange(1, len(df_out) + 1))
    plot_dir = os.path.join(os.path.dirname(logpath), "plots")
    os.makedirs(plot_dir, exist_ok=True)
    out_csv = os.path.join(plot_dir, f"history_{model_type}_seed{seed}.csv")
    df_out.to_csv(out_csv, index=False)
    print(f"[History] 已写入: {out_csv}")
    return out_csv


# ------------------ 工具：保存 best ------------------ #
def save_best_model(net, output_dir: str):
    """在模型训练过程中，将表现最好的模型状态持久化保存到本地硬盘中"""
    os.makedirs(output_dir, exist_ok=True)
    net.save_params(
        f_params=os.path.join(output_dir, "model_best.pkl"),
        f_optimizer=os.path.join(output_dir, "optim_best.pkl"),
        f_history=os.path.join(output_dir, "history_best.json"),
    )


class SaveModel(Callback):
    """
    每当 valid_acc_best 刷新，就把 best 写到：
      ./Checkpoints/{disease}/evaluate/{model_type}/seed{seed}/{phase_tag}/
    按 model_type 分目录，避免与其它架构共用 evaluate/seed 导致互相覆盖。
    """
    def __init__(self, disease: str, seed: int, phase_tag: str, model_type: str):
        self.output_dir = eval_checkpoint_dir(disease, model_type, seed, phase_tag)
        os.makedirs(self.output_dir, exist_ok=True)

    def initialize(self):
        self.critical_epoch_ = -1

    def on_epoch_end(self, net, **kwargs):
        if net.history[-1, 'valid_acc_best']:
            save_best_model(net, self.output_dir)
# --------------------------------------------------- #


def train(
    disease: str,
    feature: str,
    seed: int,
    model_type: str,              # 模型架构名称
    *,
    predefined_split=None,        # (train_ids, val_ids) -> sample_id 字符串，接收预先划分好的训练集和验证集的样本 ID 索引
    fold_name: str | None = None, # 仅用于结果文件名区分折次
    use_config: bool = False,
    mode: int = 0,                # 当前的训练阶段，0=baseline, 1/2/3 = phase1/2/3
    use_bottleneck: bool = True,  # 是否使用瓶颈层（Bottleneck）结构
    btn_init: str = "embed",      # 瓶颈层的初始化方式
    use_cross_atn: bool = True,   # 是否使用交叉注意力机制（Cross-Attention）
    noise: float = 0,             # 数据中添加的噪声比例
    **kwargs
):
    # 读取/准备超参
    if use_config:
        config_path = f"Config/{disease}.yaml"
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            modelconfig = dict(config[model_type][feature])
    else:
        modelconfig = dict(kwargs)  # 防侧写

    # 固定随机种子
    setup_seed(seed)

    # 将传入的字符串格式的特征参数解析成特征列表（List）
    feature_list = [s.strip() for s in feature.split(",") if s.strip()]

    # -------------------- 数据加载（透传本折划分） -------------------- #
    """根据模型类型和特征模态的数量，动态地路由到不同的数据加载函数，同时提取出输入数据的维度信息"""
    if model_type == "FT" and len(feature_list) == 1:  # 单模态
        x_train, x_val, y_train, y_val = load_single_features(
            seed=seed, disease=disease, feature=feature_list[0],
            predefined_split=predefined_split
        )
        inputs_dim = OrderedDict({"f1_input": x_train['f1_input'].shape})  # 记录模态输入维度
    else:
        if len(feature_list) == 2:  # 双模态
            x_train, x_val, y_train, y_val = load_full_features(
                seed=seed, disease=disease, feature=feature_list, noise=noise,
                predefined_split=predefined_split
            )
            inputs_dim = OrderedDict({  # 记录各个模态输入维度
                "f1_input": x_train['f1_input'].shape,
                "f2_input": x_train['f2_input'].shape
            })
        else:  # 多模态（大于2个）
            x_train, x_val, y_train, y_val = load_multi_features(
                seed=seed, disease=disease, features=feature_list, sort=False,
                predefined_split=predefined_split
            )
            inputs_dim = OrderedDict({k: x_train[k].shape for k in x_train.keys()})  # 记录各个模态输入维度

    # 非多输入模型（FT / FT-Concat / MLP / XGBoost）拼接所有模态的特征
    if model_type not in ['MTMFTransformer', 'FT-Vote', 'MBT', 'MVIB', 'MDL4Microbiome']:
        x_train = np.concatenate(list(x_train.values()), axis=1).astype(np.float32)
        x_val   = np.concatenate(list(x_val.values()), axis=1).astype(np.float32)

    # 设备与优化超参
    device = "cuda"
    modelconfig['lr'] = float(modelconfig['lr'])
    lr = modelconfig['lr']  # 学习率
    batch_size = int(modelconfig['batch_size'])  # 批大小

    # -------------------- 模型构造与记录 -------------------- #
    """根据不同的模型类型（model_type），动态地向模型配置字典（modelconfig）中注入专属的架构参数"""
    if model_type == "MTMFTransformer":
        modelconfig['inputs_dim'] = inputs_dim
        modelconfig['use_bottleneck'] = use_bottleneck
        modelconfig['btn_init'] = btn_init
        modelconfig['use_cross_atn'] = use_cross_atn
        record = OrderedDict(modelconfig)
        record.pop('inputs_dim')
    elif "FT" in model_type:
        record = OrderedDict(modelconfig)
        modelconfig['last_layer_query_idx'] = [-1]
        modelconfig['d_out'] = 1
        modelconfig['cat_cardinalities'] = None
        if model_type != "FT-Vote":
            modelconfig['n_num_features'] = x_train.shape[1]
        else:
            modelconfig['n_num_features'] = inputs_dim
    elif model_type == "MBT":
        modelconfig['inputs_dim'] = inputs_dim
        modelconfig['use_bottleneck'] = use_bottleneck
        record = OrderedDict(modelconfig)
        record.pop('inputs_dim')
    elif model_type == "MVIB":
        modelconfig['inputs_dim'] = inputs_dim
        record = OrderedDict(modelconfig)
        record.pop('inputs_dim')
    elif model_type == "MDL4Microbiome":
        modelconfig['inputs_dim'] = inputs_dim
        record = OrderedDict(modelconfig)
        record.pop('inputs_dim')
    elif model_type == "MLP":
        record = OrderedDict(modelconfig)
        modelconfig['in_dim'] = int(x_train.shape[1])
    elif model_type == "XGBoost":
        record = OrderedDict(modelconfig)
        record["n_num_features"] = int(x_train.shape[1])
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # 记录实验参数
    phase_tag = f"phase{mode}" if isinstance(mode, int) and mode > 0 else "baseline"
    record['seed'] = seed
    record['mode'] = phase_tag            # 可读的阶段名
    record['feature'] = ','.join(feature_list)
    if fold_name:
        record['fold'] = fold_name

    # 结果文件命名（带 fold_name，避免覆盖）
    logdir = f"./results/{disease}"
    os.makedirs(logdir, exist_ok=True)
    if use_config:
        base_log = f"{logdir}/results-{model_type}_res.csv"
    elif noise:
        base_log = f"{logdir}/results-{model_type}_noise_{noise}.csv"
    else:
        base_log = f"{logdir}/{model_type}.csv"
    if fold_name:
        if base_log.endswith("_res.csv"):
            logpath = base_log.replace("_res.csv", f"_{fold_name}_res.csv")
        elif base_log.endswith(".csv"):
            logpath = base_log.replace(".csv", f"_{fold_name}.csv")
        else:
            logpath = base_log + f"_{fold_name}"
    else:
        logpath = base_log

    print(logpath)
    record_for_log = OrderedDict(_record_row_for_results_csv(record))
    force_retrain = os.environ.get("MSFT_FORCE_RETRAIN", "").lower() in ("1", "true", "yes")
    auto_overwrite = os.environ.get("MSFT_AUTO_OVERWRITE", "").lower() in ("1", "true", "yes")
    no_prompt = os.environ.get("MSFT_NO_PROMPT", "").lower() in ("1", "true", "yes")

    if force_retrain:
        print("[Train] MSFT_FORCE_RETRAIN=1：忽略 results CSV 去重，将重新训练。")
    elif not check_record(record_for_log, logpath):
        do_retrain = False
        if auto_overwrite:
            do_retrain = True
            print("[Train] MSFT_AUTO_OVERWRITE=1：将清除重复记录并重新训练。")
        elif sys.stdin.isatty() and not no_prompt:
            try:
                ans = input(
                    "检测到 results 中已有与本 run 完全相同的记录（默认将跳过训练）。\n"
                    "是否从该 CSV 删除对应行并重新训练? [y/N]: "
                ).strip().lower()
                do_retrain = ans in ("y", "yes")
            except EOFError:
                do_retrain = False
        if not do_retrain:
            print("paras has trained.")
            for ckpt_dir in (
                eval_checkpoint_dir(disease, model_type, seed, phase_tag),
                legacy_eval_checkpoint_dir(disease, seed, phase_tag),
            ):
                if export_checkpoint_history_to_csv(ckpt_dir, logpath, seed, model_type):
                    break
            return
        n_removed = _remove_matching_rows_from_results_csv(logpath, record_for_log)
        if n_removed > 0:
            print(f"[Train] 已从 {logpath} 移除 {n_removed} 行重复记录，开始重新训练。")
        else:
            if os.path.isfile(logpath):
                os.remove(logpath)
                print(
                    f"[Train] 未能在 CSV 中逐行匹配到重复项（已确认重训），已删除整个文件 {logpath} 后重新训练。"
                )
            else:
                print("[Train] 结果文件已不存在，开始重新训练。")

    if model_type == "XGBoost":
        import joblib
        from model.tabular_xgb import fit_tabular_xgb_classifier

        cfg_fit = dict(modelconfig)
        cfg_fit.pop("lr", None)
        cfg_fit.pop("batch_size", None)
        clf = fit_tabular_xgb_classifier(cfg_fit, seed, x_train, y_train, x_val, y_val)
        ckpt_dir = eval_checkpoint_dir(disease, model_type, seed, phase_tag)
        os.makedirs(ckpt_dir, exist_ok=True)
        _xgb_path = os.path.join(ckpt_dir, "xgb_classifier.pkl")
        joblib.dump(clf, _xgb_path)
        print(f"[XGBoost] 已保存: {_xgb_path}")

        scores, _df = evaluate(clf, x_val, y_val)
        record.update(scores)
        row = _record_row_for_results_csv(record)
        record_df = pd.DataFrame([row])
        if os.path.isfile(logpath):
            res_df = pd.read_csv(logpath)
            res_df = pd.concat([res_df, record_df], ignore_index=True)
        else:
            res_df = record_df
        res_df.to_csv(logpath, index=False)
        return scores

    # 真正构建模型（PyTorch + skorch）
    modelconfig.pop('lr')
    modelconfig.pop('batch_size')

    if model_type == "MTMFTransformer":
        model = MTMFTransformer(**modelconfig).cuda()
    elif model_type in ("FT-Concat", "FT"):
        model = FTTransformer.make_default(**modelconfig).cuda()
    elif model_type == "FT-Vote":
        model = FT_Vote(**modelconfig).cuda()
    elif model_type == "MBT":
        model = MBT(**modelconfig).cuda()
    elif model_type == "MVIB":
        model = MVIB(**modelconfig).cuda()
    elif model_type == "MDL4Microbiome":
        model = MDL4Microbiome(**modelconfig).cuda()
    elif model_type == "MLP":
        model = TabularMLP(**modelconfig).cuda()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # 多卡：DataParallel 在部分机器上会触发 NCCL broadcast 失败（与其它任务共卡、P2P 等）。
    # MBT / MVIB 与单卡足够；默认不包 DataParallel（KL 与 last_kl 在子模块上不可靠）。需要时可设 MSFT_FORCE_DATAPARALLEL_MBT=1
    if torch.cuda.device_count() > 1:
        force_mbt_dp = os.environ.get("MSFT_FORCE_DATAPARALLEL_MBT", "").lower() in ("1", "true", "yes")
        if model_type in ("MBT", "MVIB") and not force_mbt_dp:
            print(
                "[Multi-GPU] MBT/MVIB: skip DataParallel (single visible GPU for compute; avoids NCCL issues). "
                "Set MSFT_FORCE_DATAPARALLEL_MBT=1 to enable DP for MBT."
            )
        elif model_type not in ("MBT", "MVIB"):
            print(f"[Multi-GPU] Using DataParallel on {torch.cuda.device_count()} GPUs")
            model = torch.nn.DataParallel(model)

    # 损失函数
    if disease == 'Obesity':
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([0.5]))
    else:
        criterion = torch.nn.BCEWithLogitsLoss

    # 验证集（skorch 的 predefined_split）
    y_val_for_skorch = np.expand_dims(np.asarray(y_val, dtype=np.float32), axis=1)
    valid_ds = SkorchDataset(x_val, y_val_for_skorch)
    splitter = sk_predefined_split(valid_ds)

    # 回调：早停 + 保存最优（按阶段目录）
    callbacks = [
        EarlyStopping(patience=15),
        SaveModel(disease=disease, seed=seed, phase_tag=phase_tag, model_type=model_type),
    ]

    net_cls = MVIBNeuralNet if model_type == "MVIB" else NeuralNetClassifier
    net = net_cls(
        model,
        max_epochs=200,
        criterion=criterion,
        lr=lr,
        iterator_train__shuffle=True,
        train_split=splitter,
        device=device,
        optimizer=torch.optim.AdamW,
        optimizer__weight_decay=1e-4,
        batch_size=batch_size,
        callbacks=callbacks
    )

    # 训练
    net.fit(x_train, y_train)
    save_training_history_csv(net, logpath, seed, model_type)

    # 从该阶段目录加载 best（若存在）
    ckpt_dir = eval_checkpoint_dir(disease, model_type, seed, phase_tag)
    try:
        net.load_params(
            f_params=os.path.join(ckpt_dir, "model_best.pkl"),
            f_optimizer=os.path.join(ckpt_dir, "optim_best.pkl"),
            f_history=os.path.join(ckpt_dir, "history_best.json"),
        )
    except Exception as e:
        print(f"[WARN] load_params 失败：{e}；使用当前权重继续评估。")

    # 评估
    scores, df = evaluate(net, x_val, y_val)
    record.update(scores)

    # 落盘（首次运行无 CSV；record 中可能含 YAML 列表如 hidden_dims，须先标量化）
    row = _record_row_for_results_csv(record)
    record_df = pd.DataFrame([row])
    if os.path.isfile(logpath):
        res_df = pd.read_csv(logpath)
        res_df = pd.concat([res_df, record_df], ignore_index=True)
    else:
        res_df = record_df
    res_df.to_csv(logpath, index=False)

    return scores
