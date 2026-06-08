# -*- coding: utf-8 -*-
"""XGBoost 二分类基线：与 MLP 相同输入——各模态拼接后的 (N, D) 数值矩阵。
需在环境中安装: pip install xgboost"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

_SKLEARN_KEYS = frozenset({
    "n_estimators", "max_depth", "max_leaves", "max_bin", "grow_policy",
    "learning_rate", "subsample", "colsample_bytree", "colsample_bylevel", "colsample_bynode",
    "reg_alpha", "reg_lambda", "gamma", "min_child_weight", "max_delta_step",
    "scale_pos_weight", "base_score", "random_state", "n_jobs", "tree_method",
    "importance_type", "max_cat_to_onehot", "max_cat_threshold", "enable_categorical",
    "eval_metric", "verbosity",
})

_CFG_SKIP = frozenset({
    "lr", "batch_size", "model_name", "model_type", "n_num_features", "in_dim",
    "hidden_dims", "dropout",
})


def fit_tabular_xgb_classifier(
    cfg: Dict[str, Any],
    seed: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
):
    """
    从 YAML/kwargs 过滤出 XGBClassifier 支持的参数，在验证集上做 early stopping 拟合。
    """
    try:
        from xgboost import XGBClassifier
    except ImportError as e:
        raise ImportError(
            "未安装 xgboost。请执行: pip install xgboost（或 pip install -r requirements_xgboost.txt）"
        ) from e

    d = {k: v for k, v in dict(cfg).items() if k not in _CFG_SKIP}
    early = int(d.pop("early_stopping_rounds", 20))

    kwargs = {k: v for k, v in d.items() if k in _SKLEARN_KEYS}
    kwargs.setdefault("n_estimators", 400)
    kwargs.setdefault("max_depth", 6)
    kwargs.setdefault("learning_rate", 0.05)
    kwargs.setdefault("subsample", 0.9)
    kwargs.setdefault("colsample_bytree", 0.9)
    kwargs.setdefault("reg_lambda", 1.0)
    kwargs.setdefault("reg_alpha", 0.0)
    kwargs.setdefault("min_child_weight", 1.0)
    kwargs.setdefault("n_jobs", -1)
    kwargs.setdefault("eval_metric", "logloss")
    kwargs["random_state"] = int(kwargs.get("random_state", seed))

    y_tr = np.asarray(y_train).squeeze().astype(int)
    y_va = np.asarray(y_val).squeeze().astype(int)

    clf_base: Dict[str, Any] = {"objective": "binary:logistic", **kwargs}
    fit_kw: Dict[str, Any] = {"eval_set": [(x_val, y_va)], "verbose": False}

    if early <= 0:
        clf = XGBClassifier(**clf_base)
        clf.fit(x_train, y_tr, **fit_kw)
        return clf

    from xgboost import callback as xgb_callback

    # 不同 xgboost 版本差异大：用尝试顺序兼容（避免依赖 fit 签名里的 **kwargs 误判）
    # 1) 当前文档推荐：early_stopping_rounds 在构造函数（fit 只传 eval_set）
    try:
        clf = XGBClassifier(**clf_base, early_stopping_rounds=early)
        clf.fit(x_train, y_tr, **fit_kw)
        return clf
    except TypeError:
        pass

    # 2) 旧版：early_stopping_rounds 仅在 fit
    try:
        clf = XGBClassifier(**clf_base)
        clf.fit(x_train, y_tr, early_stopping_rounds=early, **fit_kw)
        return clf
    except TypeError:
        pass

    # 3) callbacks 在构造函数
    try:
        clf = XGBClassifier(
            **clf_base,
            callbacks=[xgb_callback.EarlyStopping(rounds=early, save_best=True)],
        )
        clf.fit(x_train, y_tr, **fit_kw)
        return clf
    except TypeError:
        pass

    # 4) callbacks 在 fit（更老的 callback 路径）
    clf = XGBClassifier(**clf_base)
    clf.fit(
        x_train,
        y_tr,
        callbacks=[xgb_callback.EarlyStopping(rounds=early, save_best=True)],
        **fit_kw,
    )
    return clf
