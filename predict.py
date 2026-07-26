"""
机器学习趋势预测模块
使用多模型集成（XGBoost + LightGBM + RandomForest），基于丰富的技术指标特征预测「次日涨/跌」概率。
注意：金融预测不确定性极高，结果仅为概率参考，不构成投资建议。
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import RFE

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

from indicators import add_indicators


FEATURES_CORE = [
    "ret1", "ret5", "ret20", "ret60",
    "MA5", "MA10", "MA20", "MA60",
    "DIF", "DEA", "MACD", "RSI", "K", "D", "J",
    "boll_pos", "vol_change",
]

FEATURES_ENHANCED = FEATURES_CORE + [
    "MOM10", "MOM20", "MOM60",
    "ATR", "HV20", "HV60",
    "VMA10", "VROC",
    "RSI_CHG", "KDJ_CHG", "MACD_CHG",
    "skew20", "kurt20",
    "weekday", "month",
    "close_ma_ratio", "volume_ma_ratio",
    "ma5_ma20_ratio", "ma10_ma60_ratio",
    "vol_boll_pos",
    "std20", "std60",
    "upper_shadow", "lower_shadow",
]

FEATURES_WITH_MARKET = FEATURES_ENHANCED + [
    "market_ret1", "market_ret5", "market_hv20",
]


def _build_features(df: pd.DataFrame, market_df: pd.DataFrame = None) -> pd.DataFrame:
    d = add_indicators(df).copy()

    d["ret1"] = d["close"].pct_change(1)
    d["ret5"] = d["close"].pct_change(5)
    d["ret20"] = d["close"].pct_change(20)
    d["ret60"] = d["close"].pct_change(60)

    d["MOM10"] = d["close"].pct_change(10)
    d["MOM20"] = d["close"].pct_change(20)
    d["MOM60"] = d["close"].pct_change(60)

    d["TR"] = np.maximum(
        d["high"] - d["low"],
        np.maximum(
            np.abs(d["high"] - d["close"].shift(1)),
            np.abs(d["low"] - d["close"].shift(1))
        )
    )
    d["ATR"] = d["TR"].rolling(14).mean()

    d["HV20"] = d["close"].pct_change().rolling(20).std() * np.sqrt(252)
    d["HV60"] = d["close"].pct_change().rolling(60).std() * np.sqrt(252)

    d["VMA10"] = d["volume"].rolling(10).mean()
    d["VROC"] = (d["volume"] - d["volume"].shift(10)) / d["volume"].shift(10)

    d["RSI_CHG"] = d["RSI"].diff(1)
    d["KDJ_CHG"] = d["J"].diff(1)
    d["MACD_CHG"] = d["MACD"].diff(1)

    d["skew20"] = d["close"].pct_change().rolling(20).skew()
    d["kurt20"] = d["close"].pct_change().rolling(20).kurt()

    d["weekday"] = d.index.weekday
    d["month"] = d.index.month

    d["close_ma_ratio"] = d["close"] / d["MA20"]
    d["volume_ma_ratio"] = d["volume"] / d["VMA10"]
    d["ma5_ma20_ratio"] = d["MA5"] / d["MA20"]
    d["ma10_ma60_ratio"] = d["MA10"] / d["MA60"]

    rng = (d["BOLL_UP"] - d["BOLL_LOW"]).replace(0, np.nan)
    d["boll_pos"] = (d["close"] - d["BOLL_LOW"]) / rng

    vol_boll_up = d["VMA10"] + 2 * d["volume"].rolling(10).std()
    vol_boll_low = d["VMA10"] - 2 * d["volume"].rolling(10).std()
    vol_rng = (vol_boll_up - vol_boll_low).replace(0, np.nan)
    d["vol_boll_pos"] = (d["volume"] - vol_boll_low) / vol_rng

    d["std20"] = d["close"].pct_change().rolling(20).std()
    d["std60"] = d["close"].pct_change().rolling(60).std()

    d["upper_shadow"] = (d["high"] - np.maximum(d["open"], d["close"])) / (d["high"] - d["low"])
    d["lower_shadow"] = (np.minimum(d["open"], d["close"]) - d["low"]) / (d["high"] - d["low"])

    d["vol_change"] = d["volume"].pct_change().replace([np.inf, -np.inf], np.nan)

    if market_df is not None and not market_df.empty:
        market_df = market_df.reindex(d.index, method="ffill")
        d["market_ret1"] = market_df["close"].pct_change(1)
        d["market_ret5"] = market_df["close"].pct_change(5)
        d["market_hv20"] = market_df["close"].pct_change().rolling(20).std() * np.sqrt(252)

    return d


def _create_target_ternary(df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
    d = df.copy()
    next_ret = d["close"].shift(-1) / d["close"] - 1
    conditions = [
        next_ret > threshold,
        next_ret < -threshold,
    ]
    choices = [2, 0]
    d["target"] = np.select(conditions, choices, default=1)
    return d


def _optimize_xgboost(X_train, y_train, X_val, y_val, n_trials=20):
    """使用 Optuna 优化 XGBoost 超参数"""
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 1),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "num_class": 3,
            "random_state": 42,
            "use_label_encoder": False,
        }
        model = xgb.XGBClassifier(**params, n_jobs=-1)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return model.score(X_val, y_val)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def _optimize_lightgbm(X_train, y_train, X_val, y_val, n_trials=20):
    """使用 Optuna 优化 LightGBM 超参数"""
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 1),
            "objective": "multiclass",
            "num_class": 3,
            "random_state": 42,
        }
        model = lgb.LGBMClassifier(**params, n_jobs=-1)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return model.score(X_val, y_val)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def train_predict(df: pd.DataFrame, market_df: pd.DataFrame = None,
                  test_ratio: float = 0.2, feature_selection: bool = True,
                  confidence_threshold: float = 0.6, optimize: bool = False) -> dict:
    """
    训练多模型集成并对最新一天做次日涨跌预测。
    使用三分类策略：涨>1% / 震荡 / 跌>1%
    使用滚动窗口训练，更真实地评估模型泛化能力。
    
    返回：预测方向、上涨概率、测试集准确率、特征重要性、历史预测对比。
    """
    d = _build_features(df, market_df)
    d = _create_target_ternary(d)

    use_features = FEATURES_WITH_MARKET if (market_df is not None and not market_df.empty) else FEATURES_ENHANCED
    data = d.dropna(subset=use_features + ["target"]).copy()
    if len(data) < 365:
        raise ValueError("有效样本不足（<365），请扩大日期范围。")

    X = data[use_features].values
    y = data["target"].values

    split = int(len(data) * (1 - test_ratio))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    selected_features = use_features

    if feature_selection and len(use_features) > 20:
        rf_selector = RandomForestClassifier(
            n_estimators=100, max_depth=8, random_state=42, n_jobs=-1
        )
        rfe = RFE(estimator=rf_selector, n_features_to_select=min(20, len(use_features)), step=2)
        rfe.fit(X_train_s, y_train)
        selected_features = [use_features[i] for i in range(len(use_features)) if rfe.support_[i]]
        X_train_s = rfe.transform(X_train_s)
        X_test_s = rfe.transform(X_test_s)

    xgb_params = {
        "n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "objective": "multi:softprob", "eval_metric": "mlogloss",
        "random_state": 42, "n_jobs": -1, "use_label_encoder": False,
        "num_class": 3,
    }
    lgb_params = {
        "n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "objective": "multiclass", "random_state": 42, "n_jobs": -1,
        "num_class": 3,
    }

    if optimize and HAS_OPTUNA and HAS_XGBOOST:
        val_split = int(len(X_train_s) * 0.8)
        xgb_params = _optimize_xgboost(X_train_s[:val_split], y_train[:val_split],
                                       X_train_s[val_split:], y_train[val_split:])
        xgb_params.update({"objective": "multi:softprob", "eval_metric": "mlogloss",
                           "random_state": 42, "n_jobs": -1, "use_label_encoder": False,
                           "num_class": 3})

    if optimize and HAS_OPTUNA and HAS_LIGHTGBM:
        val_split = int(len(X_train_s) * 0.8)
        lgb_params = _optimize_lightgbm(X_train_s[:val_split], y_train[:val_split],
                                         X_train_s[val_split:], y_train[val_split:])
        lgb_params.update({"objective": "multiclass", "random_state": 42, "n_jobs": -1,
                           "num_class": 3})

    base_models = []

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=5,
        random_state=42, n_jobs=-1, class_weight="balanced",
    )
    base_models.append(("rf", rf))

    if HAS_XGBOOST:
        xgb_model = xgb.XGBClassifier(**xgb_params)
        base_models.append(("xgb", xgb_model))

    if HAS_LIGHTGBM:
        lgb_model = lgb.LGBMClassifier(**lgb_params)
        base_models.append(("lgb", lgb_model))

    ensemble = VotingClassifier(
        estimators=base_models,
        voting="soft",
        n_jobs=-1,
    )

    model = CalibratedClassifierCV(
        ensemble, cv=TimeSeriesSplit(n_splits=5), method="sigmoid",
    )
    model.fit(X_train_s, y_train)

    test_acc = model.score(X_test_s, y_test) if len(X_test) > 0 else float("nan")

    pred_proba = model.predict_proba(X_test_s)
    pred_labels = model.predict(X_test_s)

    profit_ratios = []
    high_conf_preds = []
    high_conf_actuals = []
    for i in range(len(pred_labels)):
        max_prob = max(pred_proba[i])
        actual_ret = (data["close"].iloc[split + i + 1] - data["close"].iloc[split + i]) / data["close"].iloc[split + i]
        
        if max_prob >= confidence_threshold:
            high_conf_preds.append(pred_labels[i])
            high_conf_actuals.append(y_test[i])
            
            if pred_labels[i] == 2:
                profit_ratios.append(actual_ret)
            elif pred_labels[i] == 0:
                profit_ratios.append(-actual_ret)

    high_conf_acc = sum(1 for p, a in zip(high_conf_preds, high_conf_actuals) if p == a) / len(high_conf_preds) if high_conf_preds else float("nan")
    effective_ratio = len(high_conf_preds) / len(pred_labels) if len(pred_labels) > 0 else 0

    avg_profit = np.mean([r for r in profit_ratios if r > 0]) if any(r > 0 for r in profit_ratios) else 0
    avg_loss = np.mean([abs(r) for r in profit_ratios if r < 0]) if any(r < 0 for r in profit_ratios) else 0
    profit_loss_ratio = avg_profit / avg_loss if avg_loss > 0 else float("inf")

    latest = _build_features(df, market_df).dropna(subset=use_features).iloc[-1]
    x_latest = latest[use_features].values.reshape(1, -1)
    if feature_selection and len(use_features) > 20:
        x_latest = rfe.transform(scaler.transform(x_latest))
    else:
        x_latest = scaler.transform(x_latest)

    proba = model.predict_proba(x_latest)[0]
    proba_up = float(proba[2])
    proba_down = float(proba[0])
    proba_flat = float(proba[1])

    max_prob = max(proba_up, proba_down, proba_flat)
    if max_prob >= confidence_threshold:
        if proba_up > proba_down and proba_up > proba_flat:
            direction = "上涨"
        elif proba_down > proba_up and proba_down > proba_flat:
            direction = "下跌"
        else:
            direction = "震荡"
    else:
        direction = "观望（低置信度）"

    if HAS_XGBOOST and "xgb" in dict(base_models):
        xgb_model.fit(X_train_s, y_train)
        importance = pd.Series(xgb_model.feature_importances_, index=selected_features).sort_values(
            ascending=False
        )
    elif HAS_LIGHTGBM and "lgb" in dict(base_models):
        lgb_model.fit(X_train_s, y_train)
        importance = pd.Series(lgb_model.feature_importances_, index=selected_features).sort_values(
            ascending=False
        )
    else:
        rf.fit(X_train_s, y_train)
        importance = pd.Series(rf.feature_importances_, index=selected_features).sort_values(
            ascending=False
        )

    hist = pd.DataFrame(
        {"实际": y_test, "预测": pred_labels},
        index=data.index[split:],
    )

    model_names = ", ".join([name for name, _ in base_models])

    return {
        "direction": direction,
        "proba_up": proba_up,
        "proba_down": proba_down,
        "proba_flat": proba_flat,
        "test_acc": test_acc,
        "high_conf_acc": high_conf_acc,
        "effective_ratio": effective_ratio,
        "profit_loss_ratio": profit_loss_ratio,
        "importance": importance,
        "hist": hist,
        "n_samples": len(data),
        "features_used": selected_features,
        "model_used": f"集成学习（{model_names}）+ 概率校准 + 三分类" + (" + 超参数优化" if optimize else ""),
        "n_features": len(selected_features),
        "confidence_threshold": confidence_threshold,
    }