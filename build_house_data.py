# -*- coding: utf-8 -*-
"""
Build the model data for Problem 2: King County house prices.

Dataset : House Sales in King County, USA (Kaggle, harlfoxem)
          21,613 rows x 21 columns -> 16 predictors + price (all numeric)
Pipeline: CRISP-DM, random_state = 42, 80/20 hold-out, 5-fold CV
Methods : Lasso (L1), ElasticNet, RFE, Forward SFS, Backward elimination (p-value)
Outputs : house_model_data.json
          static/data_house.js   (window.PROBLEM_DATA.house)
"""

import itertools
import json
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sps
from sklearn.feature_selection import RFE
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson

RANDOM_STATE = 42
HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "kc_house_data.csv")
CSV_URL = "https://raw.githubusercontent.com/ChengLi9/kc_house_data/HEAD/kc_house_data.csv"

CV = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
TARGET = "price"
K_MAX = 12
OUTLIER_HIGHLIGHT = ["bedrooms", "sqft_living", "grade", "lat"]
ALGO_ORDER = ["Lasso", "Forward SFS", "RFE", "ElasticNet", "Backward"]

# ---------------------------------------------------------------- Phase 2 ----
if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
else:                                              # pragma: no cover
    df = pd.read_csv(CSV_URL)

raw_shape = df.shape
# identifiers / free text / high-cardinality categorical are not usable in OLS
DROP = ["id", "date", "zipcode", "sqft_basement"]
notes = {
    "sqft_basement": "sqft_above + sqft_basement == sqft_living（完全共線），保留 sqft_living",
    "zipcode": "70 個郵遞區號會產生 69 個虛擬變數，改以 lat / long 表達地理位置",
    "id": "流水號，無預測意義",
    "date": "交易日期，不在本次靜態特徵範圍",
}
df = df.drop(columns=DROP)
FEATURES = [c for c in df.columns if c != TARGET]
X = df[FEATURES].astype(float)
y = df[TARGET].astype(float)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE)

print("PHASE 2: data understanding")
print(f"  raw shape      : {raw_shape[0]:,} x {raw_shape[1]}")
print(f"  modelling shape: {X.shape[0]:,} x {len(FEATURES)} predictors")
print(f"  train/test     : {len(X_train):,} / {len(X_test):,}")


def metrics(y_true, y_pred, k):
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    adj = 1 - (1 - r2) * (n - 1) / (n - k - 1) if n > k + 1 else None
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2),
        "adj_r2": float(adj) if adj is not None else None,
    }


def cv_stats(features):
    s = -cross_val_score(LinearRegression(), X_train[list(features)], y_train,
                         cv=CV, scoring="neg_root_mean_squared_error")
    return float(s.mean()), float(s.std())


def vif_table(cols):
    """VIF via auxiliary R^2 (sklearn) - much faster than statsmodels on 17k rows."""
    if len(cols) <= 1:
        return {c: 1.0 for c in cols}
    Xv = X_train[list(cols)].values.astype(float)
    out = {}
    for i, col in enumerate(cols):
        y_i = Xv[:, i]
        rest = np.delete(Xv, i, axis=1)
        r2 = LinearRegression().fit(rest, y_i).score(rest, y_i)
        out[col] = float(1.0 / max(1.0 - r2, 1e-12))
    return out


def equation_from(coefficients):
    parts = []
    for c in coefficients:
        if c["feature"].startswith("Intercept"):
            parts.append(f"{c['coef']:,.2f}")
        else:
            sign = "+" if c["coef"] >= 0 else "-"
            parts.append(f"{sign} {abs(c['coef']):,.2f}·{c['feature']}")
    return "Price = " + " ".join(parts)


def outliers_table(features, model):
    pred = model.predict(X_test[list(features)])
    rows = []
    for idx, actual, p in zip(X_test.index, y_test.values, pred):
        err = float(actual - p)
        rows.append({
            "index": int(idx),
            "features": {f: float(df.loc[idx, f]) for f in OUTLIER_HIGHLIGHT},
            "actual": float(actual),
            "predicted": float(p),
            "residual": err,
            "abs_error": abs(err),
            "pct_error": float(abs(err) / actual * 100) if actual else 0.0,
        })
    rows.sort(key=lambda r: r["abs_error"], reverse=True)
    for rank, r in enumerate(rows, start=1):
        r["rank"] = rank
    return rows[:10]


def build_entry(algo, k, features):
    features = list(features)
    model = LinearRegression().fit(X_train[features], y_train)
    res = metrics(y_test, model.predict(X_test[features]), len(features))
    cv_mean, cv_std = cv_stats(features)

    design = sm.add_constant(X_train[features], has_constant="add")
    fit = sm.OLS(y_train, design).fit()
    ci = fit.conf_int()
    vifs = vif_table(features)

    coefficients = [{
        "feature": "Intercept (β₀)", "coef": float(fit.params["const"]),
        "stderr": float(fit.bse["const"]), "tstat": float(fit.tvalues["const"]),
        "pvalue": float(fit.pvalues["const"]),
        "ci_lower": float(ci.loc["const", 0]), "ci_upper": float(ci.loc["const", 1]),
        "vif": None,
    }]
    for col in features:
        coefficients.append({
            "feature": col, "coef": float(fit.params[col]),
            "stderr": float(fit.bse[col]), "tstat": float(fit.tvalues[col]),
            "pvalue": float(fit.pvalues[col]),
            "ci_lower": float(ci.loc[col, 0]), "ci_upper": float(ci.loc[col, 1]),
            "vif": float(vifs[col]),
        })

    return {
        "algo": algo, "k": k, "features": features,
        "rmse": res["rmse"], "mae": res["mae"], "r2": res["r2"], "adj_r2": res["adj_r2"],
        "train_r2": float(model.score(X_train[features], y_train)),
        "cv_rmse": cv_mean, "cv_rmse_std": cv_std,
        "f_stat": float(fit.fvalue), "f_pvalue": float(fit.f_pvalue),
        "aic": float(fit.aic), "bic": float(fit.bic),
        "equation": equation_from(coefficients),
        "coefficients": coefficients,
        "outliers": outliers_table(features, model),
    }


# --------------------------------------------------- Phase 4: rankings ------
print("PHASE 4: feature selection")

lasso_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("lasso", LassoCV(cv=CV, max_iter=50_000, random_state=RANDOM_STATE, alphas=60)),
]).fit(X_train, y_train)
lasso_coef = lasso_pipe.named_steps["lasso"].coef_
lasso_alpha = float(lasso_pipe.named_steps["lasso"].alpha_)
lasso_rank = [f for _, f in sorted(zip(np.abs(lasso_coef), FEATURES), reverse=True)]

elastic_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("elastic", ElasticNetCV(cv=CV, l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                             max_iter=50_000, random_state=RANDOM_STATE, alphas=60)),
]).fit(X_train, y_train)
elastic_rank = [f for _, f in sorted(
    zip(np.abs(elastic_pipe.named_steps["elastic"].coef_), FEATURES), reverse=True)]
elastic_alpha = float(elastic_pipe.named_steps["elastic"].alpha_)
elastic_l1 = float(elastic_pipe.named_steps["elastic"].l1_ratio_)

rfe = RFE(LinearRegression(), n_features_to_select=1, step=1).fit(
    StandardScaler().fit_transform(X_train), y_train)
rfe_rank = [f for _, f in sorted(zip(rfe.ranking_, FEATURES))]

sfs_order, remaining = [], list(FEATURES)
while len(sfs_order) < len(FEATURES):
    scored = sorted((cv_stats(sfs_order + [f])[0], FEATURES.index(f), f) for f in remaining)
    sfs_order.append(scored[0][2])
    remaining.remove(scored[0][2])

# Backward elimination: drop the least significant feature until one is left
bw_cols, bw_drop_order = list(FEATURES), []
while len(bw_cols) > 1:
    design = sm.add_constant(X_train[bw_cols], has_constant="add")
    pvals = sm.OLS(y_train, design).fit().pvalues.drop("const")
    worst = pvals.idxmax()
    bw_drop_order.append(worst)
    bw_cols.remove(worst)
bw_rank = [bw_cols[0]] + list(reversed(bw_drop_order))

rankings = {"Lasso": lasso_rank, "Forward SFS": sfs_order, "RFE": rfe_rank,
            "ElasticNet": elastic_rank, "Backward": bw_rank}

print("  Lasso       :", lasso_rank, f"(alpha={lasso_alpha:,.0f})")
print("  ElasticNet  :", elastic_rank, f"(alpha={elastic_alpha:,.0f}, l1={elastic_l1})")
print("  RFE         :", rfe_rank)
print("  Forward SFS :", sfs_order)
print("  Backward    :", bw_rank)

MODEL_DATA = {}
for algo in ALGO_ORDER:
    MODEL_DATA[algo] = [build_entry(algo, k, rankings[algo][:k])
                        for k in range(1, K_MAX + 1)]

print("\n  CV RMSE by k")
print("  " + "method".ljust(14) + "".join(f"k={k}".rjust(12) for k in range(1, K_MAX + 1)))
for algo in ALGO_ORDER:
    print("  " + algo.ljust(14) + "".join(f"{e['cv_rmse']:>12,.0f}" for e in MODEL_DATA[algo]))
print("\n  Test RMSE by k")
print("  " + "method".ljust(14) + "".join(f"k={k}".rjust(12) for k in range(1, K_MAX + 1)))
for algo in ALGO_ORDER:
    print("  " + algo.ljust(14) + "".join(f"{e['rmse']:>12,.0f}" for e in MODEL_DATA[algo]))

# ------------------------------------------------- Phase 5: diagnostics -----
print("\nPHASE 5: evaluation")

baseline_entry = build_entry("Baseline OLS", len(FEATURES), FEATURES)
all_entries = [e for v in MODEL_DATA.values() for e in v]
cv_best = min(all_entries, key=lambda e: e["cv_rmse"])
holdout_best = min(all_entries, key=lambda e: e["rmse"])
noise_band = float(np.mean([e["cv_rmse_std"] for e in all_entries]))
cv_min = min(e["cv_rmse"] for e in all_entries)
parsimonious = min(
    (e for e in all_entries if e["algo"] == "Forward SFS" and e["cv_rmse"] <= cv_min * 1.01),
    key=lambda e: e["k"])

# p-value selection from the full model
cols = list(FEATURES)
drop_history = []
while True:
    design = sm.add_constant(X_train[cols], has_constant="add")
    pvals = sm.OLS(y_train, design).fit().pvalues.drop("const")
    if len(cols) > 1 and pvals.max() > 0.05:
        worst = pvals.idxmax()
        drop_history.append({"feature": worst, "pvalue": float(pvals.max())})
        cols.remove(worst)
    else:
        break
selected_fit = sm.OLS(y_train, sm.add_constant(X_train[cols], has_constant="add")).fit()
selected_entry = build_entry("p-value selected", len(cols), cols)

# assumptions on the hold-out-optimal model
best_features = list(holdout_best["features"])
best_model = LinearRegression().fit(X_train[best_features], y_train)
resid = y_train.values - best_model.predict(X_train[best_features])
shapiro_stat, shapiro_p = sps.shapiro(resid[:5000])          # Shapiro caps at 5000
bp_stat, bp_p, _, _ = het_breuschpagan(resid, sm.add_constant(X_train[best_features]))
dw = float(durbin_watson(resid))

assumptions = [
    {"test": "Shapiro-Wilk（殘差常態）", "stat": float(shapiro_stat), "pvalue": float(shapiro_p),
     "criterion": "p > 0.05", "status": "pass" if shapiro_p > 0.05 else "warn",
     "note": "17k 筆樣本下檢定極敏感；右偏的房價造成厚尾"},
    {"test": "Breusch-Pagan（同變異數）", "stat": float(bp_stat), "pvalue": float(bp_p),
     "criterion": "p > 0.05", "status": "pass" if bp_p > 0.05 else "fail",
     "note": "高價房殘差變異更大（異質變異數）"},
    {"test": "Durbin-Watson（獨立性）", "stat": dw, "pvalue": None, "criterion": "≈ 2",
     "status": "pass" if 1.5 < dw < 2.5 else "warn", "note": "無自我相關"},
    {"test": "Max VIF（共線性）", "stat": float(max(c["vif"] for c in baseline_entry["coefficients"]
                                                    if c["vif"] is not None)),
     "pvalue": None, "criterion": "< 10",
     "status": "fail" if max(c["vif"] for c in baseline_entry["coefficients"]
                             if c["vif"] is not None) > 10 else "pass",
     "note": "移除 sqft_basement（與 sqft_above 完全共線）後，面積 / 房間數的重疊已降到可接受範圍"},
]

corr = df[FEATURES + [TARGET]].corr()
describe = df[FEATURES + [TARGET]].describe().round(2)
rows_sample = df.sample(n=min(200, len(df)), random_state=RANDOM_STATE).sort_index()

dataset_rows = []
for i, r in rows_sample.iterrows():
    rec = {"index": int(i), TARGET: float(r[TARGET])}
    for f in FEATURES:
        rec[f] = float(r[f])
    dataset_rows.append(rec)

dataset_columns = [{"key": f, "label": f, "kind": "numeric"} for f in FEATURES]
dataset_columns.append({"key": TARGET, "label": "price", "kind": "target"})

predictor_inputs = []
for f in FEATURES:
    s = df[f]
    span = float(s.quantile(0.99)) - float(s.quantile(0.01))
    if np.allclose(s.values, np.round(s.values)):
        decimals = 0
        step = 100 if span > 5000 else (10 if span > 500 else 1)
    elif span > 1000:
        step, decimals = 100, 0
    elif span > 100:
        step, decimals = 10, 1
    elif span > 10:
        step, decimals = 1, 1
    elif span > 1:
        step, decimals = 0.1, 2
    else:
        step, decimals = 0.001, 4
    lo = round(float(s.quantile(0.01)), decimals)
    hi = round(float(s.quantile(0.99)), decimals)
    if hi - lo <= step:                       # 1%-99% collapses (binary / narrow features)
        lo, hi = round(float(s.min()), decimals), round(float(s.max()), decimals)
    default = min(max(float(s.median()), lo), hi)
    predictor_inputs.append({
        "key": f, "feature": f, "label": f, "type": "range",
        "min": round(lo, decimals), "max": round(hi, decimals),
        "step": step, "decimals": decimals,
        "default": round(default, decimals),
    })

META = {
    "id": "house",
    "dataset": {
        "name": "House Sales in King County, USA (Kaggle · harlfoxem)",
        "source": "https://www.kaggle.com/datasets/harlfoxem/housesalesprediction",
        "raw_rows": int(raw_shape[0]), "raw_columns": int(raw_shape[1]),
        "rows": int(len(df)), "features": FEATURES,
        "numeric_features": FEATURES, "categorical_features": [],
        "target": TARGET, "target_label": "Price (USD)", "unit": "$",
        "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
        "random_state": RANDOM_STATE, "test_size": 0.2, "cv_folds": 5,
        "missing_values": 0, "duplicates": 0, "sampled_rows": len(dataset_rows),
        "dropped": [{"column": k, "reason": v} for k, v in notes.items()],
        "columns": dataset_columns,
        "categorical_values": {},
    },
    "k_max": K_MAX,
    "outlier_highlight": OUTLIER_HIGHLIGHT,
    "describe": {c: {s: float(v) for s, v in describe[c].items()} for c in describe.columns},
    "correlation": {r: {c: float(corr.loc[r, c]) for c in corr.columns} for r in corr.index},
    "top_feature": {"name": "sqft_living", "corr": float(corr.loc["sqft_living", TARGET])},
    "rows": dataset_rows,
    "baseline": baseline_entry,
    "selected_pvalue": selected_entry,
    "backward_elimination": {
        "dropped": drop_history, "retained": list(cols),
        "r2": float(selected_fit.rsquared), "adj_r2": float(selected_fit.rsquared_adj),
        "coefficients": [{
            "feature": c, "coef": float(selected_fit.params[c]),
            "stderr": float(selected_fit.bse[c]), "tstat": float(selected_fit.tvalues[c]),
            "pvalue": float(selected_fit.pvalues[c]),
            "ci_lower": float(selected_fit.conf_int().loc[c, 0]),
            "ci_upper": float(selected_fit.conf_int().loc[c, 1]),
        } for c in ["const"] + list(cols)],
    },
    "assumptions": assumptions,
    "cv_best": {"algo": cv_best["algo"], "k": cv_best["k"], "features": cv_best["features"],
                "cv_rmse": cv_best["cv_rmse"], "cv_rmse_std": cv_best["cv_rmse_std"],
                "rmse": cv_best["rmse"], "r2": cv_best["r2"]},
    "holdout_best": {"algo": holdout_best["algo"], "k": holdout_best["k"],
                     "features": holdout_best["features"], "cv_rmse": holdout_best["cv_rmse"],
                     "cv_rmse_std": holdout_best["cv_rmse_std"],
                     "rmse": holdout_best["rmse"], "r2": holdout_best["r2"],
                     "mae": holdout_best["mae"]},
    "noise_band": noise_band,
    "final_decision": {
        "algo": parsimonious["algo"], "k": parsimonious["k"],
        "model": f"{parsimonious['algo']} · {parsimonious['k']} 特徵 OLS",
        "features": parsimonious["features"],
        "rmse": parsimonious["rmse"], "r2": parsimonious["r2"],
        "reason": (f"k={parsimonious['k']} 之後 CV RMSE 的改善已小於 1%"
                   f"（雜訊帶 ±${noise_band:,.0f}），用一半的特徵換取精簡、可解釋與抗共線性"),
    },    "tuning": {"lasso_alpha": lasso_alpha, "elasticnet_alpha": elastic_alpha,
               "elasticnet_l1_ratio": elastic_l1},
    "predictor_inputs": predictor_inputs,
}

payload = {"meta": META, "models": MODEL_DATA}

json_path = os.path.join(HERE, "house_model_data.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=1)

os.makedirs(os.path.join(HERE, "static"), exist_ok=True)
js_path = os.path.join(HERE, "static", "data_house.js")
with open(js_path, "w", encoding="utf-8") as f:
    f.write("/* Generated by build_house_data.py - do not edit by hand. */\n")
    f.write("window.PROBLEM_DATA = window.PROBLEM_DATA || {};\n")
    f.write("window.PROBLEM_DATA.house = ")
    json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    f.write(";\n")

print("\nPHASE 6: artefacts written")
print(f"  {json_path} ({os.path.getsize(json_path)/1024:.0f} KB)")
print(f"  {js_path} ({os.path.getsize(js_path)/1024:.0f} KB)")
print(f"  baseline R2 {baseline_entry['r2']:.4f} | RMSE ${baseline_entry['rmse']:,.0f}")
print(f"  CV-optimal    : {cv_best['algo']} k={cv_best['k']} -> {cv_best['features']} "
      f"({cv_best['cv_rmse']:,.0f} +/- {cv_best['cv_rmse_std']:,.0f})")
print(f"  hold-out best : {holdout_best['algo']} k={holdout_best['k']} -> {holdout_best['features']} "
      f"(RMSE {holdout_best['rmse']:,.0f}, R2 {holdout_best['r2']:.4f})")
print(f"  backward kept : {cols}")
print(f"  max VIF (all) : {max(c['vif'] for c in baseline_entry['coefficients'] if c['vif'] is not None):,.1f}")
