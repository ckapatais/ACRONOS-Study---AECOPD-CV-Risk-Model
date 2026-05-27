from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from PIL import Image, ImageDraw
from scipy.stats import chi2
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from statsmodels.nonparametric.smoothers_lowess import lowess

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_STATE = 42
TARGET = "cv_event"

PREDICTOR_SPECS = [
    {"name": "age", "source": "ηλικία", "label": "Age (per 10-year increase)", "scale": 10.0},
    {"name": "history_hf", "source": "ΚΑ", "label": "History of heart failure", "scale": 1.0},
    {"name": "history_af", "source": "AF", "label": "History of atrial fibrillation", "scale": 1.0},
    {"name": "ph", "source": "Phεισόδου", "label": "Arterial pH (per 0.1-unit increase)", "scale": 0.1},
    {"name": "urea", "source": "ουρία εισόδου", "label": "Urea (per unit increase)", "scale": 1.0},
    {"name": "lactate", "source": "LAC", "label": "Lactate (per unit increase)", "scale": 1.0},
]

EVENT_CANDIDATES = {
    "supraventricular_arrhythmia": ["supraventricular_arrhythmia", "ΑΡΡΥΘΜΙΑ"],
    "acute_pulmonary_edema": ["acute_pulmonary_edema", "ΟΠΟ"],
    "pulmonary_embolism": ["pulmonary_embolism", "ΠΕ"],
    "myocardial_infarction": ["myocardial_infarction", "ΟΕΜ.1", "ΟΕΜ"],
}


def nice_style() -> None:
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 16,
        "axes.labelsize": 12,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 10,
        "axes.linewidth": 0.9,
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "legend.frameon": False,
    })


def first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = {str(c).strip(): c for c in df.columns}
    for cand in candidates:
        if cand in cols:
            return cols[cand]
    return None


def load_excel(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def build_analysis_dataset(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred_df = pd.DataFrame(index=raw.index)
    pred_rows = []

    for spec in PREDICTOR_SPECS:
        src = first_existing(raw, [spec["source"]])
        if src is None:
            raise ValueError(f"Missing predictor column: {spec['source']}")

        pred_df[spec["name"]] = pd.to_numeric(raw[src], errors="coerce")
        pred_rows.append({
            "model_variable": spec["name"],
            "source_column": src,
            "plot_label": spec["label"],
            "scale_for_or": spec["scale"],
            "status": "used",
        })

    event_df = pd.DataFrame(index=raw.index)
    event_rows = []

    for model_var, candidates in EVENT_CANDIDATES.items():
        src = first_existing(raw, candidates)
        if src is None:
            event_df[model_var] = 0
            event_rows.append({
                "event_variable": model_var,
                "source_column": "",
                "status": "not_found_defaulted_to_0",
            })
        else:
            s = pd.to_numeric(raw[src], errors="coerce").fillna(0)
            event_df[model_var] = (s > 0).astype(int)
            event_rows.append({
                "event_variable": model_var,
                "source_column": src,
                "status": "used",
            })

    analysis = pred_df.copy()
    for col in event_df.columns:
        analysis[col] = event_df[col]

    analysis[TARGET] = (
        event_df["supraventricular_arrhythmia"].astype(int)
        | event_df["acute_pulmonary_edema"].astype(int)
        | event_df["pulmonary_embolism"].astype(int)
        | event_df["myocardial_infarction"].astype(int)
    ).astype(int)

    return analysis, pd.DataFrame(pred_rows), pd.DataFrame(event_rows)


def fit_logit_with_median_imputation(df: pd.DataFrame, predictors: list[str]):
    use = df[[TARGET] + predictors].copy()

    for col in use.columns:
        use[col] = pd.to_numeric(use[col], errors="coerce")

    use = use[use[TARGET].isin([0, 1])].copy()

    y = use[TARGET].astype(int)
    X = use[predictors].copy()

    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(
        imputer.fit_transform(X),
        columns=predictors,
        index=X.index,
    )

    X_sm = sm.add_constant(X_imp, has_constant="add")
    model = sm.Logit(y, X_sm).fit(disp=False, maxiter=2000)

    pred = pd.Series(
        model.predict(X_sm),
        index=X.index,
        name="predicted_probability",
    )

    return model, imputer, y, X_imp, pred


def predict_from_model(model, imputer, df: pd.DataFrame, predictors: list[str]) -> tuple[pd.Series, pd.Series]:
    X = df[predictors].copy()
    y = df[TARGET].astype(int)

    X_imp = pd.DataFrame(
        imputer.transform(X),
        columns=predictors,
        index=X.index,
    )

    X_sm = sm.add_constant(X_imp, has_constant="add")

    pred = pd.Series(
        model.predict(X_sm),
        index=X.index,
        name="predicted_probability",
    )

    return y, pred


def bootstrap_auc_ci(y: np.ndarray, p: np.ndarray, n_boot: int = 2000, seed: int = RANDOM_STATE):
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)

    vals = []
    n = len(y)

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        pb = p[idx]

        if np.unique(yb).size < 2:
            continue

        vals.append(roc_auc_score(yb, pb))

    if not vals:
        return np.nan, np.nan

    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def auc_roc_ci(y: pd.Series, score: pd.Series, seed: int):
    tmp = pd.DataFrame({"y": y, "score": score}).dropna()

    auc = roc_auc_score(tmp["y"], tmp["score"])
    ci_lo, ci_hi = bootstrap_auc_ci(
        tmp["y"].values,
        tmp["score"].values,
        n_boot=2000,
        seed=seed,
    )

    fpr, tpr, _ = roc_curve(tmp["y"], tmp["score"])

    return {
        "auc": auc,
        "ci_low": ci_lo,
        "ci_high": ci_hi,
        "fpr": fpr,
        "tpr": tpr,
        "n_used": int(len(tmp)),
    }


def calibration_stats(y_true: np.ndarray, y_prob: np.ndarray):
    eps = 1e-8
    p = np.clip(np.asarray(y_prob, dtype=float), eps, 1 - eps)

    logit_p = np.log(p / (1 - p))
    X = sm.add_constant(logit_p, has_constant="add")

    model = sm.Logit(np.asarray(y_true, dtype=int), X).fit(
        disp=False,
        maxiter=1000,
    )

    vals = np.asarray(model.params, dtype=float)

    return float(vals[0]), float(vals[1])


def fit_logistic_recalibration(y_true: np.ndarray, y_prob: np.ndarray):
    eps = 1e-8
    p = np.clip(np.asarray(y_prob, dtype=float), eps, 1 - eps)

    lp = np.log(p / (1 - p))
    X = sm.add_constant(lp, has_constant="add")

    model = sm.Logit(np.asarray(y_true, dtype=int), X).fit(
        disp=False,
        maxiter=1000,
    )

    vals = np.asarray(model.params, dtype=float)
    intercept, slope = float(vals[0]), float(vals[1])

    recal_lp = intercept + slope * lp
    recal_prob = 1.0 / (1.0 + np.exp(-recal_lp))

    return intercept, slope, pd.Series(recal_prob, index=np.arange(len(y_prob)))


def hosmer_lemeshow_test(y_true: np.ndarray, y_prob: np.ndarray, groups: int = 10):
    df = pd.DataFrame({"y": y_true, "p": y_prob}).dropna().copy()

    n_unique = df["p"].nunique()
    g = min(groups, max(2, int(n_unique)))

    if len(df) < g * 5:
        return np.nan, np.nan, g

    try:
        df["bin"] = pd.qcut(df["p"], q=g, duplicates="drop")
    except Exception:
        return np.nan, np.nan, g

    tab = df.groupby("bin", observed=False).agg(
        obs=("y", "sum"),
        exp=("p", "sum"),
        n=("y", "size"),
    )

    if len(tab) < 2:
        return np.nan, np.nan, len(tab)

    obs_non = tab["n"] - tab["obs"]
    exp_non = tab["n"] - tab["exp"]
    eps = 1e-9

    hl = (
        ((tab["obs"] - tab["exp"]) ** 2) / (tab["exp"] + eps)
        + ((obs_non - exp_non) ** 2) / (exp_non + eps)
    ).sum()

    df_hl = max(len(tab) - 2, 1)
    p = 1 - chi2.cdf(float(hl), df_hl)

    return float(hl), float(p), int(len(tab))


def wilson_ci(k: np.ndarray, n: np.ndarray, z: float = 1.96):
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)

    p = np.divide(k, n, out=np.zeros_like(k), where=n > 0)

    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (
        z * np.sqrt((p * (1 - p) / n) + (z**2 / (4 * n**2)))
    ) / denom

    lo = np.clip(center - half, 0, 1)
    hi = np.clip(center + half, 0, 1)

    return lo, hi


def coefficient_table(model) -> pd.DataFrame:
    params = model.params
    conf = model.conf_int()

    return pd.DataFrame({
        "variable": params.index.astype(str),
        "beta": params.values.astype(float),
        "or": np.exp(params.values.astype(float)),
        "ci_low": np.exp(conf[0].values.astype(float)),
        "ci_high": np.exp(conf[1].values.astype(float)),
        "p_value": model.pvalues.values.astype(float),
    })


def rescale_coef_df_for_plot(coef_df: pd.DataFrame) -> pd.DataFrame:
    scale_map = {spec["name"]: spec["scale"] for spec in PREDICTOR_SPECS}
    label_map = {spec["name"]: spec["label"] for spec in PREDICTOR_SPECS}

    rows = []

    for _, row in coef_df.iterrows():
        var = row["variable"]

        if var == "const":
            continue

        beta = float(row["beta"])
        lo = float(np.log(row["ci_low"]))
        hi = float(np.log(row["ci_high"]))
        scale = scale_map.get(var, 1.0)

        beta_scaled = beta * scale
        lo_scaled = lo * scale
        hi_scaled = hi * scale

        rows.append({
            "variable": var,
            "label": label_map.get(var, var),
            "beta_scaled": beta_scaled,
            "or": float(np.exp(beta_scaled)),
            "ci_low": float(np.exp(lo_scaled)),
            "ci_high": float(np.exp(hi_scaled)),
            "p_value": float(row["p_value"]),
        })

    return pd.DataFrame(rows)


def make_forest_plot(coef_plot_df: pd.DataFrame, outpath: Path):
    order = [spec["name"] for spec in PREDICTOR_SPECS]
    order_map = {v: i for i, v in enumerate(order)}

    df = coef_plot_df.copy()
    df["plot_order"] = df["variable"].map(order_map)
    df = df.sort_values("plot_order").copy()

    y = np.arange(len(df))[::-1]

    plt.figure(figsize=(9.2, 5.9))
    plt.axvline(1.0, linestyle="--", linewidth=1.2)

    plt.errorbar(
        df["or"].astype(float),
        y,
        xerr=[
            df["or"].astype(float) - df["ci_low"].astype(float),
            df["ci_high"].astype(float) - df["or"].astype(float),
        ],
        fmt="o",
        capsize=3,
        linewidth=1.4,
        markersize=8,
    )

    plt.yticks(y, df["label"])
    plt.xlabel("Adjusted odds ratio (95% CI)")
    plt.title("Multivariable model coefficients")

    xmin = max(0.1, float(df["ci_low"].astype(float).min()) * 0.9)
    xmax = float(df["ci_high"].astype(float).max()) * 1.08
    plt.xlim(xmin, xmax)

    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()


def make_single_roc_plot(y: pd.Series, pred: pd.Series, outpath: Path, table_out: Path):
    res = auc_roc_ci(y, pred, seed=RANDOM_STATE + 101)

    plt.figure(figsize=(6.8, 6.0))
    plt.plot(
        res["fpr"],
        res["tpr"],
        linewidth=2.2,
        label=f"AECOPD-CV model (AUC {res['auc']:.3f}, 95% CI {res['ci_low']:.3f}–{res['ci_high']:.3f})",
    )
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Internal validation ROC comparison")
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()

    pd.DataFrame([{
        "model": "AECOPD-CV model",
        "auc": res["auc"],
        "auc_ci_low": res["ci_low"],
        "auc_ci_high": res["ci_high"],
        "n_used": res["n_used"],
    }]).to_csv(table_out, index=False)

    return res


def make_original_vs_recalibrated_roc_plot(y: pd.Series, pred_original: pd.Series, pred_recal: pd.Series, outpath: Path, table_out: Path):
    orig_auc = auc_roc_ci(y, pred_original, seed=RANDOM_STATE + 201)
    recal_auc = auc_roc_ci(y, pred_recal, seed=RANDOM_STATE + 202)

    plt.figure(figsize=(6.9, 6.0))
    plt.plot(
        orig_auc["fpr"],
        orig_auc["tpr"],
        linewidth=2.2,
        label=f"Original model (AUC {orig_auc['auc']:.3f}, 95% CI {orig_auc['ci_low']:.3f}–{orig_auc['ci_high']:.3f})",
    )
    plt.plot(
        recal_auc["fpr"],
        recal_auc["tpr"],
        linewidth=2.2,
        label=f"Calibrated prediction (AUC {recal_auc['auc']:.3f}, 95% CI {recal_auc['ci_low']:.3f}–{recal_auc['ci_high']:.3f})",
    )
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Internal validation ROC: original vs calibrated")
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()

    pd.DataFrame([
        {
            "model": "original",
            "auc": orig_auc["auc"],
            "auc_ci_low": orig_auc["ci_low"],
            "auc_ci_high": orig_auc["ci_high"],
        },
        {
            "model": "calibrated",
            "auc": recal_auc["auc"],
            "auc_ci_low": recal_auc["ci_low"],
            "auc_ci_high": recal_auc["ci_high"],
        },
    ]).to_csv(table_out, index=False)

    return orig_auc, recal_auc


def make_single_calibration_plot(y_true: pd.Series, y_prob: pd.Series, outpath: Path, table_out: Path, recal_intercept: float, recal_slope: float):
    pred = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y_true, dtype=int)

    df = pd.DataFrame({"y": y, "pred": pred}).sort_values("pred").reset_index(drop=True)

    x_max_plot = min(0.60, float(np.quantile(df["pred"], 0.98)))
    df_plot = df[df["pred"] <= x_max_plot].copy()

    if len(df_plot) < 30:
        df_plot = df.copy()
        x_max_plot = min(0.60, float(df["pred"].max()))

    frac = 0.72
    lo = lowess(df_plot["y"], df_plot["pred"], frac=frac, it=0, return_sorted=True)

    grid = np.linspace(
        float(df_plot["pred"].min()),
        float(df_plot["pred"].max()),
        160,
    )

    rng = np.random.default_rng(RANDOM_STATE + 1702)
    boot_curves = []
    n = len(df_plot)

    for _ in range(160):
        idx = rng.integers(0, n, size=n)
        boot = df_plot.iloc[idx].sort_values("pred")
        lo_b = lowess(boot["y"], boot["pred"], frac=frac, it=0, return_sorted=True)

        x_b = np.asarray(lo_b[:, 0], dtype=float)
        y_b = np.asarray(lo_b[:, 1], dtype=float)

        uniq_idx = np.unique(x_b, return_index=True)[1]
        x_u = x_b[np.sort(uniq_idx)]
        y_u = y_b[np.sort(uniq_idx)]

        if len(x_u) < 2:
            continue

        boot_curves.append(
            np.interp(grid, x_u, y_u, left=y_u[0], right=y_u[-1])
        )

    cal_int, cal_slope = calibration_stats(y, pred)
    hl_stat, hl_p, hl_groups = hosmer_lemeshow_test(y, pred)
    brier = brier_score_loss(y, pred)

    plt.figure(figsize=(6.9, 6.0))
    plt.plot(
        [0, x_max_plot],
        [0, x_max_plot],
        linestyle="--",
        linewidth=1.2,
        label="Ideal calibration",
    )

    if len(boot_curves) > 20:
        boot_curves = np.asarray(boot_curves)
        lo_ci = np.clip(np.percentile(boot_curves, 2.5, axis=0), 0, 0.70)
        hi_ci = np.clip(np.percentile(boot_curves, 97.5, axis=0), 0, 0.70)
        plt.fill_between(grid, lo_ci, hi_ci, alpha=0.08, label="LOWESS 95% CI")

    plt.plot(lo[:, 0], lo[:, 1], linewidth=2.4, label="LOWESS-smoothed calibration")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed event rate")
    plt.title("Internal validation calibration plot\nAECOPD-CV model")
    plt.xlim(0, x_max_plot)
    plt.ylim(0, 0.70)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()

    row = {
        "model": "aecopd_cv_model",
        "calibration_intercept": cal_int,
        "calibration_slope": cal_slope,
        "brier_score": brier,
        "hosmer_lemeshow_stat": hl_stat,
        "hosmer_lemeshow_p": hl_p,
        "hosmer_lemeshow_groups": hl_groups,
        "x_max_plot": x_max_plot,
        "lowess_frac": frac,
        "n_used": int(len(y_true)),
        "events": int(np.sum(y)),
        "recalibration_intercept": recal_intercept,
        "recalibration_slope": recal_slope,
    }

    pd.DataFrame([row]).to_csv(table_out, index=False)

    return row


def make_quartile_plot(pred_df: pd.DataFrame, outpath: Path, summary_out: Path):
    df = pred_df.copy().dropna(subset=["predicted_probability", TARGET])
    df["quartile"] = pd.qcut(
        df["predicted_probability"],
        q=4,
        labels=["Q1", "Q2", "Q3", "Q4"],
        duplicates="drop",
    )

    g = df.groupby("quartile", observed=False).agg(
        n=(TARGET, "size"),
        events=(TARGET, "sum"),
        event_rate=(TARGET, "mean"),
        mean_pred=("predicted_probability", "mean"),
    ).reset_index()

    g["ci_low"], g["ci_high"] = wilson_ci(g["events"].values, g["n"].values)
    g.to_csv(summary_out, index=False)

    x = np.arange(len(g))

    plt.figure(figsize=(6.4, 5.2))
    plt.errorbar(
        x,
        g["event_rate"],
        yerr=[
            g["event_rate"] - g["ci_low"],
            g["ci_high"] - g["event_rate"],
        ],
        fmt="o",
        capsize=4,
        linewidth=1.5,
        markersize=8,
    )
    plt.plot(x, g["event_rate"], linewidth=1.7, alpha=0.9)
    plt.xticks(x, g["quartile"])
    plt.ylabel("Observed event rate")
    plt.xlabel("Predicted-risk quartile")
    plt.title("Observed event rate across predicted-risk quartiles")
    plt.ylim(0, max(0.40, float(g["ci_high"].max()) + 0.05))
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()


def net_benefit(y_true: np.ndarray, pred_prob: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(pred_prob).astype(float)

    n = len(y)
    out = np.full(len(thresholds), np.nan)

    for i, pt in enumerate(thresholds):
        pred_pos = p >= pt
        tp = np.sum((pred_pos == 1) & (y == 1))
        fp = np.sum((pred_pos == 1) & (y == 0))
        out[i] = (tp / n) - (fp / n) * (pt / (1 - pt))

    return out


def treat_all_net_benefit(y_true: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    prevalence = np.mean(np.asarray(y_true).astype(int))
    return prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))


def smooth_ma(y: np.ndarray, window: int = 5) -> np.ndarray:
    arr = np.asarray(y, dtype=float)

    if window <= 1 or len(arr) < window:
        return arr.copy()

    kernel = np.ones(window) / window
    pad = window // 2
    padded = np.pad(arr, (pad, pad), mode="edge")

    return np.convolve(padded, kernel, mode="valid")


def bootstrap_net_benefit_ci(y: np.ndarray, p: np.ndarray, thresholds: np.ndarray, n_boot: int = 200, seed: int = RANDOM_STATE):
    rng = np.random.default_rng(seed)
    n = len(y)
    boots = np.zeros((n_boot, len(thresholds)), dtype=float)

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b, :] = net_benefit(y[idx], p[idx], thresholds)

    lo = np.percentile(boots, 2.5, axis=0)
    hi = np.percentile(boots, 97.5, axis=0)
    mid = np.mean(boots, axis=0)

    return mid, lo, hi


def make_dca_plot(y: pd.Series, pred: pd.Series, outdir: Path):
    thresholds = np.arange(0.10, 0.50 + 1e-9, 0.01)

    tmp = pd.DataFrame({"y": y, "p": pred}).dropna()

    mid, lo, hi = bootstrap_net_benefit_ci(
        tmp["y"].values,
        tmp["p"].values,
        thresholds,
        n_boot=200,
        seed=42,
    )

    treat_all_nb = treat_all_net_benefit(tmp["y"].values, thresholds)
    treat_none_nb = np.zeros_like(thresholds)

    plt.figure(figsize=(8, 6.8))
    plt.fill_between(
        thresholds,
        smooth_ma(lo, 5),
        smooth_ma(hi, 5),
        alpha=0.12,
        label="AECOPD-CV model 95% CI",
    )
    plt.plot(
        thresholds,
        smooth_ma(mid, 5),
        linewidth=2.4,
        label="AECOPD-CV model",
    )
    plt.plot(
        thresholds,
        smooth_ma(treat_all_nb, 5),
        linewidth=2.0,
        label="Treat all",
    )
    plt.axhline(0, linestyle="--", linewidth=1.6, label="Treat none")
    plt.xlim(0.10, 0.50)
    plt.xlabel("Threshold probability")
    plt.ylabel("Net benefit")
    plt.title("Decision curve analysis")
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(outdir / "decision_curve_analysis.png", dpi=300)
    plt.close()

    pd.DataFrame({
        "threshold": thresholds,
        "model_nb": mid,
        "model_nb_low": lo,
        "model_nb_high": hi,
        "treat_all_nb": treat_all_nb,
        "treat_none_nb": treat_none_nb,
    }).to_csv(outdir / "decision_curve_analysis_values.csv", index=False)


def make_composite_figure(roc_path: Path, dca_path: Path, cal_path: Path, quart_path: Path, outpath: Path):
    imgs = [Image.open(p).convert("RGB") for p in [roc_path, dca_path, cal_path, quart_path]]

    min_w = min(img.width for img in imgs)
    resized = []

    for img in imgs:
        h = int(img.height * (min_w / img.width))
        resized.append(img.resize((min_w, h)))

    top_h = max(resized[0].height, resized[1].height)
    bot_h = max(resized[2].height, resized[3].height)

    canvas = Image.new("RGB", (min_w * 2, top_h + bot_h), color="white")

    for idx, img in enumerate(resized):
        x = (idx % 2) * min_w
        y = 0 if idx < 2 else top_h
        canvas.paste(img, (x, y))

    canvas.save(outpath)


def main():
    parser = argparse.ArgumentParser(
        description="Internal validation for the AECOPD-CV model using median imputation and held-out calibration assessment."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to Anonymized data.xlsx",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output folder",
    )

    args = parser.parse_args()

    nice_style()

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    raw = load_excel(Path(args.input))
    analysis, predictor_mapping, event_mapping = build_analysis_dataset(raw)

    predictors = [spec["name"] for spec in PREDICTOR_SPECS]

    analysis.to_csv(outdir / "analysis_dataset_internal_validation.csv", index=False)
    predictor_mapping.to_csv(outdir / "variable_mapping_internal_validation.csv", index=False)
    event_mapping.to_csv(outdir / "event_mapping_internal_validation.csv", index=False)

    avail_rows = []

    for spec in PREDICTOR_SPECS:
        s = pd.to_numeric(analysis[spec["name"]], errors="coerce")
        avail_rows.append({
            "variable": spec["name"],
            "source_column": spec["source"],
            "n_non_missing": int(s.notna().sum()),
            "pct_non_missing": float(100 * s.notna().mean()),
            "n_unique_non_missing": int(s.dropna().nunique()),
        })

    pd.DataFrame(avail_rows).to_csv(
        outdir / "variable_availability_internal_validation.csv",
        index=False,
    )

    work = analysis[analysis[TARGET].isin([0, 1])].copy()

    train_df, test_df = train_test_split(
        work,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=work[TARGET],
    )

    model_train, imputer_train, _, _, _ = fit_logit_with_median_imputation(
        train_df,
        predictors,
    )

    test_y, test_pred = predict_from_model(
        model_train,
        imputer_train,
        test_df,
        predictors,
    )

    recal_intercept, recal_slope, test_pred_recal = fit_logistic_recalibration(
        test_y.values,
        test_pred.values,
    )
    test_pred_recal.index = test_pred.index
    test_pred_recal.name = "predicted_probability_recalibrated"

    roc_path = outdir / "figure_roc_comparison_internal_validation.png"
    roc_res = make_single_roc_plot(
        test_y,
        test_pred_recal,
        roc_path,
        outdir / "table_roc_comparison_internal_validation.csv",
    )

    orig_auc, recal_auc = make_original_vs_recalibrated_roc_plot(
        test_y,
        test_pred,
        test_pred_recal,
        outdir / "figure_roc_original_vs_recalibrated.png",
        outdir / "table_roc_original_vs_recalibrated.csv",
    )

    cal_path = outdir / "figure_calibration_internal_validation.png"
    cal_row = make_single_calibration_plot(
        test_y,
        test_pred_recal,
        cal_path,
        outdir / "table_calibration_internal_validation.csv",
        recal_intercept,
        recal_slope,
    )

    full_model, full_imputer, _, _, _ = fit_logit_with_median_imputation(
        work,
        predictors,
    )
    coef_df = coefficient_table(full_model)
    coef_df.to_csv(outdir / "model_coefficients_for_figures.csv", index=False)

    coef_plot_df = rescale_coef_df_for_plot(coef_df)
    coef_plot_df.to_csv(outdir / "table_coefficients_exact.csv", index=False)

    make_forest_plot(
        coef_plot_df,
        outdir / "figure_coefficients_forest.png",
    )

    quart_df = pd.DataFrame({
        TARGET: test_y,
        "predicted_probability": test_pred_recal,
    })
    quart_path = outdir / "figure_quartiles_event_rate.png"
    make_quartile_plot(
        quart_df,
        quart_path,
        outdir / "table_quartiles_event_rate.csv",
    )

    make_dca_plot(
        test_y,
        test_pred_recal,
        outdir,
    )

    dca_path = outdir / "decision_curve_analysis.png"

    try:
        make_composite_figure(
            roc_path,
            dca_path,
            cal_path,
            quart_path,
            outdir / "Internal_Validation_Composite.png",
        )
    except Exception:
        pass

    patient_level = test_df[predictors + [TARGET]].copy()
    patient_level["predicted_probability_original"] = test_pred
    patient_level["predicted_probability_recalibrated"] = test_pred_recal
    patient_level.to_csv(outdir / "patient_level_figure_inputs.csv", index=False)

    pd.DataFrame([{
        "n_total": int(len(work)),
        "n_events": int(work[TARGET].sum()),
        "event_rate": float(work[TARGET].mean()),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "events_test": int(test_y.sum()),
        "test_auc_original": float(orig_auc["auc"]),
        "test_auc_original_ci_low": float(orig_auc["ci_low"]),
        "test_auc_original_ci_high": float(orig_auc["ci_high"]),
        "test_auc_recalibrated": float(recal_auc["auc"]),
        "test_auc_recalibrated_ci_low": float(recal_auc["ci_low"]),
        "test_auc_recalibrated_ci_high": float(recal_auc["ci_high"]),
        "recalibration_intercept": float(recal_intercept),
        "recalibration_slope": float(recal_slope),
        "calibration_intercept": float(cal_row["calibration_intercept"]),
        "calibration_slope": float(cal_row["calibration_slope"]),
        "brier_score": float(cal_row["brier_score"]),
        "hosmer_lemeshow_stat": float(cal_row["hosmer_lemeshow_stat"]),
        "hosmer_lemeshow_p": float(cal_row["hosmer_lemeshow_p"]),
    }]).to_csv(outdir / "summary_internal_validation.csv", index=False)

    with open(outdir / "recalibration_parameters.json", "w", encoding="utf-8") as f:
        json.dump({
            "recalibration_intercept": float(recal_intercept),
            "recalibration_slope": float(recal_slope),
        }, f, indent=2)

    print(f"Internal-validation outputs written to: {outdir}")
    print("Main files:")
    print("- figure_roc_comparison_internal_validation.png")
    print("- figure_roc_original_vs_recalibrated.png")
    print("- figure_calibration_internal_validation.png")
    print("- decision_curve_analysis.png")
    print("- figure_coefficients_forest.png")
    print("- figure_quartiles_event_rate.png")
    print("- Internal_Validation_Composite.png")
    print("- summary_internal_validation.csv")


if __name__ == "__main__":
    main()
