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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold
from statsmodels.nonparametric.smoothers_lowess import lowess

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_STATE = 42
TARGET = "cv_event"

PREDICTOR_SPECS = [
    {"name": "age", "sources": ["age", "Age", "ηλικία"]},
    {"name": "history_hf", "sources": ["history_hf", "heart_failure", "hf", "ΚΑ"]},
    {"name": "history_af", "sources": ["history_af", "atrial_fibrillation", "af", "AF"]},
    {"name": "ph", "sources": ["ph", "pH", "Phεισόδου"]},
    {"name": "urea", "sources": ["urea", "Urea", "ουρία εισόδου"]},
    {"name": "lactate", "sources": ["lactate", "Lactate", "LAC"]},
]

EVENT_CANDIDATES = {
    "supraventricular_arrhythmia": ["supraventricular_arrhythmia", "ΑΡΡΥΘΜΙΑ"],
    "acute_pulmonary_edema": ["acute_pulmonary_edema", "ΟΠΟ"],
    "pulmonary_embolism": ["pulmonary_embolism", "ΠΕ"],
    "myocardial_infarction": ["myocardial_infarction", "ΟΕΜ.1", "ΟΕΜ"],
}

RAW_FEATURES = ["age", "history_hf", "history_af", "ph", "urea", "lactate"]


def set_plot_style() -> None:
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


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    by_name = {str(c).strip(): c for c in df.columns}
    for candidate in candidates:
        if candidate in by_name:
            return by_name[candidate]
    return None


def parse_binary(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.where(values.isin([0, 1]), np.nan)


def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return normalize_columns(pd.read_excel(path))
    if suffix == ".csv":
        return normalize_columns(pd.read_csv(path))
    raise ValueError("Input file must be .xlsx, .xls, or .csv")


def build_analysis_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    predictors = pd.DataFrame(index=raw.index)
    for spec in PREDICTOR_SPECS:
        source = first_existing(raw, spec["sources"])
        if source is None:
            raise ValueError(f"Missing predictor column for {spec['name']}. Accepted names: {', '.join(spec['sources'])}")
        if spec["name"] in {"history_hf", "history_af"}:
            predictors[spec["name"]] = parse_binary(raw[source])
        else:
            predictors[spec["name"]] = pd.to_numeric(raw[source], errors="coerce")

    events = pd.DataFrame(index=raw.index)
    for event_name, candidates in EVENT_CANDIDATES.items():
        source = first_existing(raw, candidates)
        if source is None:
            events[event_name] = 0
        else:
            values = pd.to_numeric(raw[source], errors="coerce").fillna(0)
            events[event_name] = (values > 0).astype(int)

    analysis = predictors.copy()
    for column in events.columns:
        analysis[column] = events[column]

    analysis[TARGET] = (
        events["supraventricular_arrhythmia"].astype(int)
        | events["acute_pulmonary_edema"].astype(int)
        | events["pulmonary_embolism"].astype(int)
        | events["myocardial_infarction"].astype(int)
    ).astype(int)

    return analysis


def fit_model(X: pd.DataFrame, y: pd.Series):
    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
    model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=5000, random_state=RANDOM_STATE)
    model.fit(X_imp, y)
    pred = pd.Series(model.predict_proba(X_imp)[:, 1], index=X.index, name="predicted_probability")
    return model, imputer, pred


def predict_model(model: LogisticRegression, imputer: SimpleImputer, X: pd.DataFrame) -> pd.Series:
    X_imp = pd.DataFrame(imputer.transform(X), columns=X.columns, index=X.index)
    return pd.Series(model.predict_proba(X_imp)[:, 1], index=X.index, name="predicted_probability")


def repeated_oof_predictions(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE)
    pred_sum = pd.Series(0.0, index=X.index)
    pred_count = pd.Series(0.0, index=X.index)

    for train_idx, test_idx in cv.split(X, y):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        model, imputer, _ = fit_model(X_train, y_train)
        pred_test = predict_model(model, imputer, X_test)
        pred_sum.loc[X_test.index] += pred_test
        pred_count.loc[X_test.index] += 1.0

    out = pred_sum / pred_count.replace(0, np.nan)
    out.name = "predicted_probability"
    return out


def bootstrap_auc_ci(y: np.ndarray, p: np.ndarray, n_boot: int = 2000, seed: int = RANDOM_STATE) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    values = []
    n = len(y)

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        pb = p[idx]
        if np.unique(yb).size < 2:
            continue
        values.append(roc_auc_score(yb, pb))

    if not values:
        return np.nan, np.nan
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def calibration_stats(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    p = np.clip(np.asarray(y_prob, dtype=float), 1e-8, 1 - 1e-8)
    y = np.asarray(y_true, dtype=int)
    logit_p = np.log(p / (1 - p))
    X = sm.add_constant(logit_p, has_constant="add")
    model = sm.Logit(y, X).fit(disp=False, maxiter=1000)
    params = np.asarray(model.params, dtype=float)
    return float(params[0]), float(params[1])


def wilson_ci(k: np.ndarray, n: np.ndarray, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    p = np.divide(k, n, out=np.zeros_like(k), where=n > 0)
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt((p * (1 - p) / n) + (z**2 / (4 * n**2)))) / denom
    return np.clip(center - half, 0, 1), np.clip(center + half, 0, 1)


def net_benefit(y_true: np.ndarray, pred_prob: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(pred_prob, dtype=float)
    n = len(y)
    values = np.full(len(thresholds), np.nan)

    for i, threshold in enumerate(thresholds):
        pred_pos = p >= threshold
        tp = np.sum(pred_pos & (y == 1))
        fp = np.sum(pred_pos & (y == 0))
        values[i] = (tp / n) - (fp / n) * (threshold / (1 - threshold))

    return values


def treat_all_net_benefit(y_true: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    prevalence = np.mean(np.asarray(y_true, dtype=int))
    return prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))


def moving_average(values: np.ndarray, window: int = 5) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if window <= 1 or len(arr) < window:
        return arr.copy()
    kernel = np.ones(window) / window
    pad = window // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def bootstrap_net_benefit_ci(y: np.ndarray, p: np.ndarray, thresholds: np.ndarray, n_boot: int = 300, seed: int = RANDOM_STATE):
    rng = np.random.default_rng(seed)
    n = len(y)
    boots = np.zeros((n_boot, len(thresholds)), dtype=float)

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b, :] = net_benefit(y[idx], p[idx], thresholds)

    return np.mean(boots, axis=0), np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)


def save_model_files(model: LogisticRegression, imputer: SimpleImputer, outdir: Path) -> None:
    coef = pd.DataFrame({
        "variable": ["intercept"] + RAW_FEATURES,
        "beta": [float(model.intercept_[0])] + [float(x) for x in model.coef_.ravel()],
    })
    medians = pd.DataFrame({
        "variable": RAW_FEATURES,
        "median": [float(x) for x in imputer.statistics_],
    })
    coef.to_csv(outdir / "model_coefficients.csv", index=False)
    medians.to_csv(outdir / "model_imputation_medians.csv", index=False)


def make_roc_plot(y: pd.Series, pred: pd.Series, outpath: Path, table_out: Path) -> dict[str, float]:
    auc = roc_auc_score(y, pred)
    lo, hi = bootstrap_auc_ci(y.values, pred.values, seed=RANDOM_STATE + 101)
    fpr, tpr, _ = roc_curve(y, pred)

    plt.figure(figsize=(6.9, 6.0))
    plt.plot(fpr, tpr, linewidth=2.2, label=f"AUC {auc:.3f}, 95% CI {lo:.3f}–{hi:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Internal validation ROC curve")
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()

    metrics = {"auc": float(auc), "auc_ci_low": float(lo), "auc_ci_high": float(hi)}
    pd.DataFrame([metrics]).to_csv(table_out, index=False)
    return metrics


def make_calibration_plot(y: pd.Series, pred: pd.Series, outpath: Path, table_out: Path) -> dict[str, float]:
    p = np.clip(np.asarray(pred, dtype=float), 1e-6, 1 - 1e-6)
    y_arr = np.asarray(y, dtype=int)
    df = pd.DataFrame({"y": y_arr, "pred": p}).sort_values("pred").reset_index(drop=True)

    x_max = min(0.60, float(np.quantile(df["pred"], 0.98)))
    df_plot = df[df["pred"] <= x_max].copy()
    if len(df_plot) < 30:
        df_plot = df.copy()
        x_max = min(0.60, float(df["pred"].max()))

    frac = 0.72
    smooth = lowess(df_plot["y"], df_plot["pred"], frac=frac, it=0, return_sorted=True)
    grid = np.linspace(float(df_plot["pred"].min()), float(df_plot["pred"].max()), 160)
    rng = np.random.default_rng(RANDOM_STATE + 1701)
    curves = []
    n = len(df_plot)

    for _ in range(180):
        idx = rng.integers(0, n, size=n)
        boot = df_plot.iloc[idx].sort_values("pred")
        smooth_boot = lowess(boot["y"], boot["pred"], frac=frac, it=0, return_sorted=True)
        xb = np.asarray(smooth_boot[:, 0], dtype=float)
        yb = np.asarray(smooth_boot[:, 1], dtype=float)
        keep = np.unique(xb, return_index=True)[1]
        x_unique = xb[np.sort(keep)]
        y_unique = yb[np.sort(keep)]
        if len(x_unique) >= 2:
            curves.append(np.interp(grid, x_unique, y_unique, left=y_unique[0], right=y_unique[-1]))

    cal_intercept, cal_slope = calibration_stats(y_arr, p)

    plt.figure(figsize=(6.9, 6.0))
    plt.plot([0, x_max], [0, x_max], linestyle="--", linewidth=1.2, label="Ideal calibration")
    if len(curves) > 20:
        curves_arr = np.asarray(curves)
        lo = np.clip(np.percentile(curves_arr, 2.5, axis=0), 0, 0.70)
        hi = np.clip(np.percentile(curves_arr, 97.5, axis=0), 0, 0.70)
        plt.fill_between(grid, lo, hi, alpha=0.10, label="LOWESS 95% CI")
    plt.plot(smooth[:, 0], smooth[:, 1], linewidth=2.4, label="LOWESS-smoothed calibration")
    plt.xlim(0, x_max)
    plt.ylim(0, 0.70)
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed event rate")
    plt.title("Internal validation calibration plot")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()

    metrics = {"calibration_intercept": float(cal_intercept), "calibration_slope": float(cal_slope), "x_max_plot": float(x_max)}
    pd.DataFrame([metrics]).to_csv(table_out, index=False)
    return metrics


def make_dca_plot(y: pd.Series, pred: pd.Series, outpath: Path, table_out: Path) -> None:
    thresholds = np.arange(0.10, 0.50 + 1e-9, 0.01)
    mid, lo, hi = bootstrap_net_benefit_ci(y.values, pred.values, thresholds, n_boot=300, seed=RANDOM_STATE + 1)
    treat_all = treat_all_net_benefit(y.values, thresholds)
    treat_none = np.zeros_like(thresholds)

    mid_s = moving_average(mid, 5)
    lo_s = moving_average(lo, 5)
    hi_s = moving_average(hi, 5)
    treat_all_s = moving_average(treat_all, 5)

    plt.figure(figsize=(8, 6.8))
    plt.fill_between(thresholds, lo_s, hi_s, alpha=0.10, label="Model 95% CI")
    plt.plot(thresholds, mid_s, linewidth=2.2, label="Model")
    plt.plot(thresholds, treat_all_s, linewidth=2.0, label="Treat all")
    plt.axhline(0, linestyle="--", linewidth=1.5, label="Treat none")
    plt.xlim(0.10, 0.50)
    plt.xlabel("Threshold probability")
    plt.ylabel("Net benefit")
    plt.title("Decision curve analysis")
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()

    pd.DataFrame({
        "threshold": thresholds,
        "model_nb": mid,
        "model_nb_low": lo,
        "model_nb_high": hi,
        "treat_all_nb": treat_all,
        "treat_none_nb": treat_none,
    }).to_csv(table_out, index=False)


def make_quartile_plot(y: pd.Series, pred: pd.Series, outpath: Path, table_out: Path) -> None:
    df = pd.DataFrame({"y": y, "pred": pred}).dropna().copy()
    try:
        df["quartile"] = pd.qcut(df["pred"], q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
    except ValueError:
        df["quartile"] = pd.cut(df["pred"], bins=4, labels=["Q1", "Q2", "Q3", "Q4"], include_lowest=True)

    g = df.groupby("quartile", observed=False).agg(
        n=("y", "size"),
        events=("y", "sum"),
        event_rate=("y", "mean"),
        mean_pred=("pred", "mean"),
        pred_min=("pred", "min"),
        pred_max=("pred", "max"),
    ).reset_index()
    g["ci_low"], g["ci_high"] = wilson_ci(g["events"].values, g["n"].values)

    x = np.arange(len(g))
    plt.figure(figsize=(6.9, 5.8))
    plt.errorbar(
        x,
        g["event_rate"],
        yerr=[g["event_rate"] - g["ci_low"], g["ci_high"] - g["event_rate"]],
        fmt="o",
        capsize=4,
        linewidth=1.5,
        markersize=8,
    )
    plt.plot(x, g["event_rate"], linewidth=1.7, alpha=0.9)
    plt.xticks(x, g["quartile"])
    plt.xlabel("Predicted-risk quartile")
    plt.ylabel("Observed event rate")
    plt.title("Observed event rate across predicted-risk quartiles")
    plt.ylim(0, max(0.40, float(g["ci_high"].max()) + 0.05))
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()
    g.to_csv(table_out, index=False)


def make_composite_figure(paths: list[Path], outpath: Path) -> None:
    imgs = [Image.open(path).convert("RGB") for path in paths]
    width = max(img.width for img in imgs)
    height = max(img.height for img in imgs)
    canvas = Image.new("RGB", (2 * width, 2 * height), "white")
    labels = ["(A)", "(B)", "(C)", "(D)"]
    draw = ImageDraw.Draw(canvas)

    for idx, img in enumerate(imgs):
        x = (idx % 2) * width
        y = (idx // 2) * height
        canvas.paste(img.resize((width, height)), (x, y))
        draw.text((x + 10, y + 10), labels[idx], fill="black")

    canvas.save(outpath)


def main() -> None:
    parser = argparse.ArgumentParser(description="Internal validation for the AECOPD-CV prediction model")
    parser.add_argument("--input", required=True, help="Path to the input Excel or CSV file")
    parser.add_argument("--output", default="outputs/internal_validation", help="Output directory")
    args = parser.parse_args()

    set_plot_style()
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    raw = load_table(Path(args.input))
    analysis = build_analysis_dataset(raw)
    work = analysis[analysis[TARGET].isin([0, 1])].copy()
    y = work[TARGET].astype(int)
    X = work[RAW_FEATURES].copy()

    pred = repeated_oof_predictions(X, y)
    final_model, final_imputer, _ = fit_model(X, y)
    save_model_files(final_model, final_imputer, outdir)

    auc_metrics = make_roc_plot(y, pred, outdir / "figure_roc_internal_validation.png", outdir / "table_roc_internal_validation.csv")
    cal_metrics = make_calibration_plot(y, pred, outdir / "figure_calibration_internal_validation.png", outdir / "table_calibration_internal_validation.csv")
    make_dca_plot(y, pred, outdir / "figure_decision_curve_internal_validation.png", outdir / "table_decision_curve_internal_validation.csv")
    make_quartile_plot(y, pred, outdir / "figure_quartiles_internal_validation.png", outdir / "table_quartiles_internal_validation.csv")

    try:
        make_composite_figure(
            [
                outdir / "figure_roc_internal_validation.png",
                outdir / "figure_decision_curve_internal_validation.png",
                outdir / "figure_calibration_internal_validation.png",
                outdir / "figure_quartiles_internal_validation.png",
            ],
            outdir / "figure_internal_validation_composite.png",
        )
    except Exception:
        pass

    patient_level = work.copy()
    patient_level["predicted_probability"] = pred
    patient_level.to_csv(outdir / "patient_level_internal_validation_predictions.csv", index=False)

    summary = {
        "n_total": int(len(work)),
        "n_events": int(y.sum()),
        "event_rate": float(y.mean()),
        "auc": auc_metrics["auc"],
        "auc_ci_low": auc_metrics["auc_ci_low"],
        "auc_ci_high": auc_metrics["auc_ci_high"],
        "brier_score": float(brier_score_loss(y, pred)),
        "calibration_intercept": cal_metrics["calibration_intercept"],
        "calibration_slope": cal_metrics["calibration_slope"],
    }

    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Outputs written to: {outdir}")


if __name__ == "__main__":
    main()
