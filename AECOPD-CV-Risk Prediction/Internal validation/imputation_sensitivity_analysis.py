from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from PIL import Image
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold
from statsmodels.nonparametric.smoothers_lowess import lowess

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_STATE = 42
TARGET = "cv_event"

PREDICTOR_SPECS = [
    {"name": "age", "source": "ηλικία"},
    {"name": "history_hf", "source": "ΚΑ"},
    {"name": "history_af", "source": "AF"},
    {"name": "ph", "source": "Phεισόδου"},
    {"name": "urea", "source": "ουρία εισόδου"},
    {"name": "lactate", "source": "LAC"},
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
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 9.5,
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
    cols = {str(c).strip(): c for c in df.columns}
    for cand in candidates:
        if cand in cols:
            return cols[cand]
    return None


def parse_binary(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    return x.where(x.isin([0, 1]), np.nan)


def load_table(path: Path, sheet: str | None = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return normalize_columns(pd.read_excel(path, sheet_name=sheet or 0))
    if suffix == ".csv":
        return normalize_columns(pd.read_csv(path))
    raise ValueError("Input file must be .xlsx, .xls, or .csv")


def build_analysis_dataset(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred_df = pd.DataFrame(index=raw.index)
    pred_rows = []
    for spec in PREDICTOR_SPECS:
        src = first_existing(raw, [spec["source"]])
        if src is None:
            raise ValueError(f"Missing predictor column: {spec['source']}")
        if spec["name"] in {"history_hf", "history_af"}:
            pred_df[spec["name"]] = parse_binary(raw[src])
        else:
            pred_df[spec["name"]] = pd.to_numeric(raw[src], errors="coerce")
        pred_rows.append({
            "model_variable": spec["name"],
            "source_column": src,
            "status": "used",
        })

    event_df = pd.DataFrame(index=raw.index)
    event_rows = []
    for event_name, candidates in EVENT_CANDIDATES.items():
        src = first_existing(raw, candidates)
        if src is None:
            event_df[event_name] = 0
            event_rows.append({
                "event_variable": event_name,
                "source_column": "",
                "status": "not_found_defaulted_to_0",
            })
        else:
            s = pd.to_numeric(raw[src], errors="coerce").fillna(0)
            event_df[event_name] = (s > 0).astype(int)
            event_rows.append({
                "event_variable": event_name,
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


def make_logistic_model() -> LogisticRegression:
    return LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=5000,
        random_state=RANDOM_STATE,
    )


def make_iterative_imputer(seed: int, sample_posterior: bool) -> IterativeImputer:
    return IterativeImputer(
        estimator=BayesianRidge(),
        max_iter=20,
        sample_posterior=sample_posterior,
        random_state=seed,
        initial_strategy="median",
    )


def add_missing_indicators(X: pd.DataFrame) -> pd.DataFrame:
    out = X.copy()
    for col in ["ph", "urea", "lactate"]:
        out[f"{col}_missing"] = out[col].isna().astype(int)
    return out


def fit_predict_single_imputer(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    imputer,
) -> tuple[pd.Series, LogisticRegression]:
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns, index=X_test.index)
    model = make_logistic_model()
    model.fit(X_train_imp, y_train)
    pred = pd.Series(model.predict_proba(X_test_imp)[:, 1], index=X_test.index)
    return pred, model


def repeated_oof_predictions(
    X: pd.DataFrame,
    y: pd.Series,
    method: str,
    n_splits: int,
    n_repeats: int,
    n_imputations: int,
) -> pd.Series:
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE)
    pred_sum = pd.Series(0.0, index=X.index)
    pred_count = pd.Series(0.0, index=X.index)

    for fold_no, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        X_train = X.iloc[train_idx].copy()
        y_train = y.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()

        if method == "median":
            pred, _ = fit_predict_single_imputer(X_train, y_train, X_test, SimpleImputer(strategy="median"))
        elif method == "mean":
            pred, _ = fit_predict_single_imputer(X_train, y_train, X_test, SimpleImputer(strategy="mean"))
        elif method == "most_frequent":
            pred, _ = fit_predict_single_imputer(X_train, y_train, X_test, SimpleImputer(strategy="most_frequent"))
        elif method == "knn":
            pred, _ = fit_predict_single_imputer(X_train, y_train, X_test, KNNImputer(n_neighbors=5, weights="uniform"))
        elif method == "iterative_single":
            pred, _ = fit_predict_single_imputer(
                X_train,
                y_train,
                X_test,
                make_iterative_imputer(RANDOM_STATE + fold_no, sample_posterior=False),
            )
        elif method == "multiple_imputation":
            fold_pred_sum = pd.Series(0.0, index=X_test.index)
            for m in range(n_imputations):
                imp = make_iterative_imputer(RANDOM_STATE + 1000 * fold_no + m, sample_posterior=True)
                pred_m, _ = fit_predict_single_imputer(X_train, y_train, X_test, imp)
                fold_pred_sum += pred_m
            pred = fold_pred_sum / float(n_imputations)
        elif method == "median_plus_missing_indicators":
            pred, _ = fit_predict_single_imputer(
                add_missing_indicators(X_train),
                y_train,
                add_missing_indicators(X_test),
                SimpleImputer(strategy="median"),
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        pred_sum.loc[X_test.index] += pred
        pred_count.loc[X_test.index] += 1.0

    out = pred_sum / pred_count.replace(0, np.nan)
    out.name = method
    return out


def complete_case_oof(X: pd.DataFrame, y: pd.Series, n_splits: int, n_repeats: int) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    mask = X.notna().all(axis=1)
    X_cc = X.loc[mask].copy()
    y_cc = y.loc[mask].copy()
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE)
    pred_sum = pd.Series(0.0, index=X_cc.index)
    pred_count = pd.Series(0.0, index=X_cc.index)
    for train_idx, test_idx in cv.split(X_cc, y_cc):
        X_train = X_cc.iloc[train_idx]
        y_train = y_cc.iloc[train_idx]
        X_test = X_cc.iloc[test_idx]
        model = make_logistic_model()
        model.fit(X_train, y_train)
        pred = pd.Series(model.predict_proba(X_test)[:, 1], index=X_test.index)
        pred_sum.loc[X_test.index] += pred
        pred_count.loc[X_test.index] += 1.0
    out = pred_sum / pred_count.replace(0, np.nan)
    out.name = "complete_case"
    return out, y_cc, X_cc


def bootstrap_auc_ci(y: np.ndarray, p: np.ndarray, n_boot: int = 2000, seed: int = RANDOM_STATE) -> tuple[float, float]:
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


def calibration_stats(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    p = np.clip(np.asarray(y_prob, dtype=float), 1e-8, 1 - 1e-8)
    y = np.asarray(y_true, dtype=int)
    logit_p = np.log(p / (1 - p))
    X_design = sm.add_constant(logit_p, has_constant="add")
    model = sm.Logit(y, X_design).fit(disp=False, maxiter=1000)
    vals = np.asarray(model.params, dtype=float)
    return float(vals[0]), float(vals[1])


def summarize_predictions(method: str, y: pd.Series, pred: pd.Series, seed_offset: int = 0) -> dict[str, float | int | str]:
    valid = y.notna() & pred.notna()
    yy = y.loc[valid].astype(int)
    pp = pred.loc[valid].astype(float)
    auc = roc_auc_score(yy, pp)
    lo, hi = bootstrap_auc_ci(yy.values, pp.values, seed=RANDOM_STATE + seed_offset)
    try:
        cal_int, cal_slope = calibration_stats(yy.values, pp.values)
    except Exception:
        cal_int, cal_slope = np.nan, np.nan
    return {
        "method": method,
        "n": int(len(yy)),
        "events": int(yy.sum()),
        "event_rate": float(yy.mean()),
        "auc": float(auc),
        "auc_ci_low": float(lo),
        "auc_ci_high": float(hi),
        "brier": float(brier_score_loss(yy, pp)),
        "calibration_intercept": float(cal_int),
        "calibration_slope": float(cal_slope),
    }


def fit_full_model_coefficients(X: pd.DataFrame, y: pd.Series, method: str, n_imputations: int) -> pd.DataFrame:
    rows = []
    if method == "complete_case":
        mask = X.notna().all(axis=1)
        X_fit = X.loc[mask]
        y_fit = y.loc[mask]
        model = make_logistic_model()
        model.fit(X_fit, y_fit)
        names = ["const"] + list(X_fit.columns)
        betas = [float(model.intercept_[0])] + [float(x) for x in model.coef_.ravel()]
        return pd.DataFrame({"method": method, "variable": names, "beta": betas})

    if method == "median_plus_missing_indicators":
        X_use = add_missing_indicators(X)
        imputer = SimpleImputer(strategy="median")
        X_imp = pd.DataFrame(imputer.fit_transform(X_use), columns=X_use.columns, index=X_use.index)
        model = make_logistic_model()
        model.fit(X_imp, y)
        names = ["const"] + list(X_imp.columns)
        betas = [float(model.intercept_[0])] + [float(x) for x in model.coef_.ravel()]
        return pd.DataFrame({"method": method, "variable": names, "beta": betas})

    factories: dict[str, Callable[[], object]] = {
        "median": lambda: SimpleImputer(strategy="median"),
        "mean": lambda: SimpleImputer(strategy="mean"),
        "most_frequent": lambda: SimpleImputer(strategy="most_frequent"),
        "knn": lambda: KNNImputer(n_neighbors=5, weights="uniform"),
        "iterative_single": lambda: make_iterative_imputer(RANDOM_STATE + 500, sample_posterior=False),
    }

    if method in factories:
        imputer = factories[method]()
        X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
        model = make_logistic_model()
        model.fit(X_imp, y)
        names = ["const"] + list(X.columns)
        betas = [float(model.intercept_[0])] + [float(x) for x in model.coef_.ravel()]
        return pd.DataFrame({"method": method, "variable": names, "beta": betas})

    if method == "multiple_imputation":
        for m in range(n_imputations):
            imp = make_iterative_imputer(RANDOM_STATE + 9000 + m, sample_posterior=True)
            X_imp = pd.DataFrame(imp.fit_transform(X), columns=X.columns, index=X.index)
            model = make_logistic_model()
            model.fit(X_imp, y)
            names = ["const"] + list(X.columns)
            betas = [float(model.intercept_[0])] + [float(x) for x in model.coef_.ravel()]
            for name, beta in zip(names, betas):
                rows.append({"method": method, "imputation": m + 1, "variable": name, "beta": beta})
        raw = pd.DataFrame(rows)
        return raw.groupby(["method", "variable"], as_index=False).agg(
            beta=("beta", "mean"),
            beta_sd_across_imputations=("beta", "std"),
        )

    raise ValueError(f"Unknown method for coefficients: {method}")


def missingness_table(X: pd.DataFrame) -> pd.DataFrame:
    n = len(X)
    return pd.DataFrame({
        "variable": X.columns,
        "available_n": [int(X[c].notna().sum()) for c in X.columns],
        "available_percent": [float(100 * X[c].notna().mean()) for c in X.columns],
        "missing_n": [int(X[c].isna().sum()) for c in X.columns],
        "missing_percent": [float(100 * X[c].isna().mean()) for c in X.columns],
    })


def roc_comparison_plot(predictions: dict[str, pd.Series], y: pd.Series, outpath: Path) -> None:
    plt.figure(figsize=(7.2, 6.2))
    for method, pred in predictions.items():
        valid = pred.notna() & y.notna()
        yy = y.loc[valid].astype(int)
        pp = pred.loc[valid].astype(float)
        fpr, tpr, _ = roc_curve(yy, pp)
        auc = roc_auc_score(yy, pp)
        plt.plot(fpr, tpr, linewidth=2.0, label=f"{method} AUC {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Imputation sensitivity analysis: ROC curves")
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()


def calibration_comparison_plot(predictions: dict[str, pd.Series], y: pd.Series, outpath: Path) -> None:
    n_methods = len(predictions)
    cols = 2
    rows = int(np.ceil(n_methods / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12.0, 5.0 * rows))
    axes = np.asarray(axes).ravel()
    for ax, (method, pred) in zip(axes, predictions.items()):
        valid = pred.notna() & y.notna()
        yy = y.loc[valid].astype(int).to_numpy()
        pp = np.clip(pred.loc[valid].astype(float).to_numpy(), 1e-6, 1 - 1e-6)
        frame = pd.DataFrame({"y": yy, "pred": pp}).sort_values("pred").reset_index(drop=True)
        x_max = min(0.60, float(np.quantile(frame["pred"], 0.98)))
        plot_data = frame[frame["pred"] <= x_max].copy()
        if len(plot_data) < 30:
            plot_data = frame.copy()
            x_max = min(0.60, float(frame["pred"].max()))
        curve = lowess(plot_data["y"], plot_data["pred"], frac=0.72, it=0, return_sorted=True)
        ax.plot([0, x_max], [0, x_max], linestyle="--", linewidth=1.1, label="Ideal")
        ax.plot(curve[:, 0], curve[:, 1], linewidth=2.1, label="LOWESS")
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, 0.70)
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed event rate")
        ax.set_title(method)
    for ax in axes[n_methods:]:
        ax.axis("off")
    plt.suptitle("Imputation sensitivity analysis: calibration", y=1.01, fontsize=16)
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()


def metric_bar_plot(summary: pd.DataFrame, metric: str, ylabel: str, outpath: Path) -> None:
    plot = summary.copy().sort_values("method")
    plt.figure(figsize=(9.5, 5.8))
    x = np.arange(len(plot))
    plt.bar(x, plot[metric].astype(float))
    plt.xticks(x, plot["method"], rotation=35, ha="right")
    plt.ylabel(ylabel)
    plt.title(f"Sensitivity analysis: {ylabel}")
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()


def net_benefit(y_true: np.ndarray, pred_prob: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(pred_prob).astype(float)
    n = len(y)
    out = np.full(len(thresholds), np.nan)
    for i, pt in enumerate(thresholds):
        positive = p >= pt
        tp = np.sum((positive == 1) & (y == 1))
        fp = np.sum((positive == 1) & (y == 0))
        out[i] = (tp / n) - (fp / n) * (pt / (1 - pt))
    return out


def treat_all_net_benefit(y_true: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    prevalence = np.mean(np.asarray(y_true).astype(int))
    return prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))


def dca_comparison_plot(predictions: dict[str, pd.Series], y: pd.Series, outpath: Path, table_path: Path) -> None:
    thresholds = np.arange(0.10, 0.50 + 1e-9, 0.01)
    rows = []
    plt.figure(figsize=(8.2, 6.7))
    for method, pred in predictions.items():
        valid = pred.notna() & y.notna()
        yy = y.loc[valid].astype(int).to_numpy()
        pp = pred.loc[valid].astype(float).to_numpy()
        nb = net_benefit(yy, pp, thresholds)
        plt.plot(thresholds, nb, linewidth=1.9, label=method)
        for threshold, value in zip(thresholds, nb):
            rows.append({"method": method, "threshold": float(threshold), "net_benefit": float(value)})
    base_valid = y.notna()
    yy_all = y.loc[base_valid].astype(int).to_numpy()
    plt.plot(thresholds, treat_all_net_benefit(yy_all, thresholds), linewidth=1.6, label="Treat all")
    plt.axhline(0, linestyle="--", linewidth=1.4, label="Treat none")
    plt.xlabel("Threshold probability")
    plt.ylabel("Net benefit")
    plt.title("Imputation sensitivity analysis: decision curves")
    plt.xlim(0.10, 0.50)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()
    pd.DataFrame(rows).to_csv(table_path, index=False, encoding="utf-8-sig")


def make_composite_figure(paths: list[Path], outpath: Path) -> None:
    imgs = [Image.open(p).convert("RGB") for p in paths if p.exists()]
    if len(imgs) < 4:
        return
    min_w = min(img.width for img in imgs[:4])
    resized = []
    for img in imgs[:4]:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="AECOPD-CV imputation sensitivity analysis")
    parser.add_argument("--input", default=r"C:\Users\User\Desktop\Papaioannou\Idea 1\Anonymized data.xlsx")
    parser.add_argument("--sheet", default=None)
    parser.add_argument("--output", default=r"C:\Users\User\Desktop\Papaioannou\Idea 1\aecopd_cv_imputation_sensitivity")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=10)
    parser.add_argument("--n-imputations", type=int, default=10)
    parser.add_argument(
        "--methods",
        default="median,complete_case,multiple_imputation,mean,most_frequent,knn,iterative_single,median_plus_missing_indicators",
    )
    args = parser.parse_args()

    set_plot_style()
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    raw = load_table(Path(args.input), args.sheet)
    analysis, predictor_mapping, event_mapping = build_analysis_dataset(raw)
    work = analysis[analysis[TARGET].isin([0, 1])].copy()
    y = work[TARGET].astype(int)
    X = work[RAW_FEATURES].copy()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    allowed = {
        "median",
        "complete_case",
        "multiple_imputation",
        "mean",
        "most_frequent",
        "knn",
        "iterative_single",
        "median_plus_missing_indicators",
    }
    unknown = [m for m in methods if m not in allowed]
    if unknown:
        raise ValueError(f"Unknown methods requested: {unknown}")

    analysis.to_csv(outdir / "analysis_dataset_imputation_sensitivity.csv", index=False, encoding="utf-8-sig")
    predictor_mapping.to_csv(outdir / "variable_mapping_imputation_sensitivity.csv", index=False, encoding="utf-8-sig")
    event_mapping.to_csv(outdir / "event_mapping_imputation_sensitivity.csv", index=False, encoding="utf-8-sig")
    missingness_table(X).to_csv(outdir / "table_predictor_missingness.csv", index=False, encoding="utf-8-sig")

    predictions: dict[str, pd.Series] = {}
    summaries = []
    coefficients = []

    for i, method in enumerate(methods, start=1):
        print(f"Running method {i}/{len(methods)}: {method}")
        if method == "complete_case":
            pred, y_method, X_method = complete_case_oof(X, y, args.n_splits, args.n_repeats)
            predictions[method] = pred
            summaries.append(summarize_predictions(method, y_method, pred, seed_offset=200 + i))
        else:
            pred = repeated_oof_predictions(
                X=X,
                y=y,
                method=method,
                n_splits=args.n_splits,
                n_repeats=args.n_repeats,
                n_imputations=args.n_imputations,
            )
            predictions[method] = pred
            summaries.append(summarize_predictions(method, y, pred, seed_offset=200 + i))

        try:
            coefficients.append(fit_full_model_coefficients(X, y, method, args.n_imputations))
        except Exception as exc:
            print(f"Could not save full-model coefficients for {method}: {exc}")

    summary = pd.DataFrame(summaries)
    summary.to_csv(outdir / "summary_imputation_sensitivity.csv", index=False, encoding="utf-8-sig")
    with open(outdir / "summary_imputation_sensitivity.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    pred_table = pd.DataFrame(index=work.index)
    pred_table[TARGET] = y
    for method, pred in predictions.items():
        pred_table[f"pred_{method}"] = pred
    pred_table.to_csv(outdir / "patient_level_predictions_by_imputation_method.csv", index=False, encoding="utf-8-sig")

    if coefficients:
        pd.concat(coefficients, ignore_index=True).to_csv(
            outdir / "full_model_coefficients_by_imputation_method.csv",
            index=False,
            encoding="utf-8-sig",
        )

    roc_path = outdir / "figure_roc_imputation_sensitivity.png"
    cal_path = outdir / "figure_calibration_imputation_sensitivity.png"
    auc_path = outdir / "figure_auc_by_imputation_method.png"
    brier_path = outdir / "figure_brier_by_imputation_method.png"
    dca_path = outdir / "figure_dca_imputation_sensitivity.png"

    roc_comparison_plot(predictions, y, roc_path)
    calibration_comparison_plot(predictions, y, cal_path)
    metric_bar_plot(summary, "auc", "AUC", auc_path)
    metric_bar_plot(summary, "brier", "Brier score", brier_path)
    dca_comparison_plot(predictions, y, dca_path, outdir / "table_decision_curve_imputation_sensitivity.csv")
    make_composite_figure(
        [roc_path, dca_path, cal_path, auc_path],
        outdir / "Imputation_Sensitivity_Composite.png",
    )

    print("\nDone.")
    print(f"Outputs written to: {outdir}")
    print("Key files:")
    print("- summary_imputation_sensitivity.csv")
    print("- table_predictor_missingness.csv")
    print("- patient_level_predictions_by_imputation_method.csv")
    print("- full_model_coefficients_by_imputation_method.csv")
    print("- figure_roc_imputation_sensitivity.png")
    print("- figure_dca_imputation_sensitivity.png")
    print("- figure_calibration_imputation_sensitivity.png")
    print("- Imputation_Sensitivity_Composite.png")


if __name__ == "__main__":
    main()
