#!/usr/bin/env python3
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
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from statsmodels.nonparametric.smoothers_lowess import lowess

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_STATE = 42
CHUNK_SIZE = 1_000_000
MODEL_FEATURES = ["age", "history_hf", "history_af", "ph", "urea", "lactate"]
PH_MIN, PH_MAX = 6.8, 7.8
BUN_MIN, BUN_MAX = 1.0, 300.0
LACTATE_MIN, LACTATE_MAX = 0.1, 30.0
BUN_TO_UREA_FACTOR = 2.14


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = {str(c).strip(): c for c in df.columns}
    for c in candidates:
        if c in cols:
            return cols[c]
    return None


def pick_file(folder: Path, stem: str) -> Path:
    for name in [f"{stem}.csv", f"{stem}.csv.gz"]:
        path = folder / name
        if path.exists():
            return path
    raise FileNotFoundError(f"{stem}.csv or {stem}.csv.gz was not found in {folder}")


def read_csv_auto(path: Path, **kwargs) -> pd.DataFrame:
    if path.suffix == ".gz":
        return pd.read_csv(path, compression="gzip", **kwargs)
    return pd.read_csv(path, **kwargs)


def read_table(path: Path, sheet: str | None = None) -> pd.DataFrame:
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return normalize_columns(pd.read_excel(path, sheet_name=sheet or 0))
    return normalize_columns(pd.read_csv(path))


def find_file(folder: Path, filename: str) -> Path:
    path = folder / filename
    if path.exists():
        return path
    matches = list(folder.rglob(filename))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"{filename} was not found in {folder}")


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


def load_validation_cohort(path: Path, sheet: str | None = None) -> pd.DataFrame:
    df = read_table(path, sheet)
    if "subject_id" not in df.columns or "hadm_id" not in df.columns:
        raise ValueError("The validation cohort must contain subject_id and hadm_id.")
    if "cv_event" not in df.columns:
        raise ValueError("The validation cohort must contain cv_event.")
    for col in ["subject_id", "hadm_id"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df.dropna(subset=["subject_id", "hadm_id"]).copy()


def load_admission_times(hosp_dir: Path, cohort: pd.DataFrame) -> pd.DataFrame:
    admissions = read_csv_auto(
        pick_file(hosp_dir, "admissions"),
        usecols=["subject_id", "hadm_id", "admittime", "dischtime"],
        low_memory=False,
    )
    admissions["subject_id"] = pd.to_numeric(admissions["subject_id"], errors="coerce").astype("Int64")
    admissions["hadm_id"] = pd.to_numeric(admissions["hadm_id"], errors="coerce").astype("Int64")
    admissions["admittime"] = pd.to_datetime(admissions["admittime"], errors="coerce")
    admissions["dischtime"] = pd.to_datetime(admissions["dischtime"], errors="coerce")
    keys = cohort[["subject_id", "hadm_id"]].drop_duplicates()
    out = keys.merge(admissions, on=["subject_id", "hadm_id"], how="left")
    missing = int(out["admittime"].isna().sum())
    if missing:
        raise ValueError(f"{missing} admissions could not be linked to admittime.")
    return out


def normalize_label(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()


def get_lab_itemids(hosp_dir: Path) -> tuple[dict[str, set[int]], pd.DataFrame]:
    dlab = normalize_columns(read_csv_auto(pick_file(hosp_dir, "d_labitems"), low_memory=False))
    if "itemid" not in dlab.columns or "label" not in dlab.columns:
        raise ValueError("d_labitems must contain itemid and label.")
    dlab["label_norm"] = normalize_label(dlab["label"])
    dlab["fluid_norm"] = normalize_label(dlab["fluid"]) if "fluid" in dlab.columns else ""
    dlab["category_norm"] = normalize_label(dlab["category"]) if "category" in dlab.columns else ""
    blood_mask = dlab["fluid_norm"].str.contains("BLOOD", na=False) | dlab["category_norm"].str.contains("BLOOD GAS|CHEM", na=False)
    masks = {
        "ph": dlab["label_norm"].eq("PH") & blood_mask,
        "bun": dlab["label_norm"].str.contains("UREA NITROGEN|BUN", na=False) & blood_mask,
        "lactate": dlab["label_norm"].str.contains("LACTATE", na=False) & blood_mask,
    }
    itemids = {}
    rows = []
    for variable, mask in masks.items():
        ids = set(dlab.loc[mask, "itemid"].dropna().astype(int).tolist())
        itemids[variable] = ids
        for itemid in sorted(ids):
            row = dlab.loc[dlab["itemid"] == itemid].head(1)
            rows.append({
                "source": "labevents",
                "variable": variable,
                "itemid": itemid,
                "label": str(row["label"].iloc[0]) if not row.empty else "",
                "fluid": str(row["fluid"].iloc[0]) if "fluid" in dlab.columns and not row.empty else "",
                "category": str(row["category"].iloc[0]) if "category" in dlab.columns and not row.empty else "",
            })
    return itemids, pd.DataFrame(rows)


def plausible_value(variable: str, value: float) -> bool:
    if variable == "ph":
        return PH_MIN <= value <= PH_MAX
    if variable == "bun":
        return BUN_MIN <= value <= BUN_MAX
    if variable == "lactate":
        return LACTATE_MIN <= value <= LACTATE_MAX
    return False


def extract_first_labs(cohort: pd.DataFrame, mimic_root: Path, hours: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    hosp_dir = mimic_root / "hosp"
    if not hosp_dir.exists():
        raise FileNotFoundError(f"The hosp directory was not found: {hosp_dir}")
    times = load_admission_times(hosp_dir, cohort)
    times["window_end"] = times["admittime"] + pd.to_timedelta(hours, unit="h")
    hadm_set = set(times["hadm_id"].dropna().astype(int).tolist())
    time_map = times.set_index("hadm_id")[["admittime", "window_end", "dischtime"]].to_dict("index")
    itemids, item_table = get_lab_itemids(hosp_dir)
    reverse = {int(itemid): variable for variable, ids in itemids.items() for itemid in ids}
    best: dict[tuple[int, str], dict] = {}
    labevents = pick_file(hosp_dir, "labevents")
    usecols = ["hadm_id", "itemid", "charttime", "valuenum"]
    for chunk in read_csv_auto(labevents, usecols=usecols, chunksize=CHUNK_SIZE, low_memory=False):
        chunk["hadm_id"] = pd.to_numeric(chunk["hadm_id"], errors="coerce").astype("Int64")
        chunk["itemid"] = pd.to_numeric(chunk["itemid"], errors="coerce")
        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        chunk["valuenum"] = pd.to_numeric(chunk["valuenum"], errors="coerce")
        chunk = chunk[
            chunk["hadm_id"].isin(hadm_set)
            & chunk["itemid"].isin(reverse.keys())
            & chunk["charttime"].notna()
            & chunk["valuenum"].notna()
        ].copy()
        if chunk.empty:
            continue
        for row in chunk.itertuples(index=False):
            hadm_id = int(row.hadm_id)
            itemid = int(row.itemid)
            charttime = row.charttime
            value = float(row.valuenum)
            variable = reverse[itemid]
            entry = time_map.get(hadm_id)
            if entry is None:
                continue
            if charttime < entry["admittime"]:
                continue
            if charttime > entry["window_end"]:
                continue
            if pd.notna(entry["dischtime"]) and charttime > entry["dischtime"]:
                continue
            if not plausible_value(variable, value):
                continue
            key = (hadm_id, variable)
            if key not in best or charttime < best[key]["charttime"]:
                best[key] = {"value": value, "charttime": charttime, "itemid": itemid}
    rows = []
    for row in times.itertuples(index=False):
        hadm_id = int(row.hadm_id)
        ph = best.get((hadm_id, "ph"), {})
        bun = best.get((hadm_id, "bun"), {})
        lactate = best.get((hadm_id, "lactate"), {})
        bun_value = bun.get("value", np.nan)
        rows.append({
            "subject_id": int(row.subject_id),
            "hadm_id": hadm_id,
            "ph_12h": ph.get("value", np.nan),
            "ph_12h_charttime": ph.get("charttime", pd.NaT),
            "ph_12h_itemid": ph.get("itemid", np.nan),
            "bun_12h": bun_value,
            "bun_12h_charttime": bun.get("charttime", pd.NaT),
            "bun_12h_itemid": bun.get("itemid", np.nan),
            "urea_12h": bun_value * BUN_TO_UREA_FACTOR if pd.notna(bun_value) else np.nan,
            "lactate_12h": lactate.get("value", np.nan),
            "lactate_12h_charttime": lactate.get("charttime", pd.NaT),
            "lactate_12h_itemid": lactate.get("itemid", np.nan),
        })
    return pd.DataFrame(rows), item_table


def load_coefficients(model_dir: Path) -> tuple[float, dict[str, float], str]:
    path = find_file(model_dir, "model_coefficients_for_figures.csv")
    coef = normalize_columns(pd.read_csv(path))
    var_col = first_existing(coef, ["variable", "term", "feature", "predictor"])
    beta_col = first_existing(coef, ["beta", "coef", "coefficient", "estimate"])
    if var_col is None or beta_col is None:
        raise ValueError("The coefficient file must contain variable and beta columns.")
    intercept = None
    betas = {}
    for _, row in coef.iterrows():
        variable = str(row[var_col]).strip()
        beta = float(row[beta_col])
        if variable.lower() in ["const", "intercept", "(intercept)"]:
            intercept = beta
        elif variable in MODEL_FEATURES:
            betas[variable] = beta
    if intercept is None:
        raise ValueError("No intercept row was found in the coefficient file.")
    missing = [feature for feature in MODEL_FEATURES if feature not in betas]
    if missing:
        raise ValueError(f"The coefficient file is missing: {missing}")
    return float(intercept), {k: float(v) for k, v in betas.items()}, str(path)


def load_model_update(model_dir: Path) -> tuple[float, float, str]:
    json_path = model_dir / "recalibration_parameters.json"
    if json_path.exists():
        params = json.loads(json_path.read_text(encoding="utf-8"))
        return float(params["recalibration_intercept"]), float(params["recalibration_slope"]), str(json_path)
    path = find_file(model_dir, "summary_internal_validation.csv")
    summary = normalize_columns(pd.read_csv(path))
    return float(summary.loc[0, "recalibration_intercept"]), float(summary.loc[0, "recalibration_slope"]), str(path)


def load_medians(model_dir: Path, medians_csv: str | None = None) -> tuple[dict[str, float], str]:
    if medians_csv:
        path = Path(medians_csv)
        med = normalize_columns(pd.read_csv(path))
        var_col = first_existing(med, ["variable", "term", "feature", "predictor"])
        val_col = first_existing(med, ["median", "value", "imputation_median"])
        if var_col is None or val_col is None:
            raise ValueError("The medians file must contain variable and median columns.")
        values = {str(row[var_col]).strip(): float(row[val_col]) for _, row in med.iterrows() if str(row[var_col]).strip() in MODEL_FEATURES}
        missing = [feature for feature in MODEL_FEATURES if feature not in values]
        if missing:
            raise ValueError(f"The medians file is missing: {missing}")
        return values, str(path)
    path = find_file(model_dir, "analysis_dataset_internal_validation.csv")
    data = normalize_columns(pd.read_csv(path))
    values = {feature: float(pd.to_numeric(data[feature], errors="coerce").median()) for feature in MODEL_FEATURES}
    return values, str(path)


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-8, 1 - 1e-8)
    return np.log(p / (1 - p))


def inverse_logit(lp: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(lp, dtype=float)))


def apply_model(df: pd.DataFrame, model_dir: Path, medians_csv: str | None = None) -> tuple[pd.DataFrame, dict]:
    intercept, betas, coef_source = load_coefficients(model_dir)
    update_intercept, update_slope, update_source = load_model_update(model_dir)
    medians, medians_source = load_medians(model_dir, medians_csv)
    required = ["age", "history_hf", "history_af"]
    missing_base = [c for c in required if c not in df.columns]
    if missing_base:
        raise ValueError(f"The validation cohort is missing: {missing_base}")
    X = pd.DataFrame(index=df.index)
    X["age"] = pd.to_numeric(df["age"], errors="coerce")
    X["history_hf"] = pd.to_numeric(df["history_hf"], errors="coerce")
    X["history_af"] = pd.to_numeric(df["history_af"], errors="coerce")
    X["ph"] = pd.to_numeric(df["ph_12h"], errors="coerce")
    X["urea"] = pd.to_numeric(df["urea_12h"], errors="coerce")
    X["lactate"] = pd.to_numeric(df["lactate_12h"], errors="coerce")
    pred = pd.DataFrame(index=df.index)
    for feature in MODEL_FEATURES:
        pred[f"{feature}_12h_missing_before_imputation"] = X[feature].isna().astype(int)
        X[feature] = X[feature].fillna(medians[feature])
        pred[f"{feature}_model_input_12h"] = X[feature].astype(float)
    lp = np.full(len(X), intercept, dtype=float)
    for feature in MODEL_FEATURES:
        lp += betas[feature] * X[feature].astype(float).to_numpy()
    probability = inverse_logit(lp)
    lp_final = update_intercept + update_slope * logit(probability)
    probability_final = inverse_logit(lp_final)
    pred["linear_predictor_original_12h"] = lp
    pred["predicted_probability_original_12h"] = probability
    pred["linear_predictor_frozen_12h"] = lp_final
    pred["predicted_probability_frozen_12h"] = probability_final
    pred["predicted_probability_frozen"] = probability_final
    metadata = {
        "coefficient_source": coef_source,
        "model_update_source": update_source,
        "medians_source": medians_source,
        "intercept": intercept,
        "betas": betas,
        "model_update_intercept": update_intercept,
        "model_update_slope": update_slope,
        "medians": medians,
        "laboratory_window_hours": 12,
    }
    return pred, metadata


def bootstrap_auc_ci(y: np.ndarray, p: np.ndarray, n_boot: int = 2000, seed: int = RANDOM_STATE) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
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
    X = sm.add_constant(np.log(p / (1 - p)), has_constant="add")
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


def smooth_ma(y: np.ndarray, window: int = 5) -> np.ndarray:
    arr = np.asarray(y, dtype=float)
    if window <= 1 or len(arr) < window:
        return arr.copy()
    kernel = np.ones(window) / window
    pad = window // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def bootstrap_net_benefit_ci(y: np.ndarray, p: np.ndarray, thresholds: np.ndarray, n_boot: int = 300, seed: int = RANDOM_STATE) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(y)
    boots = np.zeros((n_boot, len(thresholds)), dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b, :] = net_benefit(y[idx], p[idx], thresholds)
    return np.mean(boots, axis=0), np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)


def make_roc_plot(y: np.ndarray, p: np.ndarray, outpath: Path, table_out: Path) -> dict:
    auc = roc_auc_score(y, p)
    lo, hi = bootstrap_auc_ci(y, p, seed=101)
    fpr, tpr, _ = roc_curve(y, p)
    plt.figure(figsize=(6.8, 6.0))
    plt.plot(fpr, tpr, linewidth=2.4, label=f"AECOPD-CV model (AUC {auc:.3f}, 95% CI {lo:.3f}–{hi:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("External validation ROC curve")
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()
    result = {"model": "AECOPD-CV model", "auc": auc, "auc_ci_low": lo, "auc_ci_high": hi}
    pd.DataFrame([result]).to_csv(table_out, index=False)
    return result


def make_calibration_plot(y: np.ndarray, p: np.ndarray, outpath: Path, table_out: Path) -> dict:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=int)
    df = pd.DataFrame({"y": y, "pred": p}).sort_values("pred").reset_index(drop=True)
    x_max = min(0.90, float(np.quantile(df["pred"], 0.98)))
    plot_df = df[df["pred"] <= x_max].copy()
    if len(plot_df) < 30:
        plot_df = df.copy()
        x_max = min(0.90, float(df["pred"].max()))
    curve = lowess(plot_df["y"], plot_df["pred"], frac=0.72, it=0, return_sorted=True)
    grid = np.linspace(float(plot_df["pred"].min()), float(plot_df["pred"].max()), 160)
    rng = np.random.default_rng(1702)
    curves = []
    for _ in range(180):
        idx = rng.integers(0, len(plot_df), size=len(plot_df))
        boot = plot_df.iloc[idx].sort_values("pred")
        lo_b = lowess(boot["y"], boot["pred"], frac=0.72, it=0, return_sorted=True)
        xb = np.asarray(lo_b[:, 0], dtype=float)
        yb = np.asarray(lo_b[:, 1], dtype=float)
        uniq = np.unique(xb, return_index=True)[1]
        xu = xb[np.sort(uniq)]
        yu = yb[np.sort(uniq)]
        if len(xu) >= 2:
            curves.append(np.interp(grid, xu, yu, left=yu[0], right=yu[-1]))
    intercept, slope = calibration_stats(y, p)
    plt.figure(figsize=(6.2, 5.6))
    plt.plot([0, x_max], [0, x_max], linestyle="--", linewidth=1.2, label="Ideal calibration")
    if len(curves) > 20:
        curves = np.asarray(curves)
        plt.fill_between(grid, np.clip(np.percentile(curves, 2.5, axis=0), 0, 0.95), np.clip(np.percentile(curves, 97.5, axis=0), 0, 0.95), alpha=0.12, label="LOWESS 95% CI")
    plt.plot(curve[:, 0], curve[:, 1], linewidth=2.4, label="LOWESS-smoothed calibration")
    plt.xlim(0, x_max)
    plt.ylim(0, max(0.75, min(0.95, float(np.nanmax(curve[:, 1])) + 0.08)))
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed event rate")
    plt.title("External validation calibration plot")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()
    result = {"model": "AECOPD-CV model", "calibration_intercept": intercept, "calibration_slope": slope, "x_max_plot": x_max}
    pd.DataFrame([result]).to_csv(table_out, index=False)
    return result


def make_dca_plot(y: np.ndarray, p: np.ndarray, outpath: Path, table_out: Path) -> None:
    thresholds = np.arange(0.10, 0.50 + 1e-9, 0.01)
    mid, lo, hi = bootstrap_net_benefit_ci(y, p, thresholds, n_boot=300, seed=84)
    all_nb = treat_all_net_benefit(y, thresholds)
    none_nb = np.zeros_like(thresholds)
    plt.figure(figsize=(8, 6.8))
    plt.fill_between(thresholds, smooth_ma(lo, 5), smooth_ma(hi, 5), alpha=0.12, label="AECOPD-CV model 95% CI")
    plt.plot(thresholds, smooth_ma(mid, 5), linewidth=2.4, label="AECOPD-CV model")
    plt.plot(thresholds, smooth_ma(all_nb, 5), linewidth=2.0, label="Treat all")
    plt.axhline(0, linestyle="--", linewidth=1.5, label="Treat none")
    plt.xlim(0.10, 0.50)
    ymin = min(np.nanmin(smooth_ma(lo, 5)), np.nanmin(smooth_ma(all_nb, 5)), -0.02)
    ymax = max(np.nanmax(smooth_ma(hi, 5)), np.nanmax(smooth_ma(all_nb, 5)), 0.02)
    plt.ylim(ymin - 0.03, ymax + 0.03)
    plt.xlabel("Threshold probability")
    plt.ylabel("Net benefit")
    plt.title("External validation decision curve analysis")
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()
    pd.DataFrame({"threshold": thresholds, "aecopd_cv_model_nb": mid, "aecopd_cv_model_nb_low": lo, "aecopd_cv_model_nb_high": hi, "treat_all_nb": all_nb, "treat_none_nb": none_nb}).to_csv(table_out, index=False)


def make_quartile_plot(y: np.ndarray, p: np.ndarray, outpath: Path, table_out: Path) -> None:
    df = pd.DataFrame({"y": y, "pred": p}).dropna()
    try:
        df["quartile"] = pd.qcut(df["pred"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
    except ValueError:
        df["quartile"] = pd.cut(df["pred"], bins=4, labels=["Q1", "Q2", "Q3", "Q4"], include_lowest=True)
    g = df.groupby("quartile", observed=False).agg(n=("y", "size"), events=("y", "sum"), event_rate=("y", "mean"), mean_pred=("pred", "mean")).reset_index()
    g["ci_low"], g["ci_high"] = wilson_ci(g["events"].values, g["n"].values)
    x = np.arange(len(g))
    plt.figure(figsize=(6.4, 5.2))
    plt.errorbar(x, g["event_rate"], yerr=[g["event_rate"] - g["ci_low"], g["ci_high"] - g["event_rate"]], fmt="o", capsize=4, linewidth=1.5, markersize=8)
    plt.plot(x, g["event_rate"], linewidth=1.7, alpha=0.9)
    plt.xticks(x, g["quartile"])
    plt.ylabel("Observed event rate")
    plt.xlabel("Predicted-risk quartile")
    plt.title("Observed event rate across predicted-risk quartiles")
    plt.ylim(0, max(0.40, float(g["ci_high"].max()) + 0.05))
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()
    g["model"] = "AECOPD-CV model"
    g.to_csv(table_out, index=False)


def make_composite_figure(roc_path: Path, dca_path: Path, cal_path: Path, quart_path: Path, outpath: Path) -> None:
    imgs = [Image.open(path).convert("RGB") for path in [roc_path, dca_path, cal_path, quart_path]]
    w = max(img.width for img in imgs)
    h = max(img.height for img in imgs)
    canvas = Image.new("RGB", (2 * w, 2 * h), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, img in enumerate(imgs):
        x = (idx % 2) * w
        y = (idx // 2) * h
        canvas.paste(img.resize((w, h)), (x, y))
        draw.text((x + 10, y + 10), ["(A)", "(B)", "(C)", "(D)"][idx], fill="black")
    canvas.save(outpath)


def availability_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["ph_12h", "urea_12h", "lactate_12h"]
    rows = []
    for col in cols:
        values = pd.to_numeric(df[col], errors="coerce")
        rows.append({"variable": col, "available_n": int(values.notna().sum()), "available_percent": round(float(values.notna().mean() * 100), 1), "missing_n": int(values.isna().sum()), "missing_percent": round(float(values.isna().mean() * 100), 1)})
    complete = df[cols].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
    rows.append({"variable": "all_three_12h_labs_complete", "available_n": int(complete.sum()), "available_percent": round(float(complete.mean() * 100), 1), "missing_n": int((~complete).sum()), "missing_percent": round(float((~complete).mean() * 100), 1)})
    return pd.DataFrame(rows)


def validate(df: pd.DataFrame, outdir: Path) -> dict:
    y = pd.to_numeric(df["cv_event"], errors="coerce").astype(int).to_numpy()
    p = pd.to_numeric(df["predicted_probability_frozen"], errors="coerce").to_numpy(dtype=float)
    roc_path = outdir / "figure_roc_external_validation_12h.png"
    dca_path = outdir / "figure_decision_curve_external_validation_12h.png"
    cal_path = outdir / "figure_calibration_external_validation_12h.png"
    quart_path = outdir / "figure_quartiles_external_validation_12h.png"
    roc = make_roc_plot(y, p, roc_path, outdir / "table_roc_external_validation_12h.csv")
    cal = make_calibration_plot(y, p, cal_path, outdir / "table_calibration_external_validation_12h.csv")
    make_dca_plot(y, p, dca_path, outdir / "table_decision_curve_external_validation_12h.csv")
    make_quartile_plot(y, p, quart_path, outdir / "table_quartiles_external_validation_12h.csv")
    try:
        make_composite_figure(roc_path, dca_path, cal_path, quart_path, outdir / "External_Validation_Composite_12h.png")
    except Exception:
        pass
    return {"n": int(len(df)), "events": int(y.sum()), "event_rate": float(np.mean(y)), "auc": float(roc["auc"]), "auc_ci_low": float(roc["auc_ci_low"]), "auc_ci_high": float(roc["auc_ci_high"]), "brier_score": float(brier_score_loss(y, p)), "calibration_intercept": float(cal["calibration_intercept"]), "calibration_slope": float(cal["calibration_slope"])}


def main() -> None:
    parser = argparse.ArgumentParser(description="MIMIC-IV external validation with 12-hour laboratory extraction.")
    parser.add_argument("--validation-cohort", required=True)
    parser.add_argument("--validation-sheet", default=None)
    parser.add_argument("--mimic-root", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hours", type=int, default=12)
    parser.add_argument("--medians-csv", default=None)
    args = parser.parse_args()
    set_plot_style()
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    cohort = load_validation_cohort(Path(args.validation_cohort), args.validation_sheet)
    labs, item_table = extract_first_labs(cohort, Path(args.mimic_root), hours=args.hours)
    merged = cohort.merge(labs, on=["subject_id", "hadm_id"], how="left", validate="one_to_one")
    pred, metadata = apply_model(merged, Path(args.model_dir), args.medians_csv)
    old_prediction_cols = ["linear_predictor_original", "predicted_probability_original", "linear_predictor_frozen", "predicted_probability_frozen", "predicted_risk_original"]
    merged = merged.drop(columns=[c for c in old_prediction_cols if c in merged.columns], errors="ignore")
    final_df = pd.concat([merged.reset_index(drop=True), pred.reset_index(drop=True)], axis=1)
    final_df["predicted_risk_original"] = pd.to_numeric(final_df["predicted_probability_frozen"], errors="coerce")
    final_df.to_csv(outdir / "patient_level_external_validation_predictions_MIMIC-IV_12h_labs.csv", index=False)
    with pd.ExcelWriter(outdir / "Database_MIMIC-IV_external_validation_12h_labs_predictions.xlsx", engine="openpyxl", mode="w") as writer:
        final_df.to_excel(writer, sheet_name="AECOPD_validation_dataset_12h", index=False)
        availability_table(final_df).to_excel(writer, sheet_name="lab_availability_12h", index=False)
        item_table.to_excel(writer, sheet_name="lab_itemids_used", index=False)
    availability = availability_table(final_df)
    availability.to_csv(outdir / "lab_availability_12h.csv", index=False)
    item_table.to_csv(outdir / "lab_itemids_used_12h.csv", index=False)
    summary = validate(final_df, outdir)
    summary["laboratory_window_hours"] = args.hours
    summary["metadata"] = metadata
    pd.DataFrame([summary]).to_csv(outdir / "summary_external_validation_12h.csv", index=False)
    (outdir / "summary_external_validation_12h.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (outdir / "model_application_metadata_12h.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("External validation complete.")
    print(f"Rows analysed: {summary['n']}")
    print(f"Events: {summary['events']}")
    print(f"AUC: {summary['auc']:.3f} ({summary['auc_ci_low']:.3f}-{summary['auc_ci_high']:.3f})")
    print(f"Output: {outdir}")


if __name__ == "__main__":
    main()
