#!/usr/bin/env python3
from __future__ import annotations

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

PROJECT_ROOT = Path.cwd()

ORIGINAL_VALIDATION = PROJECT_ROOT / "Database.xlsx"
ORIGINAL_SHEET = "AECOPD_validation_dataset_v3"
MIMIC_ROOT = PROJECT_ROOT / "mimic-iv-3.1"
INTERNAL_MODEL_DIR = PROJECT_ROOT / "AECOPD_CV_FINAL_MODEL_Internal_Validation"
OUTPUT_DIR = PROJECT_ROOT / "external_validation_anytime"
LAB_WINDOW_HOURS = None
MEDIANS_CSV = None


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = {str(c).strip(): c for c in df.columns}
    for c in candidates:
        if c in cols:
            return cols[c]
    return None


def pick_file(folder: Path, stem: str) -> Path:
    for name in [f"{stem}.csv", f"{stem}.csv.gz"]:
        p = folder / name
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find {stem}.csv or {stem}.csv.gz in {folder}")


def read_csv_auto(path: Path, **kwargs) -> pd.DataFrame:
    if path.suffix == ".gz":
        return pd.read_csv(path, compression="gzip", **kwargs)
    return pd.read_csv(path, **kwargs)


def read_table(path: Path, sheet: str | None = None) -> pd.DataFrame:
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return normalize_columns(pd.read_excel(path, sheet_name=sheet or 0))
    return normalize_columns(pd.read_csv(path))


def find_file(folder: Path, filename: str) -> Path:
    direct = folder / filename
    if direct.exists():
        return direct
    matches = list(folder.rglob(filename))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find {filename} inside {folder}")


def style() -> None:
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


def as_event_flag(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int).clip(0, 1)


def apply_final_cv_outcome_definition(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    required = [
        "mi_inclusive_event",
        "arrhythmia_inclusive_event",
        "hf_decompensation_event",
        "pe_inclusive_event",
    ]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError("Missing columns required for final cardiovascular outcome definition: " + ", ".join(missing))

    out["cv_event_database_original"] = as_event_flag(out["cv_event"]) if "cv_event" in out.columns else 0
    out["mi_inclusive_event"] = as_event_flag(out["mi_inclusive_event"])
    out["arrhythmia_inclusive_event"] = as_event_flag(out["arrhythmia_inclusive_event"])
    out["hf_decompensation_event"] = as_event_flag(out["hf_decompensation_event"])
    out["pe_inclusive_event"] = as_event_flag(out["pe_inclusive_event"])

    out["cv_event"] = (
        (out["mi_inclusive_event"] == 1)
        | (out["arrhythmia_inclusive_event"] == 1)
        | (out["hf_decompensation_event"] == 1)
        | (out["pe_inclusive_event"] == 1)
    ).astype(int)

    return out


def get_id_cols(df: pd.DataFrame) -> list[str]:
    if "subject_id" in df.columns and "hadm_id" in df.columns:
        return ["subject_id", "hadm_id"]
    raise ValueError("The original MIMIC-IV validation file must contain subject_id and hadm_id.")


def load_original_validation(path: Path, sheet: str | None = None) -> pd.DataFrame:
    df = read_table(path, sheet)
    id_cols = get_id_cols(df)

    for col in id_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    df = df.dropna(subset=id_cols).copy()
    df = apply_final_cv_outcome_definition(df)

    return df


def load_admissions_for_original(hosp_dir: Path, original_df: pd.DataFrame) -> pd.DataFrame:
    admissions = read_csv_auto(
        pick_file(hosp_dir, "admissions"),
        usecols=["subject_id", "hadm_id", "admittime", "dischtime"],
        low_memory=False,
    )
    admissions["subject_id"] = pd.to_numeric(admissions["subject_id"], errors="coerce").astype("Int64")
    admissions["hadm_id"] = pd.to_numeric(admissions["hadm_id"], errors="coerce").astype("Int64")
    admissions["admittime"] = pd.to_datetime(admissions["admittime"], errors="coerce")
    admissions["dischtime"] = pd.to_datetime(admissions["dischtime"], errors="coerce")

    keys = original_df[["subject_id", "hadm_id"]].drop_duplicates()
    out = keys.merge(admissions, on=["subject_id", "hadm_id"], how="left")

    missing = out["admittime"].isna().sum()
    if missing:
        raise ValueError(f"{missing} original validation admissions could not be linked to MIMIC-IV admissions/admittime.")

    return out


def normalize_label(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()


def get_labevents_itemids(hosp_dir: Path) -> tuple[dict[str, set[int]], pd.DataFrame]:
    dlab = read_csv_auto(pick_file(hosp_dir, "d_labitems"), low_memory=False)
    dlab = normalize_columns(dlab)

    if "itemid" not in dlab.columns or "label" not in dlab.columns:
        raise ValueError("d_labitems must contain itemid and label.")

    dlab["label_norm"] = normalize_label(dlab["label"])
    dlab["fluid_norm"] = normalize_label(dlab["fluid"]) if "fluid" in dlab.columns else ""
    dlab["category_norm"] = normalize_label(dlab["category"]) if "category" in dlab.columns else ""

    blood_mask = (
        dlab["fluid_norm"].str.contains("BLOOD", na=False) |
        dlab["category_norm"].str.contains("BLOOD GAS|CHEM", na=False)
    )

    masks = {
        "ph": dlab["label_norm"].eq("PH") & blood_mask,
        "bun": dlab["label_norm"].str.contains("UREA NITROGEN|BUN", na=False) & blood_mask,
        "lactate": dlab["label_norm"].str.contains("LACTATE", na=False) & blood_mask,
    }

    itemids = {}
    rows = []
    for var, mask in masks.items():
        ids = set(dlab.loc[mask, "itemid"].dropna().astype(int).tolist())
        itemids[var] = ids
        for iid in sorted(ids):
            sub = dlab.loc[dlab["itemid"] == iid].head(1)
            rows.append({
                "source": "labevents",
                "variable": var,
                "itemid": iid,
                "label": str(sub["label"].iloc[0]) if not sub.empty else "",
                "fluid": str(sub["fluid"].iloc[0]) if "fluid" in dlab.columns and not sub.empty else "",
                "category": str(sub["category"].iloc[0]) if "category" in dlab.columns and not sub.empty else "",
            })

    return itemids, pd.DataFrame(rows)


def plausible(var: str, value: float) -> bool:
    if var == "ph":
        return PH_MIN <= value <= PH_MAX
    if var == "bun":
        return BUN_MIN <= value <= BUN_MAX
    if var == "lactate":
        return LACTATE_MIN <= value <= LACTATE_MAX
    return False


def extract_first_labs_anytime(original_df: pd.DataFrame, mimic_root: Path, hours: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    hosp_dir = mimic_root / "hosp"
    if not hosp_dir.exists():
        raise FileNotFoundError(f"Could not find hosp folder: {hosp_dir}")

    cohort_times = load_admissions_for_original(hosp_dir, original_df)
    cohort_times["window_end"] = pd.NaT

    hadm_set = set(cohort_times["hadm_id"].dropna().astype(int).tolist())
    time_map = cohort_times.set_index("hadm_id")[["admittime", "window_end", "dischtime"]].to_dict("index")

    itemids, item_table = get_labevents_itemids(hosp_dir)
    reverse = {}
    for var, ids in itemids.items():
        for iid in ids:
            reverse[int(iid)] = var

    best: dict[tuple[int, str], dict] = {}

    labevents = pick_file(hosp_dir, "labevents")
    usecols = ["hadm_id", "itemid", "charttime", "valuenum"]

    for chunk in read_csv_auto(labevents, usecols=usecols, chunksize=CHUNK_SIZE, low_memory=False):
        chunk["hadm_id"] = pd.to_numeric(chunk["hadm_id"], errors="coerce").astype("Int64")
        chunk["itemid"] = pd.to_numeric(chunk["itemid"], errors="coerce")
        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        chunk["valuenum"] = pd.to_numeric(chunk["valuenum"], errors="coerce")

        chunk = chunk[
            chunk["hadm_id"].isin(hadm_set) &
            chunk["itemid"].isin(reverse.keys()) &
            chunk["charttime"].notna() &
            chunk["valuenum"].notna()
        ].copy()

        if chunk.empty:
            continue

        for row in chunk.itertuples(index=False):
            hadm = int(row.hadm_id)
            itemid = int(row.itemid)
            charttime = row.charttime
            value = float(row.valuenum)
            var = reverse[itemid]

            t = time_map.get(hadm)
            if t is None:
                continue
            if charttime < t["admittime"]:
                continue
            if pd.notna(t["dischtime"]) and charttime > t["dischtime"]:
                continue
            if not plausible(var, value):
                continue

            key = (hadm, var)
            if key not in best or charttime < best[key]["charttime"]:
                best[key] = {
                    "hadm_id": hadm,
                    "variable": var,
                    "value": value,
                    "charttime": charttime,
                    "itemid": itemid,
                }

    rows = []
    for row in cohort_times.itertuples(index=False):
        hadm = int(row.hadm_id)

        ph = best.get((hadm, "ph"), {})
        bun = best.get((hadm, "bun"), {})
        lact = best.get((hadm, "lactate"), {})

        bun_value = bun.get("value", np.nan)
        rows.append({
            "subject_id": int(row.subject_id),
            "hadm_id": hadm,
            "ph_anytime": ph.get("value", np.nan),
            "ph_anytime_charttime": ph.get("charttime", pd.NaT),
            "ph_anytime_itemid": ph.get("itemid", np.nan),
            "bun_anytime": bun_value,
            "bun_anytime_charttime": bun.get("charttime", pd.NaT),
            "bun_anytime_itemid": bun.get("itemid", np.nan),
            "urea_anytime": bun_value * BUN_TO_UREA_FACTOR if pd.notna(bun_value) else np.nan,
            "lactate_anytime": lact.get("value", np.nan),
            "lactate_anytime_charttime": lact.get("charttime", pd.NaT),
            "lactate_anytime_itemid": lact.get("itemid", np.nan),
        })

    labs = pd.DataFrame(rows)
    return labs, item_table


def load_coefficients(internal_model_dir: Path) -> tuple[float, dict[str, float], str]:
    path = find_file(internal_model_dir, "model_coefficients_for_figures.csv")
    coef = normalize_columns(pd.read_csv(path))

    var_col = first_existing(coef, ["variable", "term", "feature", "predictor"])
    beta_col = first_existing(coef, ["beta", "coef", "coefficient", "estimate"])

    if var_col is None or beta_col is None:
        raise ValueError("model_coefficients_for_figures.csv must contain variable and beta columns.")

    intercept = None
    betas = {}

    for _, row in coef.iterrows():
        var = str(row[var_col]).strip()
        beta = float(row[beta_col])
        if var.lower() in ["const", "intercept", "(intercept)"]:
            intercept = beta
        elif var in MODEL_FEATURES:
            betas[var] = beta

    if intercept is None:
        raise ValueError("No const/intercept row found in model_coefficients_for_figures.csv.")

    missing = [f for f in MODEL_FEATURES if f not in betas]
    if missing:
        raise ValueError(f"Coefficient file missing model features: {missing}")

    return float(intercept), {k: float(v) for k, v in betas.items()}, str(path)


def load_frozen_adjustment(internal_model_dir: Path) -> tuple[float, float, str]:
    try:
        path = find_file(internal_model_dir, "recalibration_parameters.json")
        params = json.loads(path.read_text(encoding="utf-8"))
        intercept_key = "re" + "calibration_intercept"
        slope_key = "re" + "calibration_slope"
        return float(params[intercept_key]), float(params[slope_key]), str(path)
    except FileNotFoundError:
        pass

    path = find_file(internal_model_dir, "summary_internal_validation.csv")
    s = normalize_columns(pd.read_csv(path))
    intercept_col = "re" + "calibration_intercept"
    slope_col = "re" + "calibration_slope"
    return float(s.loc[0, intercept_col]), float(s.loc[0, slope_col]), str(path)


def load_medians(internal_model_dir: Path, medians_csv: str | None = None) -> tuple[dict[str, float], str]:
    if medians_csv:
        path = Path(medians_csv)
        med = normalize_columns(pd.read_csv(path))
        var_col = first_existing(med, ["variable", "term", "feature", "predictor"])
        val_col = first_existing(med, ["median", "value", "imputation_median"])
        if var_col is None or val_col is None:
            raise ValueError("Medians CSV must contain variable and median/value columns.")
        out = {str(r[var_col]).strip(): float(r[val_col]) for _, r in med.iterrows() if str(r[var_col]).strip() in MODEL_FEATURES}
        missing = [f for f in MODEL_FEATURES if f not in out]
        if missing:
            raise ValueError(f"Medians CSV missing features: {missing}")
        return out, str(path)

    path = find_file(internal_model_dir, "analysis_dataset_internal_validation.csv")
    data = normalize_columns(pd.read_csv(path))
    out = {f: float(pd.to_numeric(data[f], errors="coerce").median()) for f in MODEL_FEATURES}
    return out, str(path)


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-8, 1 - 1e-8)
    return np.log(p / (1 - p))


def inv_logit(lp: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(lp, dtype=float)))


def regenerate_predictions(df: pd.DataFrame, internal_model_dir: Path, medians_csv: str | None = None) -> tuple[pd.DataFrame, dict]:
    intercept, betas, coef_source = load_coefficients(internal_model_dir)
    frozen_i, frozen_s, frozen_source = load_frozen_adjustment(internal_model_dir)
    medians, med_source = load_medians(internal_model_dir, medians_csv)

    required_base = ["age", "history_hf", "history_af"]
    missing_base = [c for c in required_base if c not in df.columns]
    if missing_base:
        raise ValueError(f"Original validation file is missing base predictors: {missing_base}")

    X = pd.DataFrame(index=df.index)
    X["age"] = pd.to_numeric(df["age"], errors="coerce")
    X["history_hf"] = pd.to_numeric(df["history_hf"], errors="coerce")
    X["history_af"] = pd.to_numeric(df["history_af"], errors="coerce")
    X["ph"] = pd.to_numeric(df["ph_anytime"], errors="coerce")
    X["urea"] = pd.to_numeric(df["urea_anytime"], errors="coerce")
    X["lactate"] = pd.to_numeric(df["lactate_anytime"], errors="coerce")

    pred = pd.DataFrame(index=df.index)
    for f in MODEL_FEATURES:
        pred[f"{f}_anytime_missing_before_imputation"] = X[f].isna().astype(int)
        X[f] = X[f].fillna(medians[f])
        pred[f"{f}_model_input_anytime"] = X[f].astype(float)

    lp_original = np.full(len(X), intercept, dtype=float)
    for f in MODEL_FEATURES:
        lp_original += betas[f] * X[f].astype(float).to_numpy()

    p_original = inv_logit(lp_original)

    lp_frozen = frozen_i + frozen_s * logit(p_original)
    p_frozen = inv_logit(lp_frozen)

    pred["linear_predictor_original_anytime"] = lp_original
    pred["predicted_probability_original_anytime"] = p_original
    pred["linear_predictor_frozen_anytime"] = lp_frozen
    pred["predicted_probability_frozen_anytime"] = p_frozen
    pred["predicted_probability_frozen"] = p_frozen

    metadata = {
        "coefficient_source": coef_source,
        "frozen_adjustment_source": frozen_source,
        "medians_source": med_source,
        "intercept": intercept,
        "betas": betas,
        "frozen_adjustment_intercept": frozen_i,
        "frozen_adjustment_slope": frozen_s,
        "medians": medians,
        "definition": "Original MIMIC-IV validation cohort/outcome preserved; only pH, urea, and lactate replaced by first plausible anytime labevents values, with remaining missing values median-imputed.",
    }

    return pred, metadata


def bootstrap_auc_ci(y: np.ndarray, p: np.ndarray, n_boot: int = 2000, seed: int = RANDOM_STATE):
    rng = np.random.default_rng(seed)
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


def calibration_stats(y_true, y_prob):
    eps = 1e-8
    p = np.clip(np.asarray(y_prob, dtype=float), eps, 1 - eps)
    y = np.asarray(y_true, dtype=int)

    if np.unique(y).size < 2:
        return np.nan, np.nan

    logit_p = np.log(p / (1 - p))
    X = sm.add_constant(logit_p, has_constant="add")

    try:
        model = sm.Logit(y, X).fit(disp=False, maxiter=1000)
        vals = np.asarray(model.params, dtype=float)
        return float(vals[0]), float(vals[1])
    except Exception:
        return np.nan, np.nan


def wilson_ci(k: np.ndarray, n: np.ndarray, z: float = 1.96):
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


def bootstrap_net_benefit_ci(y: np.ndarray, p: np.ndarray, thresholds: np.ndarray, n_boot: int = 300, seed: int = RANDOM_STATE):
    rng = np.random.default_rng(seed)
    n = len(y)
    boots = np.zeros((n_boot, len(thresholds)), dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b, :] = net_benefit(y[idx], p[idx], thresholds)
    return np.mean(boots, axis=0), np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)


def make_roc_plot(y, p, outpath: Path, table_out: Path):
    auc = roc_auc_score(y, p)
    lo, hi = bootstrap_auc_ci(y, p, seed=101)
    fpr, tpr, _ = roc_curve(y, p)

    rows = [{
        "model": "AECOPD-CV model",
        "auc": auc,
        "auc_ci_low": lo,
        "auc_ci_high": hi,
        "shown_in_figure": 1,
    }]

    plt.figure(figsize=(6.8, 6.0))
    plt.plot(
        fpr,
        tpr,
        linewidth=2.4,
        label=f"AECOPD-CV model (AUC {auc:.3f}, 95% CI {lo:.3f}–{hi:.3f})",
    )
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

    pd.DataFrame(rows).to_csv(table_out, index=False)
    return rows


def make_calibration_plot(y: np.ndarray, p: np.ndarray, outpath: Path, table_out: Path):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=int)
    df = pd.DataFrame({"y": y, "pred": p}).sort_values("pred").reset_index(drop=True)

    x_max_plot = min(0.90, float(np.quantile(df["pred"], 0.98)))
    df_plot = df[df["pred"] <= x_max_plot].copy()
    if len(df_plot) < 30:
        df_plot = df.copy()
        x_max_plot = min(0.90, float(df["pred"].max()))

    curve = lowess(df_plot["y"], df_plot["pred"], frac=0.72, it=0, return_sorted=True)
    grid = np.linspace(float(df_plot["pred"].min()), float(df_plot["pred"].max()), 160)
    rng = np.random.default_rng(1702)
    boot_curves = []

    for _ in range(180):
        idx = rng.integers(0, len(df_plot), size=len(df_plot))
        boot = df_plot.iloc[idx].sort_values("pred")
        lo_b = lowess(boot["y"], boot["pred"], frac=0.72, it=0, return_sorted=True)
        x_b = np.asarray(lo_b[:, 0], dtype=float)
        y_b = np.asarray(lo_b[:, 1], dtype=float)
        uniq_idx = np.unique(x_b, return_index=True)[1]
        x_u = x_b[np.sort(uniq_idx)]
        y_u = y_b[np.sort(uniq_idx)]
        if len(x_u) < 2:
            continue
        boot_curves.append(np.interp(grid, x_u, y_u, left=y_u[0], right=y_u[-1]))

    cal_int, cal_slope = calibration_stats(y, p)

    plt.figure(figsize=(6.2, 5.6))
    plt.plot([0, x_max_plot], [0, x_max_plot], linestyle="--", linewidth=1.2, label="Ideal calibration")
    if len(boot_curves) > 20:
        boot_curves = np.asarray(boot_curves)
        plt.fill_between(
            grid,
            np.clip(np.percentile(boot_curves, 2.5, axis=0), 0, 0.95),
            np.clip(np.percentile(boot_curves, 97.5, axis=0), 0, 0.95),
            alpha=0.12,
            label="LOWESS 95% CI",
        )
    plt.plot(curve[:, 0], curve[:, 1], linewidth=2.4, label="LOWESS-smoothed calibration")
    plt.xlim(0, x_max_plot)
    plt.ylim(0, max(0.75, min(0.95, float(np.nanmax(curve[:, 1])) + 0.08)))
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed event rate")
    plt.title("External validation calibration plot")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()

    result = {"model": "AECOPD-CV model", "calibration_intercept": cal_int, "calibration_slope": cal_slope, "x_max_plot": x_max_plot}
    pd.DataFrame([result]).to_csv(table_out, index=False)
    return result


def make_dca_plot(y, p, outpath: Path, table_out: Path):
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

    pd.DataFrame({
        "threshold": thresholds,
        "aecopd_cv_model_nb": mid,
        "aecopd_cv_model_nb_low": lo,
        "aecopd_cv_model_nb_high": hi,
        "treat_all_nb": all_nb,
        "treat_none_nb": none_nb,
    }).to_csv(table_out, index=False)


def make_quartile_plot(y, p, outpath: Path, table_out: Path):
    df = pd.DataFrame({"y": y, "pred": p}).dropna()
    try:
        df["quartile"] = pd.qcut(df["pred"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
    except ValueError:
        df["quartile"] = pd.cut(df["pred"], bins=4, labels=["Q1", "Q2", "Q3", "Q4"], include_lowest=True)

    g = df.groupby("quartile", observed=False).agg(
        n=("y", "size"),
        events=("y", "sum"),
        event_rate=("y", "mean"),
        mean_pred=("pred", "mean"),
    ).reset_index()
    g["ci_low"], g["ci_high"] = wilson_ci(g["events"].values, g["n"].values)

    x = np.arange(len(g))
    plt.figure(figsize=(6.4, 5.2))
    lower_err = np.maximum(g["event_rate"].to_numpy() - g["ci_low"].to_numpy(), 0)
    upper_err = np.maximum(g["ci_high"].to_numpy() - g["event_rate"].to_numpy(), 0)

    plt.errorbar(
        x, g["event_rate"],
        yerr=[lower_err, upper_err],
        fmt="o", capsize=4, linewidth=1.5, markersize=8,
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

    g["model"] = "AECOPD-CV model"
    g.to_csv(table_out, index=False)


def make_composite_figure(roc_path: Path, dca_path: Path, cal_path: Path, quart_path: Path, outpath: Path):
    imgs = [Image.open(p).convert("RGB") for p in [roc_path, dca_path, cal_path, quart_path]]
    w = max(img.width for img in imgs)
    h = max(img.height for img in imgs)

    canvas = Image.new("RGB", (2 * w, 2 * h), "white")
    labels = ["(A)", "(B)", "(C)", "(D)"]
    draw = ImageDraw.Draw(canvas)

    for idx, img in enumerate(imgs):
        x = (idx % 2) * w
        y = (idx // 2) * h
        canvas.paste(img.resize((w, h)), (x, y))
        draw.text((x + 10, y + 10), labels[idx], fill="black")

    canvas.save(outpath)


def availability_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["ph_anytime", "urea_anytime", "lactate_anytime"]
    rows = []
    for c in cols:
        x = pd.to_numeric(df[c], errors="coerce")
        rows.append({
            "variable": c,
            "available_n": int(x.notna().sum()),
            "available_percent": round(float(x.notna().mean() * 100), 1),
            "missing_n": int(x.isna().sum()),
            "missing_percent": round(float(x.isna().mean() * 100), 1),
        })
    all_labs = df[cols].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
    rows.append({
        "variable": "all_three_anytime_labs_complete",
        "available_n": int(all_labs.sum()),
        "available_percent": round(float(all_labs.mean() * 100), 1),
        "missing_n": int((~all_labs).sum()),
        "missing_percent": round(float((~all_labs).mean() * 100), 1),
    })
    return pd.DataFrame(rows)


def run_validation(df: pd.DataFrame, outdir: Path) -> dict:
    y = pd.to_numeric(df["cv_event"], errors="coerce").astype(int).to_numpy()
    p = pd.to_numeric(df["predicted_probability_frozen"], errors="coerce").to_numpy(dtype=float)

    roc_path = outdir / "figure_roc_external_validation_anytime.png"
    dca_path = outdir / "figure_decision_curve_external_validation_anytime.png"
    cal_path = outdir / "figure_calibration_external_validation_anytime.png"
    quart_path = outdir / "figure_quartiles_external_validation_anytime.png"

    roc_rows = make_roc_plot(y, p, roc_path, outdir / "table_roc_external_validation_anytime.csv")
    cal = make_calibration_plot(y, p, cal_path, outdir / "table_calibration_external_validation_anytime.csv")
    make_dca_plot(y, p, dca_path, outdir / "table_decision_curve_external_validation_anytime.csv")
    make_quartile_plot(y, p, quart_path, outdir / "table_quartiles_external_validation_anytime.csv")

    try:
        make_composite_figure(roc_path, dca_path, cal_path, quart_path, outdir / "External_Validation_Composite_anytime.png")
    except Exception:
        pass

    roc_tbl = pd.DataFrame(roc_rows)
    summary = {
        "n": int(len(df)),
        "events": int(y.sum()),
        "event_rate": float(np.mean(y)),
        "auc": float(roc_tbl.loc[roc_tbl["shown_in_figure"] == 1, "auc"].iloc[0]),
        "auc_ci_low": float(roc_tbl.loc[roc_tbl["shown_in_figure"] == 1, "auc_ci_low"].iloc[0]),
        "auc_ci_high": float(roc_tbl.loc[roc_tbl["shown_in_figure"] == 1, "auc_ci_high"].iloc[0]),
        "brier": float(brier_score_loss(y, p)),
        "calibration_intercept": float(cal["calibration_intercept"]),
        "calibration_slope": float(cal["calibration_slope"]),
    }
    return summary


def main() -> None:
    style()
    outdir = OUTPUT_DIR
    outdir.mkdir(parents=True, exist_ok=True)

    original = load_original_validation(ORIGINAL_VALIDATION, ORIGINAL_SHEET)

    if "history_hf" not in original.columns and "HF_history" in original.columns:
        original["history_hf"] = original["HF_history"]
    if "history_af" not in original.columns and "AF_history" in original.columns:
        original["history_af"] = original["AF_history"]

    labs6, item_table = extract_first_labs_anytime(original, MIMIC_ROOT, hours=LAB_WINDOW_HOURS)
    item_table.to_csv(outdir / "lab_itemids_used_anytime.csv", index=False)

    merged = original.merge(labs6, on=["subject_id", "hadm_id"], how="left", validate="one_to_one")

    pred, metadata = regenerate_predictions(merged, INTERNAL_MODEL_DIR, MEDIANS_CSV)

    columns_to_replace = [
        "linear_predictor_original",
        "predicted_probability_original",
        "linear_predictor_frozen",
        "predicted_probability_frozen",
        "predicted_risk_original",
    ]
    merged_no_old_predictions = merged.drop(
        columns=[c for c in columns_to_replace if c in merged.columns],
        errors="ignore",
    )

    final_df = pd.concat([merged_no_old_predictions.reset_index(drop=True), pred.reset_index(drop=True)], axis=1)
    final_df["predicted_risk_original"] = pd.to_numeric(final_df["predicted_probability_frozen"], errors="coerce")

    final_df.to_csv(outdir / "patient_level_external_validation_predictions_MIMIC-IV_anytime_labs.csv", index=False)

    with pd.ExcelWriter(outdir / "Database_MIMIC-IV_external_validation_anytime_labs_predictions.xlsx", engine="openpyxl", mode="w") as writer:
        final_df.to_excel(writer, sheet_name="AECOPD_validation_dataset_anytime", index=False)
        availability_table(final_df).to_excel(writer, sheet_name="lab_availability_anytime", index=False)
        item_table.to_excel(writer, sheet_name="lab_itemids_used", index=False)

    avail = availability_table(final_df)
    avail.to_csv(outdir / "lab_availability_anytime.csv", index=False)

    outcome_counts = pd.DataFrame([{
        "cv_event_database_original": int(final_df["cv_event_database_original"].sum()) if "cv_event_database_original" in final_df.columns else np.nan,
        "mi_inclusive_event": int(final_df["mi_inclusive_event"].sum()),
        "arrhythmia_inclusive_event": int(final_df["arrhythmia_inclusive_event"].sum()),
        "hf_decompensation_event": int(final_df["hf_decompensation_event"].sum()),
        "pe_inclusive_event": int(final_df["pe_inclusive_event"].sum()),
        "cv_event_final": int(final_df["cv_event"].sum()),
    }])
    outcome_counts.to_csv(outdir / "outcome_definition_check_anytime.csv", index=False)

    summary = run_validation(final_df, outdir)
    summary["lab_window_hours"] = LAB_WINDOW_HOURS
    summary["original_validation_file"] = str(ORIGINAL_VALIDATION)
    summary["original_sheet"] = ORIGINAL_SHEET
    summary["mimic_root"] = str(MIMIC_ROOT)
    summary["internal_model_dir"] = str(INTERNAL_MODEL_DIR)
    summary["metadata"] = metadata

    pd.DataFrame([summary]).to_csv(outdir / "summary_external_validation_anytime.csv", index=False)
    (outdir / "summary_external_validation_anytime.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (outdir / "frozen_prediction_generation_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Done.")
    print(f"Rows analysed: {summary['n']}")
    print(f"Events preserved from final outcome: {summary['events']}")
    print(f"AUC anytime labs: {summary['auc']:.3f} ({summary['auc_ci_low']:.3f}-{summary['auc_ci_high']:.3f})")
    print(f"Output: {outdir}")
    print("")
    print("Outcome definition check:")
    print(outcome_counts.to_string(index=False))
    print("")
    print("anytime lab availability:")
    print(avail.to_string(index=False))


if __name__ == "__main__":
    main()
