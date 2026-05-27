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
LAB_END_MIN = 360
WINDOW_LABEL = "6h"
CORE = ["age", "history_hf", "history_af", "ph", "urea", "lactate"]

EMBEDDED_COEFFICIENTS = {
    "const": 46.95019433422684,
    "age": 0.0056297841639299,
    "history_hf": 0.5535675156229428,
    "history_af": 1.2434884123076857,
    "ph": -6.746131471731943,
    "urea": 0.0069608750607488,
    "lactate": 0.1190962972054364,
}

EMBEDDED_MEDIANS = {
    "age": 72.0,
    "history_hf": 0.0,
    "history_af": 0.0,
    "ph": 7.39,
    "urea": 34.05,
    "lactate": 1.2,
}


def style() -> None:
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


def find_file(folder: Path, stem: str, required: bool = True) -> Path | None:
    paths = []
    for suffix in [".csv.gz", ".csv", ".CSV.GZ", ".CSV"]:
        paths.append(folder / f"{stem}{suffix}")
        paths.extend(folder.rglob(f"{stem}{suffix}"))
    seen = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        if path.exists():
            return path
    if required:
        raise FileNotFoundError(f"Missing eICU file: {stem}.csv.gz or {stem}.csv")
    return None


def read_csv(path: Path) -> pd.DataFrame:
    print(f"Reading {path}")
    return pd.read_csv(path, low_memory=False)


def normalize_columns(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data.columns = [str(column).strip() for column in data.columns]
    return data


def text(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().str.strip()


def first_column(data: pd.DataFrame, names: list[str]) -> str | None:
    lookup = {column.lower(): column for column in data.columns}
    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]
    return None


def load_coefficients(path: Path | None) -> dict[str, float]:
    if path is not None and path.exists():
        data = normalize_columns(pd.read_csv(path))
        if {"variable", "beta"}.issubset(data.columns):
            data["variable"] = data["variable"].astype(str).str.strip()
            data["beta"] = pd.to_numeric(data["beta"], errors="coerce")
            values = dict(zip(data["variable"], data["beta"]))
            if all(name in values and pd.notna(values[name]) for name in ["const"] + CORE):
                return {name: float(values[name]) for name in ["const"] + CORE}
    return EMBEDDED_COEFFICIENTS.copy()


def load_medians(path: Path | None) -> dict[str, float]:
    if path is not None and path.exists():
        data = normalize_columns(pd.read_csv(path))
        if {"variable", "median"}.issubset(data.columns):
            data["variable"] = data["variable"].astype(str).str.strip()
            data["median"] = pd.to_numeric(data["median"], errors="coerce")
            values = dict(zip(data["variable"], data["median"]))
            if all(name in values and pd.notna(values[name]) for name in CORE):
                return {name: float(values[name]) for name in CORE}
    return EMBEDDED_MEDIANS.copy()


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -50, 50)
    return 1.0 / (1.0 + np.exp(-values))


def predict_frozen(data: pd.DataFrame, coefficients: dict[str, float]) -> np.ndarray:
    lp = np.full(len(data), coefficients["const"], dtype=float)
    for variable in CORE:
        lp += coefficients[variable] * pd.to_numeric(data[variable], errors="coerce").to_numpy(dtype=float)
    return sigmoid(lp)


def prepare_age(patient: pd.DataFrame) -> pd.Series:
    age = patient["age"].astype(str).str.strip().replace({"> 89": "90", ">89": "90", "90+": "90"})
    return pd.to_numeric(age, errors="coerce")


def extract_aecopd_ids(diagnosis: pd.DataFrame) -> set[int]:
    dx = text(diagnosis["diagnosisstring"])
    copd = dx.str.contains(r"\bcopd\b|chronic obstructive pulmonary", regex=True, na=False)
    acute = dx.str.contains(r"exacerb|acute|respiratory failure|bronchitis|bronchospasm", regex=True, na=False)
    ids = set(diagnosis.loc[copd & acute, "patientunitstayid"].dropna().astype(int))
    if not ids:
        ids = set(diagnosis.loc[copd, "patientunitstayid"].dropna().astype(int))
    return ids


def extract_history(past_history: pd.DataFrame, ids: set[int]) -> pd.DataFrame:
    out = pd.DataFrame({"patientunitstayid": sorted(ids)})
    columns = [column for column in ["pasthistoryvalue", "pasthistorypath", "pasthistorynotetype"] if column in past_history.columns]
    if columns:
        content = text(past_history[columns[0]])
        for column in columns[1:]:
            content = content + " " + text(past_history[column])
    else:
        content = past_history.astype(str).agg(" ".join, axis=1).str.lower()
    frame = past_history.copy()
    frame["content"] = content
    frame = frame[frame["patientunitstayid"].isin(ids)].copy()
    hf = frame["content"].str.contains(r"heart failure|congestive heart failure|cardiac failure|\bchf\b", regex=True, na=False)
    af = frame["content"].str.contains(r"atrial fibrillation|atrial flutter|\bafib\b|\baf\b", regex=True, na=False)
    out["history_hf"] = out["patientunitstayid"].isin(frame.loc[hf, "patientunitstayid"].dropna().astype(int)).astype(int)
    out["history_af"] = out["patientunitstayid"].isin(frame.loc[af, "patientunitstayid"].dropna().astype(int)).astype(int)
    return out


def extract_events(diagnosis: pd.DataFrame, ids: set[int]) -> pd.DataFrame:
    frame = diagnosis[diagnosis["patientunitstayid"].isin(ids)].copy()
    dx = text(frame["diagnosisstring"])
    chronic = dx.str.contains(r"history of|past history|hx of|chronic|known|baseline|prior|previous|old ", regex=True, na=False)
    mi = dx.str.contains(r"myocardial infarction|nstemi|stemi|acute mi|non.?st elevation|st elevation", regex=True, na=False)
    mi = mi & ~dx.str.contains(r"old myocardial infarction|old mi|history of|hx of|prior|previous", regex=True, na=False)
    pe = dx.str.contains(r"pulmonary embolism|\bpe\b", regex=True, na=False) & ~chronic
    edema = dx.str.contains(r"pulmonary edema|pulmonary oedema|acute pulmonary edema|acute pulmonary oedema", regex=True, na=False) & ~chronic
    arrhythmia = dx.str.contains(r"atrial fibrillation|atrial flutter|arrhythmia|tachyarrhythmia|supraventricular tachycardia|\bsvt\b|rapid ventricular response|\brvr\b", regex=True, na=False) & ~chronic
    out = pd.DataFrame({"patientunitstayid": sorted(ids)})
    out["mi_event"] = out["patientunitstayid"].isin(frame.loc[mi, "patientunitstayid"].dropna().astype(int)).astype(int)
    out["pe_event"] = out["patientunitstayid"].isin(frame.loc[pe, "patientunitstayid"].dropna().astype(int)).astype(int)
    out["pulmonary_edema_event"] = out["patientunitstayid"].isin(frame.loc[edema, "patientunitstayid"].dropna().astype(int)).astype(int)
    out["acute_arrhythmia_event"] = out["patientunitstayid"].isin(frame.loc[arrhythmia, "patientunitstayid"].dropna().astype(int)).astype(int)
    out["cv_event"] = ((out["mi_event"] == 1) | (out["pe_event"] == 1) | (out["pulmonary_edema_event"] == 1) | (out["acute_arrhythmia_event"] == 1)).astype(int)
    strings = frame.groupby("patientunitstayid")["diagnosisstring"].apply(lambda values: " | ".join(sorted(set(map(str, values.dropna()))))[:3000]).reset_index(name="diagnosis_strings")
    return out.merge(strings, on="patientunitstayid", how="left")


def extract_labs(lab: pd.DataFrame, ids: set[int], start_min: int, end_min: int) -> pd.DataFrame:
    lab = lab[lab["patientunitstayid"].isin(ids)].copy()
    offset_column = first_column(lab, ["labresultoffset", "labresultrevisedoffset"])
    if offset_column is None:
        raise ValueError("The lab table must contain labresultoffset or labresultrevisedoffset.")
    lab["lab_name"] = text(lab["labname"])
    lab["lab_value"] = pd.to_numeric(lab["labresult"], errors="coerce")
    lab["lab_offset"] = pd.to_numeric(lab[offset_column], errors="coerce")
    lab = lab.dropna(subset=["lab_value", "lab_offset"])
    lab = lab[(lab["lab_offset"] >= start_min) & (lab["lab_offset"] <= end_min)].copy()
    ph_mask = lab["lab_name"].str.fullmatch(r"ph", na=False) | lab["lab_name"].str.contains(r"\bph\b", regex=True, na=False)
    lactate_mask = lab["lab_name"].str.contains(r"lactate|lactic acid", regex=True, na=False)
    bun_mask = lab["lab_name"].str.contains(r"\bbun\b|blood urea nitrogen", regex=True, na=False)
    urea_mask = lab["lab_name"].str.contains(r"\burea\b", regex=True, na=False) & ~bun_mask

    def first_value(mask: pd.Series, name: str) -> pd.DataFrame:
        part = lab.loc[mask, ["patientunitstayid", "lab_offset", "lab_value", "labname"]].copy()
        if part.empty:
            return pd.DataFrame(columns=["patientunitstayid", name, f"{name}_offset_min", f"{name}_labname"])
        index = part.sort_values(["patientunitstayid", "lab_offset"]).groupby("patientunitstayid").head(1).index
        part = part.loc[index].rename(columns={"lab_value": name, "lab_offset": f"{name}_offset_min", "labname": f"{name}_labname"})
        return part[["patientunitstayid", name, f"{name}_offset_min", f"{name}_labname"]]

    out = pd.DataFrame({"patientunitstayid": sorted(ids)})
    out = out.merge(first_value(ph_mask, "ph"), on="patientunitstayid", how="left")
    out = out.merge(first_value(lactate_mask, "lactate"), on="patientunitstayid", how="left")
    out = out.merge(first_value(bun_mask, "bun"), on="patientunitstayid", how="left")
    out = out.merge(first_value(urea_mask, "urea_direct"), on="patientunitstayid", how="left")
    out["urea_from_bun"] = out["bun"] * 2.14
    out["urea"] = out["urea_from_bun"].where(out["urea_from_bun"].notna(), out["urea_direct"])
    out["urea_source"] = np.where(out["urea_from_bun"].notna(), "bun_converted", np.where(out["urea_direct"].notna(), "direct_urea", "missing"))
    return out


def select_first_stay(data: pd.DataFrame) -> pd.DataFrame:
    if "uniquepid" not in data.columns:
        return data.copy()
    sort_column = None
    for column in ["hospitaladmitoffset", "unitadmitoffset", "unitadmittime24", "patientunitstayid"]:
        if column in data.columns:
            sort_column = column
            break
    out = data.copy()
    out["_sort_key"] = pd.to_numeric(out[sort_column], errors="coerce") if sort_column else pd.to_numeric(out["patientunitstayid"], errors="coerce")
    out = out.sort_values(["uniquepid", "_sort_key", "patientunitstayid"]).groupby("uniquepid", as_index=False).first()
    return out.drop(columns=["_sort_key"], errors="ignore")


def build_dataset(eicu_dir: Path, lab_start_min: int, lab_end_min: int, medians: dict[str, float]) -> pd.DataFrame:
    patient = normalize_columns(read_csv(find_file(eicu_dir, "patient")))
    diagnosis = normalize_columns(read_csv(find_file(eicu_dir, "diagnosis")))
    lab = normalize_columns(read_csv(find_file(eicu_dir, "lab")))
    past_history = normalize_columns(read_csv(find_file(eicu_dir, "pastHistory")))
    ids = extract_aecopd_ids(diagnosis)
    events = extract_events(diagnosis, ids)
    history = extract_history(past_history, ids)
    labs = extract_labs(lab, ids, lab_start_min, lab_end_min)
    patient = patient[patient["patientunitstayid"].isin(ids)].copy()
    patient["age"] = prepare_age(patient)
    columns = ["patientunitstayid", "age"]
    for column in ["uniquepid", "gender", "hospitalid", "hospitaladmitoffset", "unitadmitoffset", "unitadmittime24"]:
        if column in patient.columns:
            columns.append(column)
    data = patient[columns].merge(history, on="patientunitstayid", how="left")
    data = data.merge(labs, on="patientunitstayid", how="left")
    data = data.merge(events, on="patientunitstayid", how="left")
    for column in ["history_hf", "history_af", "cv_event", "mi_event", "pe_event", "pulmonary_edema_event", "acute_arrhythmia_event"]:
        data[column] = data[column].fillna(0).astype(int)
    for variable in ["age", "ph", "urea", "lactate"]:
        data[variable] = pd.to_numeric(data[variable], errors="coerce")
    for variable in ["ph", "urea", "lactate"]:
        data[f"{variable}_missing"] = data[variable].isna().astype(int)
    plausible = (
        (data["age"].isna() | data["age"].between(18, 110))
        & (data["ph"].isna() | data["ph"].between(6.5, 8.0))
        & (data["lactate"].isna() | data["lactate"].between(0, 40))
        & (data["urea"].isna() | data["urea"].between(0, 500))
    )
    data = data[plausible].copy()
    data = select_first_stay(data)
    for variable in CORE:
        data[variable] = pd.to_numeric(data[variable], errors="coerce").fillna(medians[variable])
    data["ph"] = data["ph"].clip(6.8, 7.8)
    data["lactate"] = data["lactate"].clip(0.0, 20.0)
    data["urea"] = data["urea"].clip(0.0, 300.0)
    return data


def bootstrap_auc_ci(y: np.ndarray, p: np.ndarray, n_boot: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(RANDOM_STATE)
    values = []
    n = len(y)
    for _ in range(n_boot):
        index = rng.integers(0, n, size=n)
        if np.unique(y[index]).size < 2:
            continue
        values.append(roc_auc_score(y[index], p[index]))
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def calibration_stats(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    p = np.clip(p.astype(float), 1e-8, 1 - 1e-8)
    logit_p = np.log(p / (1 - p))
    design = sm.add_constant(logit_p, has_constant="add")
    model = sm.Logit(y.astype(int), design).fit(disp=False, maxiter=1000)
    values = np.asarray(model.params, dtype=float)
    return float(values[0]), float(values[1])


def wilson_ci(events: np.ndarray, total: np.ndarray, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    events = events.astype(float)
    total = total.astype(float)
    proportion = np.divide(events, total, out=np.zeros_like(events), where=total > 0)
    denominator = 1 + z**2 / total
    centre = (proportion + z**2 / (2 * total)) / denominator
    half_width = (z * np.sqrt((proportion * (1 - proportion) / total) + z**2 / (4 * total**2))) / denominator
    return np.clip(centre - half_width, 0, 1), np.clip(centre + half_width, 0, 1)


def net_benefit(y: np.ndarray, p: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    result = np.zeros(len(thresholds), dtype=float)
    n = len(y)
    for i, threshold in enumerate(thresholds):
        selected = p >= threshold
        tp = np.sum(selected & (y == 1))
        fp = np.sum(selected & (y == 0))
        result[i] = (tp / n) - (fp / n) * (threshold / (1 - threshold))
    return result


def treat_all_net_benefit(y: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    prevalence = np.mean(y.astype(int))
    return prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))


def smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return values.copy()
    kernel = np.ones(window) / window
    padded = np.pad(values.astype(float), (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def bootstrap_net_benefit_ci(y: np.ndarray, p: np.ndarray, thresholds: np.ndarray, n_boot: int = 300) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(RANDOM_STATE + 84)
    n = len(y)
    matrix = np.zeros((n_boot, len(thresholds)), dtype=float)
    for row in range(n_boot):
        index = rng.integers(0, n, size=n)
        matrix[row, :] = net_benefit(y[index], p[index], thresholds)
    return matrix.mean(axis=0), np.percentile(matrix, 2.5, axis=0), np.percentile(matrix, 97.5, axis=0)


def make_roc(y: np.ndarray, p: np.ndarray, output: Path, table: Path, label: str) -> dict[str, float]:
    auc = roc_auc_score(y, p)
    low, high = bootstrap_auc_ci(y, p)
    fpr, tpr, _ = roc_curve(y, p)
    plt.figure(figsize=(6.8, 6.0))
    plt.plot(fpr, tpr, linewidth=2.4, label=f"{label} (AUC {auc:.3f}, 95% CI {low:.3f}–{high:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("External validation ROC curve")
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight")
    plt.close()
    result = {"model": label, "auc": auc, "auc_ci_low": low, "auc_ci_high": high}
    pd.DataFrame([result]).to_csv(table, index=False)
    return result


def make_calibration(y: np.ndarray, p: np.ndarray, output: Path, table: Path, label: str) -> dict[str, float]:
    p = np.clip(p.astype(float), 1e-6, 1 - 1e-6)
    frame = pd.DataFrame({"y": y.astype(int), "pred": p}).sort_values("pred").reset_index(drop=True)
    xmax = min(0.90, float(np.quantile(frame["pred"], 0.98)))
    plot_frame = frame[frame["pred"] <= xmax].copy()
    if len(plot_frame) < 30:
        plot_frame = frame.copy()
        xmax = min(0.90, float(frame["pred"].max()))
    curve = lowess(plot_frame["y"], plot_frame["pred"], frac=0.72, it=0, return_sorted=True)
    grid = np.linspace(float(plot_frame["pred"].min()), float(plot_frame["pred"].max()), 160)
    rng = np.random.default_rng(RANDOM_STATE + 17)
    curves = []
    for _ in range(180):
        index = rng.integers(0, len(plot_frame), size=len(plot_frame))
        sample = plot_frame.iloc[index].sort_values("pred")
        boot = lowess(sample["y"], sample["pred"], frac=0.72, it=0, return_sorted=True)
        x = np.asarray(boot[:, 0], dtype=float)
        z = np.asarray(boot[:, 1], dtype=float)
        unique = np.unique(x, return_index=True)[1]
        x = x[np.sort(unique)]
        z = z[np.sort(unique)]
        if len(x) >= 2:
            curves.append(np.interp(grid, x, z, left=z[0], right=z[-1]))
    intercept, slope = calibration_stats(y, p)
    plt.figure(figsize=(6.2, 5.6))
    plt.plot([0, xmax], [0, xmax], linestyle="--", linewidth=1.2, label="Ideal calibration")
    if len(curves) > 20:
        curves = np.asarray(curves)
        plt.fill_between(grid, np.clip(np.percentile(curves, 2.5, axis=0), 0, 0.95), np.clip(np.percentile(curves, 97.5, axis=0), 0, 0.95), alpha=0.12, label="LOWESS 95% CI")
    plt.plot(curve[:, 0], curve[:, 1], linewidth=2.4, label="LOWESS-smoothed calibration")
    plt.xlim(0, xmax)
    plt.ylim(0, max(0.75, min(0.95, float(np.nanmax(curve[:, 1])) + 0.08)))
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed event rate")
    plt.title("External validation calibration plot")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight")
    plt.close()
    result = {"model": label, "calibration_intercept": intercept, "calibration_slope": slope, "x_max_plot": xmax}
    pd.DataFrame([result]).to_csv(table, index=False)
    return result


def make_dca(y: np.ndarray, p: np.ndarray, output: Path, table: Path, label: str) -> None:
    thresholds = np.arange(0.10, 0.50 + 1e-9, 0.01)
    mid, low, high = bootstrap_net_benefit_ci(y, p, thresholds)
    all_strategy = treat_all_net_benefit(y, thresholds)
    none_strategy = np.zeros_like(thresholds)
    plt.figure(figsize=(8, 6.8))
    plt.fill_between(thresholds, smooth(low), smooth(high), alpha=0.12, label=f"{label} 95% CI")
    plt.plot(thresholds, smooth(mid), linewidth=2.4, label=label)
    plt.plot(thresholds, smooth(all_strategy), linewidth=2.0, label="Treat all")
    plt.axhline(0, linestyle="--", linewidth=1.5, label="Treat none")
    plt.xlim(0.10, 0.50)
    plt.xlabel("Threshold probability")
    plt.ylabel("Net benefit")
    plt.title("External validation decision curve analysis")
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight")
    plt.close()
    pd.DataFrame({
        "threshold": thresholds,
        "model_net_benefit": mid,
        "model_net_benefit_low": low,
        "model_net_benefit_high": high,
        "treat_all_net_benefit": all_strategy,
        "treat_none_net_benefit": none_strategy,
    }).to_csv(table, index=False)


def make_quartiles(y: np.ndarray, p: np.ndarray, output: Path, table: Path, label: str) -> pd.DataFrame:
    frame = pd.DataFrame({"y": y.astype(int), "pred": p.astype(float)}).dropna()
    frame["quartile"] = pd.qcut(frame["pred"], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
    grouped = frame.groupby("quartile", observed=False).agg(n=("y", "size"), events=("y", "sum"), event_rate=("y", "mean"), mean_pred=("pred", "mean")).reset_index()
    grouped = grouped[grouped["n"] > 0].copy()
    grouped["ci_low"], grouped["ci_high"] = wilson_ci(grouped["events"].to_numpy(), grouped["n"].to_numpy())
    grouped["model"] = label
    grouped.to_csv(table, index=False)
    x = np.arange(len(grouped))
    plt.figure(figsize=(6.4, 5.2))
    plt.errorbar(x, grouped["event_rate"], yerr=[grouped["event_rate"] - grouped["ci_low"], grouped["ci_high"] - grouped["event_rate"]], fmt="o", capsize=4, linewidth=1.5, markersize=8)
    plt.plot(x, grouped["event_rate"], linewidth=1.7)
    plt.xticks(x, grouped["quartile"].astype(str))
    plt.ylabel("Observed event rate")
    plt.xlabel("Predicted-risk quartile")
    plt.title("Observed event rate across predicted-risk quartiles")
    plt.ylim(0, max(0.40, float(grouped["ci_high"].max()) + 0.05))
    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight")
    plt.close()
    return grouped


def make_composite(paths: list[Path], output: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in paths]
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    canvas = Image.new("RGB", (2 * width, 2 * height), "white")
    labels = ["(A)", "(B)", "(C)", "(D)"]
    draw = ImageDraw.Draw(canvas)
    for index, image in enumerate(images):
        x = (index % 2) * width
        y = (index // 2) * height
        canvas.paste(image.resize((width, height)), (x, y))
        draw.text((x + 10, y + 10), labels[index], fill="black")
    canvas.save(output)


def write_lab_availability(data: pd.DataFrame, output: Path, lab_start_min: int, lab_end_min: int) -> None:
    row = {
        "lab_window": f"{lab_start_min}-{lab_end_min} min",
        "lab_strategy": "first",
        "lab_selection_rule": "earliest available value within window",
        "n": int(len(data)),
        "events": int(data["cv_event"].sum()),
        "event_rate": float(data["cv_event"].mean()),
        "ph_available_n": int((data["ph_missing"] == 0).sum()),
        "ph_available_pct": float((data["ph_missing"] == 0).mean() * 100),
        "urea_available_n": int((data["urea_missing"] == 0).sum()),
        "urea_available_pct": float((data["urea_missing"] == 0).mean() * 100),
        "lactate_available_n": int((data["lactate_missing"] == 0).sum()),
        "lactate_available_pct": float((data["lactate_missing"] == 0).mean() * 100),
        "all_three_labs_available_n": int(((data["ph_missing"] == 0) & (data["urea_missing"] == 0) & (data["lactate_missing"] == 0)).sum()),
        "all_three_labs_available_pct": float(((data["ph_missing"] == 0) & (data["urea_missing"] == 0) & (data["lactate_missing"] == 0)).mean() * 100),
    }
    pd.DataFrame([row]).to_csv(output, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eicu-dir", default=r"C:\Users\User\Desktop\Papaioannou\Idea 1\FINAL MODEL\3. external validation_Final_eICU Database")
    parser.add_argument("--coefficients", default=r"C:\Users\User\Desktop\Papaioannou\Idea 1\FINAL MODEL\1. AECOPD_CV_FINAL_MODEL_Internal Validation\model_coefficients_for_figures.csv")
    parser.add_argument("--medians", default=None)
    parser.add_argument("--output", default=rf"C:\Users\User\Desktop\Papaioannou\Idea 1\eicu_external_validation_{WINDOW_LABEL}_frozen")
    parser.add_argument("--lab-start-min", type=int, default=0)
    parser.add_argument("--lab-end-min", type=int, default=LAB_END_MIN)
    args = parser.parse_args()

    style()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    coefficients = load_coefficients(Path(args.coefficients) if args.coefficients else None)
    medians = load_medians(Path(args.medians) if args.medians else None)
    data = build_dataset(Path(args.eicu_dir), args.lab_start_min, args.lab_end_min, medians)
    if len(data) == 0:
        raise ValueError("No eligible patients found.")
    y = data["cv_event"].astype(int).to_numpy()
    if np.unique(y).size < 2:
        raise ValueError("The outcome contains only one class.")
    p = predict_frozen(data, coefficients)
    data["predicted_probability"] = p
    data.to_csv(output / "patient_level_predictions.csv", index=False)
    write_lab_availability(data, output / f"table_lab_availability_{WINDOW_LABEL}.csv", args.lab_start_min, args.lab_end_min)
    roc = make_roc(y, p, output / "figure_roc_external_validation.png", output / "table_roc_external_validation.csv", "AECOPD-CV model")
    calibration = make_calibration(y, p, output / "figure_calibration_external_validation.png", output / "table_calibration_external_validation.csv", "AECOPD-CV model")
    make_dca(y, p, output / "figure_decision_curve_external_validation.png", output / "table_decision_curve_external_validation.csv", "AECOPD-CV model")
    quartiles = make_quartiles(y, p, output / "figure_quartiles_external_validation.png", output / "table_quartiles_external_validation.csv", "AECOPD-CV model")
    make_composite([
        output / "figure_roc_external_validation.png",
        output / "figure_decision_curve_external_validation.png",
        output / "figure_calibration_external_validation.png",
        output / "figure_quartiles_external_validation.png",
    ], output / "External_Validation_Composite.png")
    summary = {
        "n": int(len(data)),
        "events": int(y.sum()),
        "event_rate": float(y.mean()),
        "lab_window_minutes": f"{args.lab_start_min}-{args.lab_end_min}",
        "lab_strategy": "first",
        "lab_selection_rule": "earliest available value within window",
        "auc": float(roc["auc"]),
        "auc_ci_low": float(roc["auc_ci_low"]),
        "auc_ci_high": float(roc["auc_ci_high"]),
        "calibration_intercept": float(calibration["calibration_intercept"]),
        "calibration_slope": float(calibration["calibration_slope"]),
        "brier_score": float(brier_score_loss(y, p)),
        "ph_available_n": int((data["ph_missing"] == 0).sum()),
        "urea_available_n": int((data["urea_missing"] == 0).sum()),
        "lactate_available_n": int((data["lactate_missing"] == 0).sum()),
        "all_three_labs_available_n": int(((data["ph_missing"] == 0) & (data["urea_missing"] == 0) & (data["lactate_missing"] == 0)).sum()),
        "quartile_event_rates": {str(row["quartile"]): float(row["event_rate"]) for _, row in quartiles.iterrows()},
        "coefficients_used": coefficients,
        "medians_used": medians,
    }
    with open(output / f"summary_eicu_{WINDOW_LABEL}.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(f"Rows analysed: {summary['n']}")
    print(f"Events: {summary['events']}")
    print(f"AUC: {summary['auc']:.3f} ({summary['auc_ci_low']:.3f}-{summary['auc_ci_high']:.3f})")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
