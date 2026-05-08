#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


CORE = ["age", "history_hf", "history_af", "ph", "urea", "lactate"]


def data_file(directory: Path, name: str) -> Path:
    candidates = []
    for suffix in [".csv.gz", ".csv", ".CSV.GZ", ".CSV"]:
        candidates.append(directory / f"{name}{suffix}")
        candidates.extend(directory.rglob(f"{name}{suffix}"))
    seen = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing eICU file: {name}")


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def normalise_columns(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data.columns = [str(column).strip() for column in data.columns]
    return data


def clean_text(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().str.strip()


def first_existing_column(data: pd.DataFrame, names: list[str]) -> str | None:
    lookup = {column.lower(): column for column in data.columns}
    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]
    return None


def prepare_age(patient: pd.DataFrame) -> pd.Series:
    age = patient["age"].astype(str).str.strip()
    age = age.replace({"> 89": "90", ">89": "90", "90+": "90"})
    return pd.to_numeric(age, errors="coerce")


def extract_aecopd_ids(diagnosis: pd.DataFrame) -> set[int]:
    text = clean_text(diagnosis["diagnosisstring"])
    copd = text.str.contains(r"\bcopd\b|chronic obstructive pulmonary", regex=True, na=False)
    acute = text.str.contains(r"exacerb|acute|respiratory failure|bronchitis|bronchospasm", regex=True, na=False)
    ids = set(diagnosis.loc[copd & acute, "patientunitstayid"].dropna().astype(int))
    if len(ids) == 0:
        ids = set(diagnosis.loc[copd, "patientunitstayid"].dropna().astype(int))
    return ids


def extract_history(past_history: pd.DataFrame, ids: set[int]) -> pd.DataFrame:
    output = pd.DataFrame({"patientunitstayid": sorted(ids)})
    text_columns = [column for column in ["pasthistoryvalue", "pasthistorypath", "pasthistorynotetype"] if column in past_history.columns]
    if text_columns:
        text = clean_text(past_history[text_columns[0]])
        for column in text_columns[1:]:
            text = text + " " + clean_text(past_history[column])
    else:
        text = past_history.astype(str).agg(" ".join, axis=1).str.lower()
    frame = past_history.copy()
    frame["text"] = text
    frame = frame[frame["patientunitstayid"].isin(ids)]
    hf = frame["text"].str.contains(r"heart failure|congestive heart failure|cardiac failure|\bchf\b", regex=True, na=False)
    af = frame["text"].str.contains(r"atrial fibrillation|atrial flutter|\bafib\b|\baf\b", regex=True, na=False)
    output["history_hf"] = output["patientunitstayid"].isin(frame.loc[hf, "patientunitstayid"].dropna().astype(int)).astype(int)
    output["history_af"] = output["patientunitstayid"].isin(frame.loc[af, "patientunitstayid"].dropna().astype(int)).astype(int)
    return output


def extract_events(diagnosis: pd.DataFrame, ids: set[int]) -> pd.DataFrame:
    frame = diagnosis[diagnosis["patientunitstayid"].isin(ids)].copy()
    text = clean_text(frame["diagnosisstring"])
    chronic = text.str.contains(r"history of|past history|hx of|chronic|known|baseline|prior|previous|old ", regex=True, na=False)
    mi = text.str.contains(r"myocardial infarction|nstemi|stemi|acute mi|non.?st elevation|st elevation", regex=True, na=False)
    mi = mi & (~text.str.contains(r"old myocardial infarction|old mi|history of|hx of|prior|previous", regex=True, na=False))
    pe = text.str.contains(r"pulmonary embolism|\bpe\b", regex=True, na=False) & (~chronic)
    edema = text.str.contains(r"pulmonary edema|pulmonary oedema|acute pulmonary edema|acute pulmonary oedema", regex=True, na=False) & (~chronic)
    arrhythmia = text.str.contains(r"atrial fibrillation|atrial flutter|arrhythmia|tachyarrhythmia|supraventricular tachycardia|\bsvt\b|rapid ventricular response|\brvr\b", regex=True, na=False) & (~chronic)
    hf_decompensation = text.str.contains(r"heart failure|cardiac failure|congestive heart failure|\bchf\b|decompensated heart failure|fluid overload", regex=True, na=False) & (~chronic)
    output = pd.DataFrame({"patientunitstayid": sorted(ids)})
    output["mi_event"] = output["patientunitstayid"].isin(frame.loc[mi, "patientunitstayid"].dropna().astype(int)).astype(int)
    output["pe_event"] = output["patientunitstayid"].isin(frame.loc[pe, "patientunitstayid"].dropna().astype(int)).astype(int)
    output["pulmonary_edema_event"] = output["patientunitstayid"].isin(frame.loc[edema, "patientunitstayid"].dropna().astype(int)).astype(int)
    output["acute_arrhythmia_event"] = output["patientunitstayid"].isin(frame.loc[arrhythmia, "patientunitstayid"].dropna().astype(int)).astype(int)
    output["hf_decompensation_event"] = output["patientunitstayid"].isin(frame.loc[hf_decompensation, "patientunitstayid"].dropna().astype(int)).astype(int)
    output["cv_event"] = (
        (output["mi_event"] == 1)
        | (output["pe_event"] == 1)
        | (output["pulmonary_edema_event"] == 1)
        | (output["acute_arrhythmia_event"] == 1)
    ).astype(int)
    diagnosis_strings = frame.groupby("patientunitstayid")["diagnosisstring"].apply(lambda values: " | ".join(sorted(set(map(str, values.dropna()))))[:3000]).reset_index(name="diagnosis_strings")
    return output.merge(diagnosis_strings, on="patientunitstayid", how="left")


def select_labs(lab: pd.DataFrame, ids: set[int], start_min: int, end_min: int) -> pd.DataFrame:
    lab = lab[lab["patientunitstayid"].isin(ids)].copy()
    offset_column = first_existing_column(lab, ["labresultoffset", "labresultrevisedoffset"])
    if offset_column is None:
        raise ValueError("The lab table must contain labresultoffset or labresultrevisedoffset.")
    lab["lab_name"] = clean_text(lab["labname"])
    lab["lab_value"] = pd.to_numeric(lab["labresult"], errors="coerce")
    lab["lab_offset"] = pd.to_numeric(lab[offset_column], errors="coerce")
    lab = lab.dropna(subset=["lab_value", "lab_offset"])
    lab = lab[(lab["lab_offset"] >= start_min) & (lab["lab_offset"] <= end_min)]
    ph_mask = lab["lab_name"].str.fullmatch(r"ph", na=False) | lab["lab_name"].str.contains(r"\bph\b", regex=True, na=False)
    lactate_mask = lab["lab_name"].str.contains(r"lactate|lactic acid", regex=True, na=False)
    bun_mask = lab["lab_name"].str.contains(r"\bbun\b|blood urea nitrogen", regex=True, na=False)
    urea_direct_mask = lab["lab_name"].str.contains(r"\burea\b", regex=True, na=False) & (~bun_mask)

    def choose(mask: pd.Series, name: str, mode: str) -> pd.DataFrame:
        subset = lab.loc[mask, ["patientunitstayid", "lab_offset", "lab_value", "labname"]].copy()
        if subset.empty:
            return pd.DataFrame(columns=["patientunitstayid", name, f"{name}_offset_min", f"{name}_labname"])
        if mode == "minimum":
            index = subset.groupby("patientunitstayid")["lab_value"].idxmin()
        elif mode == "maximum":
            index = subset.groupby("patientunitstayid")["lab_value"].idxmax()
        else:
            index = subset.sort_values(["patientunitstayid", "lab_offset"]).groupby("patientunitstayid").head(1).index
        output = subset.loc[index].rename(columns={"lab_value": name, "lab_offset": f"{name}_offset_min", "labname": f"{name}_labname"})
        return output[["patientunitstayid", name, f"{name}_offset_min", f"{name}_labname"]]

    output = pd.DataFrame({"patientunitstayid": sorted(ids)})
    output = output.merge(choose(ph_mask, "ph", "minimum"), on="patientunitstayid", how="left")
    output = output.merge(choose(lactate_mask, "lactate", "maximum"), on="patientunitstayid", how="left")
    output = output.merge(choose(bun_mask, "bun", "maximum"), on="patientunitstayid", how="left")
    output = output.merge(choose(urea_direct_mask, "urea_direct", "maximum"), on="patientunitstayid", how="left")
    output["urea_from_bun"] = output["bun"] * 2.14
    output["urea"] = output["urea_from_bun"].where(output["urea_from_bun"].notna(), output["urea_direct"])
    output["urea_source"] = np.where(output["urea_from_bun"].notna(), "bun_converted", np.where(output["urea_direct"].notna(), "direct_urea", "missing"))
    return output


def row_text(data: pd.DataFrame) -> pd.Series:
    frame = data.copy().fillna("")
    for column in frame.columns:
        frame[column] = frame[column].astype(str)
    return frame.apply(lambda row: " ".join(value for value in row.values if value != ""), axis=1)


def extract_respiratory_support(treatment: pd.DataFrame | None, respiratory: pd.DataFrame | None, ids: set[int]) -> pd.DataFrame:
    output = pd.DataFrame({"patientunitstayid": sorted(ids)})
    support_ids = set()
    pattern = r"non-invasive|noninvasive|nippv|nppv|\bniv\b|cpap|bipap|bi-pap|mechanical ventilation|ventilator|intubat|endotracheal|tracheostomy|tidal volume|assist control|simv|prvc"
    if treatment is not None and "patientunitstayid" in treatment.columns:
        text = row_text(treatment).str.lower()
        support_ids |= set(treatment.loc[text.str.contains(pattern, regex=True, na=False), "patientunitstayid"].dropna().astype(int))
    if respiratory is not None and "patientunitstayid" in respiratory.columns:
        text = row_text(respiratory).str.lower()
        support_ids |= set(respiratory.loc[text.str.contains(pattern, regex=True, na=False), "patientunitstayid"].dropna().astype(int))
    output["resp_support"] = output["patientunitstayid"].isin(support_ids).astype(int)
    return output


def first_stay_per_patient(data: pd.DataFrame) -> pd.DataFrame:
    if "uniquepid" not in data.columns:
        return data.copy()
    sort_column = None
    for candidate in ["hospitaladmitoffset", "unitadmitoffset", "unitadmittime24", "patientunitstayid"]:
        if candidate in data.columns:
            sort_column = candidate
            break
    output = data.copy()
    output["_sort_key"] = pd.to_numeric(output[sort_column], errors="coerce") if sort_column else pd.to_numeric(output["patientunitstayid"], errors="coerce")
    output = output.sort_values(["uniquepid", "_sort_key", "patientunitstayid"])
    output = output.groupby("uniquepid", as_index=False).first()
    return output.drop(columns=["_sort_key"], errors="ignore")


def load_coefficients(path: Path) -> dict[str, float]:
    coefficients = pd.read_csv(path)
    coefficients.columns = [str(column).strip() for column in coefficients.columns]
    if not {"variable", "beta"}.issubset(coefficients.columns):
        raise ValueError("Coefficient file must contain variable and beta columns.")
    coefficients["variable"] = coefficients["variable"].astype(str).str.strip()
    coefficients["beta"] = pd.to_numeric(coefficients["beta"], errors="coerce")
    values = dict(zip(coefficients["variable"], coefficients["beta"]))
    required = ["const"] + CORE
    missing = [name for name in required if name not in values or pd.isna(values[name])]
    if missing:
        raise ValueError(f"Coefficient file missing values for: {missing}")
    return {name: float(values[name]) for name in required}


def load_medians(path: Path) -> dict[str, float]:
    medians = pd.read_csv(path)
    medians.columns = [str(column).strip() for column in medians.columns]
    if not {"variable", "median"}.issubset(medians.columns):
        raise ValueError("Median file must contain variable and median columns.")
    medians["variable"] = medians["variable"].astype(str).str.strip()
    medians["median"] = pd.to_numeric(medians["median"], errors="coerce")
    values = dict(zip(medians["variable"], medians["median"]))
    missing = [name for name in CORE if name not in values or pd.isna(values[name])]
    if missing:
        raise ValueError(f"Median file missing values for: {missing}")
    return {name: float(values[name]) for name in CORE}


def predicted_probability(data: pd.DataFrame, coefficients: dict[str, float]) -> np.ndarray:
    linear_predictor = np.full(len(data), coefficients["const"], dtype=float)
    for variable in CORE:
        linear_predictor += coefficients[variable] * pd.to_numeric(data[variable], errors="coerce").to_numpy(dtype=float)
    linear_predictor = np.clip(linear_predictor, -50, 50)
    return 1.0 / (1.0 + np.exp(-linear_predictor))


def build_dataset(eicu_dir: Path, start_min: int, end_min: int, coefficients: Path | None, medians: Path | None) -> pd.DataFrame:
    patient = normalise_columns(read_table(data_file(eicu_dir, "patient")))
    diagnosis = normalise_columns(read_table(data_file(eicu_dir, "diagnosis")))
    lab = normalise_columns(read_table(data_file(eicu_dir, "lab")))
    past_history = normalise_columns(read_table(data_file(eicu_dir, "pastHistory")))
    treatment_path = None
    respiratory_path = None
    try:
        treatment_path = data_file(eicu_dir, "treatment")
    except FileNotFoundError:
        pass
    try:
        respiratory_path = data_file(eicu_dir, "respiratoryCharting")
    except FileNotFoundError:
        pass
    treatment = normalise_columns(read_table(treatment_path)) if treatment_path else None
    respiratory = normalise_columns(read_table(respiratory_path)) if respiratory_path else None

    ids = extract_aecopd_ids(diagnosis)
    events = extract_events(diagnosis, ids)
    history = extract_history(past_history, ids)
    labs = select_labs(lab, ids, start_min, end_min)
    support = extract_respiratory_support(treatment, respiratory, ids)

    patient = patient[patient["patientunitstayid"].isin(ids)].copy()
    patient["age"] = prepare_age(patient)
    keep = ["patientunitstayid", "age"]
    for column in ["uniquepid", "gender", "hospitalid", "hospitaladmitoffset", "unitadmitoffset", "unitadmittime24"]:
        if column in patient.columns:
            keep.append(column)

    data = patient[keep].merge(history, on="patientunitstayid", how="left")
    data = data.merge(labs, on="patientunitstayid", how="left")
    data = data.merge(support, on="patientunitstayid", how="left")
    data = data.merge(events, on="patientunitstayid", how="left")

    for column in ["history_hf", "history_af", "resp_support", "cv_event", "mi_event", "pe_event", "pulmonary_edema_event", "acute_arrhythmia_event", "hf_decompensation_event"]:
        if column in data.columns:
            data[column] = data[column].fillna(0).astype(int)

    for variable in CORE:
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
    data = first_stay_per_patient(data)

    if medians is not None:
        median_values = load_medians(medians)
        for variable in CORE:
            data[variable] = pd.to_numeric(data[variable], errors="coerce").fillna(median_values[variable])

    data["ph"] = pd.to_numeric(data["ph"], errors="coerce").clip(6.8, 7.8)
    data["lactate"] = pd.to_numeric(data["lactate"], errors="coerce").clip(0, 20)
    data["urea"] = pd.to_numeric(data["urea"], errors="coerce").clip(0, 300)

    if coefficients is not None:
        if medians is None:
            raise ValueError("Medians are required when generating frozen model predictions.")
        data["predicted_probability_frozen"] = predicted_probability(data, load_coefficients(coefficients))

    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eicu-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--coefficients", default=None)
    parser.add_argument("--medians", default=None)
    parser.add_argument("--lab-start-min", type=int, default=0)
    parser.add_argument("--lab-end-min", type=int, default=720)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    data = build_dataset(
        eicu_dir=Path(args.eicu_dir),
        start_min=args.lab_start_min,
        end_min=args.lab_end_min,
        coefficients=Path(args.coefficients) if args.coefficients else None,
        medians=Path(args.medians) if args.medians else None,
    )
    data.to_csv(output, index=False)

    print(f"Rows written: {len(data)}")
    print(f"Unique ICU stays: {data['patientunitstayid'].nunique()}")
    print(f"CV events: {int(data['cv_event'].sum())}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
