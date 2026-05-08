#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


CHUNK_SIZE = 1_000_000
EARLY_HOURS = 6
PH_MIN, PH_MAX = 6.8, 7.8
BUN_MIN, BUN_MAX = 1.0, 300.0
LACTATE_MIN, LACTATE_MAX = 0.1, 30.0


def data_file(directory: Path, name: str) -> Path:
    plain = directory / f"{name}.csv"
    zipped = directory / f"{name}.csv.gz"
    if plain.exists():
        return plain
    if zipped.exists():
        return zipped
    raise FileNotFoundError(f"Missing file for {name}")


def read_table(path: Path, **kwargs) -> pd.DataFrame:
    if path.suffix == ".gz":
        return pd.read_csv(path, compression="gzip", **kwargs)
    return pd.read_csv(path, **kwargs)


def normalise_icd(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.upper()
        .str.replace(".", "", regex=False)
    )


def flag_hf(code: str, version) -> bool:
    if pd.isna(version):
        return False
    version = int(version)
    if version == 10:
        return code.startswith(("I50", "I110", "I130", "I132"))
    if version == 9:
        return code.startswith(("428", "40201", "40211", "40291", "40401", "40403", "40411", "40413", "40491", "40493"))
    return False


def flag_af(code: str, version) -> bool:
    if pd.isna(version):
        return False
    version = int(version)
    if version == 10:
        return code.startswith("I48")
    if version == 9:
        return code.startswith(("42731", "42732"))
    return False


def flag_mi(code: str, version) -> bool:
    if pd.isna(version):
        return False
    version = int(version)
    if version == 10:
        return code != "I21A1" and code.startswith(("I21", "I22"))
    if version == 9:
        return code.startswith("410")
    return False


def flag_type2_mi(code: str, version) -> bool:
    return not pd.isna(version) and int(version) == 10 and code == "I21A1"


def flag_pe(code: str, version) -> bool:
    if pd.isna(version):
        return False
    version = int(version)
    if version == 10:
        return code.startswith("I26")
    if version == 9:
        return code.startswith("4151")
    return False


def flag_pulmonary_edema(code: str, version) -> bool:
    if pd.isna(version):
        return False
    version = int(version)
    if version == 10:
        return code.startswith("J81")
    if version == 9:
        return code.startswith("5184")
    return False


def flag_arrhythmia(code: str, version) -> bool:
    if pd.isna(version):
        return False
    version = int(version)
    if version == 10:
        return code.startswith(("I47", "I48", "I490", "I491", "I498", "I499"))
    if version == 9:
        return code.startswith(("4270", "4271", "42731", "42732", "42741", "42742", "4279"))
    return False


def flag_cardiac_arrest(code: str, version) -> bool:
    if pd.isna(version):
        return False
    version = int(version)
    if version == 10:
        return code.startswith("I46")
    if version == 9:
        return code.startswith("4275")
    return False


def load_cohort(path: Path) -> pd.DataFrame:
    cohort = pd.read_csv(path)
    required = ["subject_id", "hadm_id", "admittime", "dischtime"]
    missing = [column for column in required if column not in cohort.columns]
    if missing:
        raise ValueError(f"Missing cohort columns: {missing}")
    cohort = cohort[required].copy()
    cohort["subject_id"] = pd.to_numeric(cohort["subject_id"], errors="coerce").astype("Int64")
    cohort["hadm_id"] = pd.to_numeric(cohort["hadm_id"], errors="coerce").astype("Int64")
    cohort["admittime"] = pd.to_datetime(cohort["admittime"], errors="coerce")
    cohort["dischtime"] = pd.to_datetime(cohort["dischtime"], errors="coerce")
    cohort = cohort.dropna(subset=["subject_id", "hadm_id", "admittime"])
    cohort = cohort.drop_duplicates(subset=["subject_id"], keep="first").reset_index(drop=True)
    return cohort


def add_age(cohort: pd.DataFrame, hosp_dir: Path) -> pd.DataFrame:
    patients = read_table(
        data_file(hosp_dir, "patients"),
        usecols=["subject_id", "anchor_age", "anchor_year"],
        low_memory=False,
    )
    patients["subject_id"] = pd.to_numeric(patients["subject_id"], errors="coerce").astype("Int64")
    patients["anchor_age"] = pd.to_numeric(patients["anchor_age"], errors="coerce")
    patients["anchor_year"] = pd.to_numeric(patients["anchor_year"], errors="coerce")
    cohort = cohort.merge(patients, on="subject_id", how="left")
    cohort["age"] = cohort["anchor_age"] + (cohort["admittime"].dt.year - cohort["anchor_year"])
    cohort["age"] = cohort["age"].round(1)
    return cohort.drop(columns=["anchor_age", "anchor_year"])


def admissions_table(hosp_dir: Path) -> pd.DataFrame:
    admissions = read_table(
        data_file(hosp_dir, "admissions"),
        usecols=["subject_id", "hadm_id", "admittime"],
        low_memory=False,
    )
    admissions["subject_id"] = pd.to_numeric(admissions["subject_id"], errors="coerce").astype("Int64")
    admissions["hadm_id"] = pd.to_numeric(admissions["hadm_id"], errors="coerce").astype("Int64")
    admissions["admittime"] = pd.to_datetime(admissions["admittime"], errors="coerce")
    return admissions.dropna(subset=["subject_id", "hadm_id", "admittime"])


def diagnosis_table(hosp_dir: Path, cohort: pd.DataFrame, admissions: pd.DataFrame) -> pd.DataFrame:
    subjects = set(cohort["subject_id"].dropna().astype(int))
    reader = read_table(
        data_file(hosp_dir, "diagnoses_icd"),
        usecols=["subject_id", "hadm_id", "icd_code", "icd_version"],
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )
    parts = []
    for chunk in reader:
        chunk["subject_id"] = pd.to_numeric(chunk["subject_id"], errors="coerce").astype("Int64")
        chunk["hadm_id"] = pd.to_numeric(chunk["hadm_id"], errors="coerce").astype("Int64")
        chunk = chunk[chunk["subject_id"].isin(subjects)].copy()
        if len(chunk) == 0:
            continue
        chunk["code_norm"] = normalise_icd(chunk["icd_code"])
        parts.append(chunk[["subject_id", "hadm_id", "icd_version", "code_norm"]])
    if not parts:
        raise RuntimeError("No diagnosis records found for selected cohort.")
    diagnoses = pd.concat(parts, ignore_index=True)
    diagnoses = diagnoses.merge(admissions[["subject_id", "hadm_id", "admittime"]], on=["subject_id", "hadm_id"], how="left")
    return diagnoses.dropna(subset=["admittime"])


def add_diagnosis_flags(cohort: pd.DataFrame, diagnoses: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in cohort.itertuples(index=False):
        subject_id = int(row.subject_id)
        hadm_id = int(row.hadm_id)
        admission_time = row.admittime
        subject_diagnoses = diagnoses[diagnoses["subject_id"] == subject_id]
        index_diagnoses = subject_diagnoses[subject_diagnoses["hadm_id"] == hadm_id]
        prior_diagnoses = subject_diagnoses[subject_diagnoses["admittime"] < admission_time]
        index_codes = sorted(index_diagnoses["code_norm"].dropna().astype(str).unique())
        prior_hf = int(any(flag_hf(c, v) for c, v in zip(prior_diagnoses["code_norm"], prior_diagnoses["icd_version"])))
        prior_af = int(any(flag_af(c, v) for c, v in zip(prior_diagnoses["code_norm"], prior_diagnoses["icd_version"])))
        prior_mi = int(any(flag_mi(c, v) for c, v in zip(prior_diagnoses["code_norm"], prior_diagnoses["icd_version"])))
        prior_pe = int(any(flag_pe(c, v) for c, v in zip(prior_diagnoses["code_norm"], prior_diagnoses["icd_version"])))
        prior_arrhythmia = int(any(flag_arrhythmia(c, v) for c, v in zip(prior_diagnoses["code_norm"], prior_diagnoses["icd_version"])))
        rows.append({
            "subject_id": subject_id,
            "hadm_id": hadm_id,
            "index_icd_codes": ";".join(index_codes),
            "history_hf": prior_hf,
            "history_af": prior_af,
            "history_mi_prior": prior_mi,
            "history_pe_prior": prior_pe,
            "history_arrhythmia_prior": prior_arrhythmia,
            "mi_code_index": int(any(flag_mi(c, v) for c, v in zip(index_diagnoses["code_norm"], index_diagnoses["icd_version"]))),
            "type2_mi_code_index": int(any(flag_type2_mi(c, v) for c, v in zip(index_diagnoses["code_norm"], index_diagnoses["icd_version"]))),
            "pe_code_index": int(any(flag_pe(c, v) for c, v in zip(index_diagnoses["code_norm"], index_diagnoses["icd_version"]))),
            "pulmonary_edema_code_index": int(any(flag_pulmonary_edema(c, v) for c, v in zip(index_diagnoses["code_norm"], index_diagnoses["icd_version"]))),
            "acute_arrhythmia_code_index": int(any(flag_arrhythmia(c, v) for c, v in zip(index_diagnoses["code_norm"], index_diagnoses["icd_version"]))),
            "cardiac_arrest_code_index": int(any(flag_cardiac_arrest(c, v) for c, v in zip(index_diagnoses["code_norm"], index_diagnoses["icd_version"]))),
        })
    flags = pd.DataFrame(rows)
    return cohort.merge(flags, on=["subject_id", "hadm_id"], how="left")


def lab_itemids(hosp_dir: Path) -> dict[str, set[int]]:
    labitems = read_table(data_file(hosp_dir, "d_labitems"), low_memory=False)
    names = {column.lower(): column for column in labitems.columns}
    itemid_col = names["itemid"]
    label_col = names["label"]
    labitems["label_norm"] = labitems[label_col].astype(str).str.strip().str.upper()
    labitems["fluid_norm"] = labitems[names["fluid"]].astype(str).str.strip().str.upper() if "fluid" in names else ""
    labitems["category_norm"] = labitems[names["category"]].astype(str).str.strip().str.upper() if "category" in names else ""
    blood = labitems["fluid_norm"].str.contains("BLOOD", na=False) | labitems["category_norm"].str.contains("BLOOD GAS|CHEM", na=False)
    return {
        "ph": set(labitems.loc[labitems["label_norm"].eq("PH") & blood, itemid_col].dropna().astype(int)),
        "bun": set(labitems.loc[labitems["label_norm"].str.contains("UREA NITROGEN", na=False) & blood, itemid_col].dropna().astype(int)),
        "lactate": set(labitems.loc[labitems["label_norm"].str.contains("LACTATE", na=False) & blood, itemid_col].dropna().astype(int)),
        "troponin": set(labitems.loc[labitems["label_norm"].str.contains("TROPONIN", na=False) & blood, itemid_col].dropna().astype(int)),
    }


def plausible(name: str, value: float) -> bool:
    if name == "ph":
        return PH_MIN <= value <= PH_MAX
    if name == "bun":
        return BUN_MIN <= value <= BUN_MAX
    if name == "lactate":
        return LACTATE_MIN <= value <= LACTATE_MAX
    if name == "troponin":
        return value >= 0
    return False


def extract_labs(hosp_dir: Path, cohort: pd.DataFrame, items: dict[str, set[int]]) -> pd.DataFrame:
    stays = cohort.set_index("hadm_id")[["admittime", "dischtime"]].to_dict("index")
    hadm_ids = set(cohort["hadm_id"].dropna().astype(int))
    reverse = {}
    for name, ids in items.items():
        for itemid in ids:
            reverse[int(itemid)] = name
    reader = read_table(
        data_file(hosp_dir, "labevents"),
        usecols=["hadm_id", "itemid", "charttime", "valuenum"],
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )
    selected = {}
    for chunk in reader:
        chunk["hadm_id"] = pd.to_numeric(chunk["hadm_id"], errors="coerce").astype("Int64")
        chunk["itemid"] = pd.to_numeric(chunk["itemid"], errors="coerce")
        chunk["valuenum"] = pd.to_numeric(chunk["valuenum"], errors="coerce")
        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        chunk = chunk[
            chunk["hadm_id"].isin(hadm_ids)
            & chunk["itemid"].isin(reverse)
            & chunk["valuenum"].notna()
            & chunk["charttime"].notna()
        ]
        if len(chunk) == 0:
            continue
        for row in chunk.itertuples(index=False):
            hadm_id = int(row.hadm_id)
            itemid = int(row.itemid)
            name = reverse[itemid]
            value = float(row.valuenum)
            charttime = row.charttime
            stay = stays.get(hadm_id)
            if stay is None:
                continue
            admittime = stay["admittime"]
            dischtime = stay["dischtime"]
            if pd.isna(admittime) or charttime < admittime:
                continue
            if pd.notna(dischtime) and charttime > dischtime:
                continue
            if charttime > admittime + pd.Timedelta(hours=EARLY_HOURS):
                continue
            if not plausible(name, value):
                continue
            key = (hadm_id, name)
            if key not in selected or charttime < selected[key][0]:
                selected[key] = (charttime, value)
    rows = []
    for row in cohort.itertuples(index=False):
        hadm_id = int(row.hadm_id)
        ph = selected.get((hadm_id, "ph"), (pd.NaT, pd.NA))[1]
        bun = selected.get((hadm_id, "bun"), (pd.NaT, pd.NA))[1]
        lactate = selected.get((hadm_id, "lactate"), (pd.NaT, pd.NA))[1]
        troponin = selected.get((hadm_id, "troponin"), (pd.NaT, pd.NA))[1]
        rows.append({
            "hadm_id": hadm_id,
            "ph": ph,
            "bun_raw": bun,
            "urea": (float(bun) * 2.14) if pd.notna(bun) else pd.NA,
            "lactate": lactate,
            "troponin_early_first": troponin,
            "troponin_early_available": int(pd.notna(troponin)),
        })
    return pd.DataFrame(rows)


def add_outcomes(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["mi_event"] = ((data["mi_code_index"] == 1) & (data["type2_mi_code_index"] == 0) & (data["history_mi_prior"] == 0)).astype(int)
    data["pe_event"] = ((data["pe_code_index"] == 1) & (data["history_pe_prior"] == 0)).astype(int)
    data["pulmonary_edema_event"] = (data["pulmonary_edema_code_index"] == 1).astype(int)
    data["acute_arrhythmia_event"] = ((data["acute_arrhythmia_code_index"] == 1) & (data["history_arrhythmia_prior"] == 0) & (data["history_af"] == 0)).astype(int)
    data["cardiac_arrest_event"] = (data["cardiac_arrest_code_index"] == 1).astype(int)
    data["cv_event"] = (
        (data["mi_event"] == 1)
        | (data["pe_event"] == 1)
        | (data["pulmonary_edema_event"] == 1)
        | (data["acute_arrhythmia_event"] == 1)
    ).astype(int)
    data["mi_confidence"] = data.apply(lambda row: "probable" if row["mi_event"] == 1 and row["troponin_early_available"] == 1 else ("uncertain" if row["mi_event"] == 1 else ""), axis=1)
    data["event_confidence"] = data.apply(lambda row: "probable" if row["cv_event"] == 1 and (row["mi_event"] == 0 or row["troponin_early_available"] == 1) else ("uncertain" if row["cv_event"] == 1 else ""), axis=1)
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic-hosp", required=True)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    hosp_dir = Path(args.mimic_hosp)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    cohort = load_cohort(Path(args.cohort))
    cohort = add_age(cohort, hosp_dir)
    admissions = admissions_table(hosp_dir)
    diagnoses = diagnosis_table(hosp_dir, cohort, admissions)
    cohort = add_diagnosis_flags(cohort, diagnoses)
    labs = extract_labs(hosp_dir, cohort, lab_itemids(hosp_dir))
    data = cohort.merge(labs, on="hadm_id", how="left")
    data = add_outcomes(data)

    data.to_csv(output, index=False)

    print(f"Rows written: {len(data)}")
    print(f"Unique patients: {data['subject_id'].nunique()}")
    print(f"CV events: {int(data['cv_event'].sum())}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
