#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


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


def build_cohort(hosp_dir: Path) -> pd.DataFrame:
    diagnoses = read_table(
        data_file(hosp_dir, "diagnoses_icd"),
        usecols=["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"],
        low_memory=False,
    )

    diagnoses["icd_code_norm"] = normalise_icd(diagnoses["icd_code"])

    eligible = diagnoses[
        (diagnoses["icd_version"] == 10)
        & (diagnoses["seq_num"] == 1)
        & (diagnoses["icd_code_norm"] == "J441")
    ][["subject_id", "hadm_id"]].drop_duplicates()

    admissions = read_table(
        data_file(hosp_dir, "admissions"),
        usecols=["subject_id", "hadm_id", "admittime", "dischtime"],
        low_memory=False,
    )

    admissions["admittime"] = pd.to_datetime(admissions["admittime"], errors="coerce")
    admissions["dischtime"] = pd.to_datetime(admissions["dischtime"], errors="coerce")

    cohort = eligible.merge(admissions, on=["subject_id", "hadm_id"], how="inner")
    cohort = cohort.dropna(subset=["subject_id", "hadm_id", "admittime"])
    cohort = cohort[["subject_id", "hadm_id", "admittime", "dischtime"]].drop_duplicates()
    cohort = cohort.sort_values(["subject_id", "admittime", "hadm_id"], kind="stable")
    cohort = cohort.drop_duplicates(subset=["subject_id"], keep="first").reset_index(drop=True)

    if cohort["subject_id"].duplicated().any():
        raise RuntimeError("Duplicate patients remain after first-admission selection.")

    return cohort


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic-hosp", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    hosp_dir = Path(args.mimic_hosp)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    cohort = build_cohort(hosp_dir)
    cohort.to_csv(output, index=False)

    print(f"Rows written: {len(cohort)}")
    print(f"Unique patients: {cohort['subject_id'].nunique()}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
