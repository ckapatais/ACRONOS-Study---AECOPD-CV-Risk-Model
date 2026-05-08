#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def read_input(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path, low_memory=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = read_input(Path(args.input))
    rows = [
        {"metric": "total_patients", "value": int(len(data))},
        {"metric": "cv_events", "value": int(pd.to_numeric(data["cv_event"], errors="coerce").fillna(0).sum())},
        {"metric": "myocardial_infarction", "value": int(pd.to_numeric(data["mi_event"], errors="coerce").fillna(0).sum())},
        {"metric": "pulmonary_embolism", "value": int(pd.to_numeric(data["pe_event"], errors="coerce").fillna(0).sum())},
        {"metric": "pulmonary_edema", "value": int(pd.to_numeric(data["pulmonary_edema_event"], errors="coerce").fillna(0).sum())},
        {"metric": "acute_arrhythmia", "value": int(pd.to_numeric(data["acute_arrhythmia_event"], errors="coerce").fillna(0).sum())},
        {"metric": "ph_available_percent", "value": float(pd.to_numeric(data["ph"], errors="coerce").notna().mean() * 100)},
        {"metric": "urea_available_percent", "value": float(pd.to_numeric(data["urea"], errors="coerce").notna().mean() * 100)},
        {"metric": "lactate_available_percent", "value": float(pd.to_numeric(data["lactate"], errors="coerce").notna().mean() * 100)},
        {"metric": "history_hf_available_percent", "value": float(data["history_hf"].notna().mean() * 100)},
        {"metric": "history_af_available_percent", "value": float(data["history_af"].notna().mean() * 100)},
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
