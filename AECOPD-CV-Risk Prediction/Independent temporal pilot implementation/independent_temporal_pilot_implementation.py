from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta, gaussian_kde
from sklearn.metrics import roc_auc_score, roc_curve
from statsmodels.nonparametric.smoothers_lowess import lowess

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


COEFFICIENTS = {
    "intercept": 46.950194,
    "age": 0.00563,
    "history_hf": 0.553568,
    "history_af": 1.243488,
    "ph": -6.746131,
    "urea": 0.006961,
    "lactate": 0.119096,
}


COLUMN_MAP = {
    "age": "ηλικία",
    "history_hf": "ΚΑ",
    "history_af": "AF",
    "ph": "Phεισόδου",
    "urea": "ουρία εισόδου",
    "lactate": "LAC",
    "myocardial_infarction": "ΟΕΜ.1",
    "acute_arrhythmia": "ΑΡΡΥΘΜΙΑ",
    "pulmonary_edema": "ΟΠΟ",
    "pulmonary_embolism": "ΠΕ",
}


def read_input(path: Path, sheet: str | None) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls", ".xlx"}:
        return pd.read_excel(path, sheet_name=sheet or 0)
    return pd.read_csv(path, low_memory=False)


def normalise_columns(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data.columns = [str(column).strip() for column in data.columns]
    return data


def clean_binary(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.fillna(0).astype(int).clip(0, 1)


def prepare_dataset(data: pd.DataFrame) -> pd.DataFrame:
    data = normalise_columns(data)
    missing = [column for column in COLUMN_MAP.values() if column not in data.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    out = pd.DataFrame()
    if "Ν" in data.columns:
        out["patient_index"] = data["Ν"]

    out["age"] = pd.to_numeric(data[COLUMN_MAP["age"]], errors="coerce")
    out["history_hf"] = clean_binary(data[COLUMN_MAP["history_hf"]])
    out["history_af"] = clean_binary(data[COLUMN_MAP["history_af"]])
    out["ph"] = pd.to_numeric(data[COLUMN_MAP["ph"]], errors="coerce")
    out["urea"] = pd.to_numeric(data[COLUMN_MAP["urea"]], errors="coerce")
    out["lactate"] = pd.to_numeric(data[COLUMN_MAP["lactate"]], errors="coerce")

    out["myocardial_infarction"] = clean_binary(data[COLUMN_MAP["myocardial_infarction"]])
    out["acute_arrhythmia"] = clean_binary(data[COLUMN_MAP["acute_arrhythmia"]])
    out["pulmonary_edema"] = clean_binary(data[COLUMN_MAP["pulmonary_edema"]])
    out["pulmonary_embolism"] = clean_binary(data[COLUMN_MAP["pulmonary_embolism"]])

    out["composite_cv_event"] = (
        (out["myocardial_infarction"] == 1)
        | (out["acute_arrhythmia"] == 1)
        | (out["pulmonary_edema"] == 1)
        | (out["pulmonary_embolism"] == 1)
    ).astype(int)

    predictors = ["age", "history_hf", "history_af", "ph", "urea", "lactate"]
    out["complete_predictors"] = out[predictors].notna().all(axis=1).astype(int)
    return out


def add_predictions(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    lp = (
        COEFFICIENTS["intercept"]
        + COEFFICIENTS["age"] * out["age"]
        + COEFFICIENTS["history_hf"] * out["history_hf"]
        + COEFFICIENTS["history_af"] * out["history_af"]
        + COEFFICIENTS["ph"] * out["ph"]
        + COEFFICIENTS["urea"] * out["urea"]
        + COEFFICIENTS["lactate"] * out["lactate"]
    )
    out["linear_predictor"] = lp
    out["predicted_probability"] = 1 / (1 + np.exp(-np.clip(lp, -50, 50)))
    return out


def bootstrap_auc_ci(y: np.ndarray, p: np.ndarray, n_boot: int = 5000, seed: int = 42) -> tuple[float, float]:
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


def exact_binomial_ci(events: int, n: int) -> tuple[float, float]:
    low = beta.ppf(0.025, events, n - events + 1) if events > 0 else 0.0
    high = beta.ppf(0.975, events + 1, n - events) if events < n else 1.0
    return float(low), float(high)


def style() -> None:
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 9.5,
        "axes.linewidth": 0.9,
        "figure.dpi": 160,
        "savefig.dpi": 600,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save_all(fig: plt.Figure, output_dir: Path, name: str) -> None:
    fig.savefig(output_dir / f"{name}.png", bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.tiff", bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_roc(y: np.ndarray, p: np.ndarray, output_dir: Path) -> dict:
    auc = roc_auc_score(y, p)
    lo, hi = bootstrap_auc_ci(y, p)
    fpr, tpr, _ = roc_curve(y, p)

    fig, ax = plt.subplots(figsize=(6.2, 5.3), facecolor="white")
    ax.plot(fpr, tpr, linewidth=2.4, label=f"AECOPD-CV model: AUC {auc:.3f} ({lo:.3f}–{hi:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, color="#bfbfbf")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Independent temporal pilot implementation ROC curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", frameon=False)
    ax.grid(False)
    fig.tight_layout()
    save_all(fig, output_dir, "01_roc_curve")

    return {"auc": float(auc), "auc_ci_low": float(lo), "auc_ci_high": float(hi)}


def plot_scatter_lowess(y: np.ndarray, p: np.ndarray, output_dir: Path) -> None:
    rng = np.random.default_rng(42)
    y_jitter = np.clip(y.astype(float) + rng.normal(0, 0.035, size=len(y)), -0.06, 1.06)
    ordered = pd.DataFrame({"pred": p, "event": y}).sort_values("pred")
    smoothed = lowess(ordered["event"], ordered["pred"], frac=0.75, it=0, return_sorted=True)

    fig, ax = plt.subplots(figsize=(6.8, 5.3), facecolor="white")
    ax.scatter(p[y == 0], y_jitter[y == 0], s=48, alpha=0.70, label="No event")
    ax.scatter(p[y == 1], y_jitter[y == 1], s=58, alpha=0.90, marker="D", label="Event")
    ax.plot(smoothed[:, 0], smoothed[:, 1], linewidth=2.4, label="LOWESS trend")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed outcome")
    ax.set_title("Predicted risk by observed outcome")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["No event", "Event"])
    ax.set_xlim(0, max(0.75, float(np.max(p)) + 0.05))
    ax.set_ylim(-0.12, 1.12)
    ax.legend(loc="upper left", frameon=False)
    ax.grid(False)
    fig.tight_layout()
    save_all(fig, output_dir, "02_scatter_lowess")


def plot_violin_box(y: np.ndarray, p: np.ndarray, output_dir: Path) -> None:
    no_event = p[y == 0]
    event = p[y == 1]

    fig, ax = plt.subplots(figsize=(6.2, 5.3), facecolor="white")
    parts = ax.violinplot([no_event, event], positions=[0, 1], widths=0.72, showmeans=False, showmedians=False, showextrema=False)
    for body in parts["bodies"]:
        body.set_alpha(0.25)

    box = ax.boxplot([no_event, event], positions=[0, 1], widths=0.28, patch_artist=True, showfliers=False)
    for patch in box["boxes"]:
        patch.set_alpha(0.55)

    rng = np.random.default_rng(42)
    ax.scatter(rng.normal(0, 0.045, len(no_event)), no_event, s=44, alpha=0.75, label="No event")
    ax.scatter(rng.normal(1, 0.045, len(event)), event, s=58, alpha=0.90, marker="D", label="Event")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No event", "Event"])
    ax.set_ylabel("Predicted probability")
    ax.set_title("Distribution of predicted risk by observed outcome")
    ax.set_ylim(0, max(0.80, float(np.max(p)) + 0.06))
    ax.legend(loc="upper left", frameon=False)
    ax.grid(False)
    fig.tight_layout()
    save_all(fig, output_dir, "03_violin_box_predicted_risk")


def plot_box_swarm(y: np.ndarray, p: np.ndarray, output_dir: Path) -> None:
    no_event = p[y == 0]
    event = p[y == 1]
    rng = np.random.default_rng(42)

    fig, ax = plt.subplots(figsize=(6.2, 5.3), facecolor="white")
    ax.boxplot([no_event, event], positions=[0, 1], widths=0.35, patch_artist=True, showfliers=False)
    ax.scatter(rng.normal(0, 0.055, len(no_event)), no_event, s=46, alpha=0.75, label="No event")
    ax.scatter(rng.normal(1, 0.055, len(event)), event, s=60, alpha=0.90, marker="D", label="Event")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No event", "Event"])
    ax.set_ylabel("Predicted probability")
    ax.set_title("Predicted risk distribution")
    ax.set_ylim(0, max(0.80, float(np.max(p)) + 0.06))
    ax.legend(loc="upper left", frameon=False)
    ax.grid(False)
    fig.tight_layout()
    save_all(fig, output_dir, "04_box_swarm_predicted_risk")


def plot_density(y: np.ndarray, p: np.ndarray, output_dir: Path) -> None:
    no_event = p[y == 0]
    event = p[y == 1]
    xmax = max(0.80, float(np.max(p)) + 0.06)
    grid = np.linspace(0, xmax, 300)

    fig, ax = plt.subplots(figsize=(6.8, 5.3), facecolor="white")

    if len(no_event) >= 2 and np.std(no_event) > 0:
        kde_no = gaussian_kde(no_event)
        y_no = kde_no(grid)
        ax.fill_between(grid, y_no, alpha=0.25)
        ax.plot(grid, y_no, linewidth=2.0, label="No event")
    else:
        ax.hist(no_event, bins=8, density=True, alpha=0.25, label="No event")

    if len(event) >= 2 and np.std(event) > 0:
        kde_event = gaussian_kde(event)
        y_event = kde_event(grid)
        ax.fill_between(grid, y_event, alpha=0.25)
        ax.plot(grid, y_event, linewidth=2.0, label="Event")
    else:
        ax.hist(event, bins=8, density=True, alpha=0.25, label="Event")

    ax.plot(no_event, np.zeros_like(no_event), "|", markersize=12, alpha=0.70)
    ax.plot(event, np.zeros_like(event), "|", markersize=14, alpha=0.90)

    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Density")
    ax.set_title("Predicted-risk distribution by observed outcome")
    ax.set_xlim(0, xmax)
    ax.legend(frameon=False)
    ax.grid(False)
    fig.tight_layout()
    save_all(fig, output_dir, "05_density_predicted_risk")


def plot_histogram(y: np.ndarray, p: np.ndarray, output_dir: Path) -> None:
    no_event = p[y == 0]
    event = p[y == 1]
    xmax = max(0.80, float(np.max(p)) + 0.06)
    bins = np.linspace(0, xmax, 9)

    fig, ax = plt.subplots(figsize=(6.8, 5.3), facecolor="white")
    ax.hist(no_event, bins=bins, alpha=0.55, label="No event")
    ax.hist(event, bins=bins, alpha=0.70, label="Event")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Number of patients")
    ax.set_title("Predicted-risk histogram by observed outcome")
    ax.set_xlim(0, xmax)
    ax.legend(frameon=False)
    ax.grid(False)
    fig.tight_layout()
    save_all(fig, output_dir, "06_histogram_predicted_risk")


def composite(figures: list[Path], output_dir: Path, name: str) -> None:
    from PIL import Image, ImageDraw, ImageFont

    imgs = [Image.open(p).convert("RGB") for p in figures]
    w = max(i.width for i in imgs)
    h = max(i.height for i in imgs)
    canvas = Image.new("RGB", (2 * w, 2 * h), "white")
    labels = ["A", "B", "C", "D"]
    draw = ImageDraw.Draw(canvas)

    for i, img in enumerate(imgs[:4]):
        x = (i % 2) * w
        y = (i // 2) * h
        img = img.resize((w, h))
        canvas.paste(img, (x, y))
        draw.text((x + 18, y + 14), labels[i], fill="black")

    canvas.save(output_dir / f"{name}.png")
    canvas.save(output_dir / f"{name}.tiff")
    canvas.save(output_dir / f"{name}.pdf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="Final Validation excel.xlsx")
    parser.add_argument("--sheet", default=None)
    parser.add_argument("--output-dir", default="independent_temporal_pilot_implementation_visual_options")
    args = parser.parse_args()

    style()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = add_predictions(prepare_dataset(read_input(Path(args.input), args.sheet)))
    valid = data["complete_predictors"].eq(1) & data["predicted_probability"].notna()
    analysis = data.loc[valid].copy()

    y = analysis["composite_cv_event"].astype(int).to_numpy()
    p = analysis["predicted_probability"].astype(float).to_numpy()

    if len(y) == 0 or np.unique(y).size < 2:
        raise ValueError("The analysis dataset must contain complete predictors and both outcome classes.")

    auc_info = plot_roc(y, p, output_dir)
    plot_scatter_lowess(y, p, output_dir)
    plot_violin_box(y, p, output_dir)
    plot_box_swarm(y, p, output_dir)
    plot_density(y, p, output_dir)
    plot_histogram(y, p, output_dir)

    composite(
        [
            output_dir / "01_roc_curve.png",
            output_dir / "03_violin_box_predicted_risk.png",
            output_dir / "05_density_predicted_risk.png",
            output_dir / "06_histogram_predicted_risk.png",
        ],
        output_dir,
        "independent_temporal_pilot_implementation_visual_options_composite",
    )

    n = len(y)
    events = int(y.sum())
    event_low, event_high = exact_binomial_ci(events, n)

    summary = {
        "n": int(n),
        "events": int(events),
        "event_rate": float(events / n),
        "event_rate_ci_low": event_low,
        "event_rate_ci_high": event_high,
        "mean_predicted_probability": float(np.mean(p)),
        "median_predicted_probability": float(np.median(p)),
        **auc_info,
    }

    analysis.to_csv(output_dir / "independent_temporal_pilot_implementation_patient_predictions.csv", index=False)
    pd.DataFrame([summary]).to_csv(output_dir / "independent_temporal_pilot_implementation_summary.csv", index=False)

    with open(output_dir / "independent_temporal_pilot_implementation_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "input_file": str(Path(args.input)),
                "outcome_definition": "Composite cardiovascular event = myocardial infarction, pulmonary embolism, pulmonary edema, or acute arrhythmia",
                "columns": COLUMN_MAP,
                "summary": summary,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Rows analysed: {n}")
    print(f"Events: {events}")
    print(f"AUC: {summary['auc']:.3f} ({summary['auc_ci_low']:.3f}–{summary['auc_ci_high']:.3f})")
    print(f"Output folder: {output_dir}")


if __name__ == "__main__":
    main()
