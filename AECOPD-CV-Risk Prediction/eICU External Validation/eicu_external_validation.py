#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from PIL import Image, ImageDraw
from sklearn.metrics import roc_auc_score, roc_curve
from statsmodels.nonparametric.smoothers_lowess import lowess


RANDOM_STATE = 42


def read_input(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path, low_memory=False)


def normalise_columns(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data.columns = [str(column).strip() for column in data.columns]
    return data


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


def bootstrap_auc_ci(y: np.ndarray, p: np.ndarray, n_boot: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(RANDOM_STATE)
    values = []
    n = len(y)
    for _ in range(n_boot):
        index = rng.integers(0, n, size=n)
        yb = y[index]
        pb = p[index]
        if len(np.unique(yb)) < 2:
            continue
        values.append(roc_auc_score(yb, pb))
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
    values = np.zeros(len(thresholds), dtype=float)
    n = len(y)
    for i, threshold in enumerate(thresholds):
        predicted_positive = p >= threshold
        true_positive = np.sum((predicted_positive == 1) & (y == 1))
        false_positive = np.sum((predicted_positive == 1) & (y == 0))
        values[i] = (true_positive / n) - (false_positive / n) * (threshold / (1 - threshold))
    return values


def treat_all_net_benefit(y: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    prevalence = np.mean(y.astype(int))
    return prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))


def moving_average(values: np.ndarray, window: int = 5) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return values.copy()
    kernel = np.ones(window) / window
    padded = np.pad(values.astype(float), (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def bootstrap_net_benefit_ci(y: np.ndarray, p: np.ndarray, thresholds: np.ndarray, n_boot: int = 300) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(y)
    matrix = np.zeros((n_boot, len(thresholds)), dtype=float)
    for row in range(n_boot):
        index = rng.integers(0, n, size=n)
        matrix[row, :] = net_benefit(y[index], p[index], thresholds)
    return matrix.mean(axis=0), np.percentile(matrix, 2.5, axis=0), np.percentile(matrix, 97.5, axis=0)


def roc_plot(y: np.ndarray, p: np.ndarray, output: Path, table: Path, label: str) -> dict:
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

    result = {"auc": auc, "auc_ci_low": low, "auc_ci_high": high}
    pd.DataFrame([result]).to_csv(table, index=False)
    return result


def calibration_plot(y: np.ndarray, p: np.ndarray, output: Path, table: Path, label: str) -> dict:
    p = np.clip(p.astype(float), 1e-6, 1 - 1e-6)
    frame = pd.DataFrame({"y": y.astype(int), "pred": p}).sort_values("pred").reset_index(drop=True)
    x_max = min(0.90, float(np.quantile(frame["pred"], 0.98)))
    plot_data = frame[frame["pred"] <= x_max].copy()
    if len(plot_data) < 30:
        plot_data = frame.copy()
        x_max = min(0.90, float(frame["pred"].max()))

    curve = lowess(plot_data["y"], plot_data["pred"], frac=0.72, it=0, return_sorted=True)
    grid = np.linspace(float(plot_data["pred"].min()), float(plot_data["pred"].max()), 160)
    rng = np.random.default_rng(RANDOM_STATE + 17)
    curves = []

    for _ in range(180):
        index = rng.integers(0, len(plot_data), size=len(plot_data))
        sample = plot_data.iloc[index].sort_values("pred")
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
    plt.savefig(output, bbox_inches="tight")
    plt.close()

    result = {"calibration_intercept": intercept, "calibration_slope": slope, "x_max_plot": x_max}
    pd.DataFrame([result]).to_csv(table, index=False)
    return result


def decision_curve_plot(y: np.ndarray, p: np.ndarray, output: Path, table: Path, label: str) -> None:
    thresholds = np.arange(0.10, 0.50 + 1e-9, 0.01)
    middle, low, high = bootstrap_net_benefit_ci(y, p, thresholds)
    all_strategy = treat_all_net_benefit(y, thresholds)
    none_strategy = np.zeros_like(thresholds)

    plt.figure(figsize=(8, 6.8))
    plt.fill_between(thresholds, moving_average(low), moving_average(high), alpha=0.12, label=f"{label} 95% CI")
    plt.plot(thresholds, moving_average(middle), linewidth=2.4, label=label)
    plt.plot(thresholds, moving_average(all_strategy), linewidth=2.0, label="Treat all")
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
        "model_net_benefit": middle,
        "model_net_benefit_low": low,
        "model_net_benefit_high": high,
        "treat_all_net_benefit": all_strategy,
        "treat_none_net_benefit": none_strategy,
    }).to_csv(table, index=False)


def quartile_plot(y: np.ndarray, p: np.ndarray, output: Path, table: Path, label: str) -> pd.DataFrame:
    frame = pd.DataFrame({"y": y.astype(int), "pred": p.astype(float)}).dropna()
    frame["quartile"] = pd.qcut(frame["pred"], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
    grouped = frame.groupby("quartile", observed=False).agg(
        n=("y", "size"),
        events=("y", "sum"),
        event_rate=("y", "mean"),
        mean_pred=("pred", "mean"),
    ).reset_index()
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


def composite(paths: list[Path], output: Path) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--prediction-column", default="predicted_probability_frozen")
    parser.add_argument("--outcome-column", default="cv_event")
    parser.add_argument("--output", required=True)
    parser.add_argument("--label", default="AECOPD-CV model")
    args = parser.parse_args()

    style()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    data = normalise_columns(read_input(Path(args.input)))
    missing = [column for column in [args.outcome_column, args.prediction_column] if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    y = pd.to_numeric(data[args.outcome_column], errors="coerce")
    p = pd.to_numeric(data[args.prediction_column], errors="coerce")
    valid = y.isin([0, 1]) & p.notna() & np.isfinite(p)
    y = y[valid].astype(int).to_numpy()
    p = p[valid].astype(float).to_numpy()

    if len(np.unique(y)) < 2:
        raise ValueError("Outcome has only one class after filtering.")

    roc = roc_plot(y, p, output / "figure_roc_external_validation.png", output / "table_roc_external_validation.csv", args.label)
    calibration = calibration_plot(y, p, output / "figure_calibration_external_validation.png", output / "table_calibration_external_validation.csv", args.label)
    decision_curve_plot(y, p, output / "figure_decision_curve_external_validation.png", output / "table_decision_curve_external_validation.csv", args.label)
    quartiles = quartile_plot(y, p, output / "figure_quartiles_external_validation.png", output / "table_quartiles_external_validation.csv", args.label)
    composite([
        output / "figure_roc_external_validation.png",
        output / "figure_decision_curve_external_validation.png",
        output / "figure_calibration_external_validation.png",
        output / "figure_quartiles_external_validation.png",
    ], output / "external_validation_composite.png")

    summary = {
        "n": int(len(y)),
        "events": int(y.sum()),
        "event_rate": float(y.mean()),
        "prediction_column": args.prediction_column,
        "auc": float(roc["auc"]),
        "auc_ci_low": float(roc["auc_ci_low"]),
        "auc_ci_high": float(roc["auc_ci_high"]),
        "calibration_intercept": float(calibration["calibration_intercept"]),
        "calibration_slope": float(calibration["calibration_slope"]),
        "quartile_event_rates": {str(row["quartile"]): float(row["event_rate"]) for _, row in quartiles.iterrows()},
    }
    pd.DataFrame([summary]).to_csv(output / "summary_external_validation.csv", index=False)
    with open(output / "summary_external_validation.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Rows analysed: {summary['n']}")
    print(f"Events: {summary['events']}")
    print(f"AUC: {summary['auc']:.3f}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
