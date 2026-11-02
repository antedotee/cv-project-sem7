"""Build the ablation table and figures from the downloaded runs.

Run on the laptop after the runs are copied into experiments/<run name>/:

    python experiments/summarize.py

Writes experiments/ablation_table.md, roc_curves.png, and confusion_matrices.png.
Row A' needs no training: it reuses run A's saved probabilities and picks the
threshold on the validation films only.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ROWS = {
    "A": ("Baseline (cross-entropy)", "A_baseline"),
    "A'": ("Baseline, threshold tuned on val", "A_baseline"),
    "B": ("Focal loss, gamma = 2 (hypothesis)", "B_focal_g2"),
    "C": ("Baseline, lung mask off", "C_no_mask"),
}
SEED_SUFFIX = {42: "", 1: "_s1", 2: "_s2"}


def load_predictions(folder: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    with (folder / f"{split}_predictions.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    return np.array([float(r["p_pneumonia"]) for r in rows]), np.array([int(r["label"]) for r in rows])


def metrics(p: np.ndarray, y: np.ndarray, threshold: float) -> dict[str, float]:
    from sklearn.metrics import roc_auc_score

    pred = p >= threshold
    tp = int(np.sum(pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    return {
        "acc": 100 * (tp + tn) / len(y),
        "sens": 100 * tp / (tp + fn),
        "spec": 100 * tn / (tn + fp),
        "auc": roc_auc_score(y, p),
        "fp": fp,
        "fn": fn,
    }


def youden_threshold(p: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(y, p)
    return float(min(thresholds[int(np.argmax(tpr - fpr))], 1.0))


def row_results(key: str) -> list[dict[str, float]]:
    _, run = ROWS[key]
    results = []
    for seed, suffix in SEED_SUFFIX.items():
        folder = HERE / f"{run}{suffix}"
        if not (folder / "test_predictions.csv").exists():
            continue
        test_p, test_y = load_predictions(folder, "test")
        threshold = 0.5
        if key == "A'":
            threshold = youden_threshold(*load_predictions(folder, "val"))
        results.append(metrics(test_p, test_y, threshold) | {"seed": seed, "threshold": threshold})
    return results


def cell(values: list[float], digits: int) -> str:
    if len(values) == 1:
        return f"{values[0]:.{digits}f}"
    return f"{np.mean(values):.{digits}f} ± {np.std(values, ddof=1):.{digits}f}"


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    lines = [
        "| Row | Configuration | Seeds | Accuracy % | Sensitivity % | Specificity % | AUC | FP | FN |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    per_seed = ["", "| Row | Seed | Threshold | Acc % | Sens % | Spec % | AUC | FP | FN |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for key, (label, _) in ROWS.items():
        results = row_results(key)
        if not results:
            continue
        get = lambda name: [r[name] for r in results]
        lines.append(
            f"| {key} | {label} | {len(results)} | {cell(get('acc'), 2)} | {cell(get('sens'), 2)} | "
            f"{cell(get('spec'), 2)} | {cell(get('auc'), 4)} | {cell(get('fp'), 1)} | {cell(get('fn'), 1)} |"
        )
        for r in results:
            per_seed.append(
                f"| {key} | {r['seed']} | {r['threshold']:.3f} | {r['acc']:.2f} | {r['sens']:.2f} | "
                f"{r['spec']:.2f} | {r['auc']:.4f} | {r['fp']} | {r['fn']} |"
            )
    table = "\n".join(lines + per_seed) + "\n"
    (HERE / "ablation_table.md").write_text(
        "# Ablation results (585 test films: 427 pneumonia, 158 normal)\n\n"
        "Mean ± standard deviation over seeds. Same split for every run.\n\n" + table
    )
    print(table)

    # ROC curves and confusion matrices, seed 42 only.
    figure, axis = plt.subplots(figsize=(4.5, 4.5))
    confusion_figure, confusion_axes = plt.subplots(1, 3, figsize=(10, 3.2))
    for column, key in enumerate(("A", "B", "C")):
        folder = HERE / ROWS[key][1]
        if not folder.exists():
            continue
        p, y = load_predictions(folder, "test")
        fpr, tpr, _ = roc_curve(y, p)
        axis.plot(fpr, tpr, label=f"{key}: {ROWS[key][0]} (AUC {metrics(p, y, 0.5)['auc']:.3f})")
        m = metrics(p, y, 0.5)
        matrix = np.array([[158 - m["fp"], m["fp"]], [m["fn"], 427 - m["fn"]]])
        ax = confusion_axes[column]
        ax.imshow(matrix, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, matrix[i, j], ha="center", va="center", fontsize=12)
        ax.set_xticks([0, 1], ["pred normal", "pred pneumonia"], fontsize=8)
        ax.set_yticks([0, 1], ["true normal", "true pneumonia"], fontsize=8)
        ax.set_title(f"{key}: {ROWS[key][0]}", fontsize=8)
    axis.plot([0, 1], [0, 1], color="0.7", linestyle="--")
    axis.set_xlabel("1 - specificity")
    axis.set_ylabel("sensitivity")
    axis.legend(fontsize=6, loc="lower right")
    figure.tight_layout()
    figure.savefig(HERE / "roc_curves.png", dpi=150)
    confusion_figure.tight_layout()
    confusion_figure.savefig(HERE / "confusion_matrices.png", dpi=150)
    print("wrote ablation_table.md, roc_curves.png, confusion_matrices.png")


if __name__ == "__main__":
    main()
