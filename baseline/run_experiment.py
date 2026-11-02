"""Train and test one ablation row on a Colab GPU.

Examples (run on the Colab VM, from the folder that holds xccnet.py):

    python run_experiment.py --name A_baseline
    python run_experiment.py --name B_focal --loss focal --gamma 2
    python run_experiment.py --name C_no_mask --no-lung-mask

Each run writes to results/<name>/:
    metrics.json          validation and test metrics, at threshold 0.5 and at the
                          threshold picked on validation, plus AUC and history
    test_predictions.csv  one row per test film: file, label, p(pneumonia)
    val_predictions.csv   the same for validation
    failures.png          misclassified test films with Grad-CAM
    gradcam.png           the same 4 test films for every run, with Grad-CAM
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

import xccnet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--loss", choices=("bce", "focal"), default="bce")
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--no-lung-mask", action="store_true")
    parser.add_argument("--seed", type=int, default=42, help="model init and sampler")
    parser.add_argument("--split-seed", type=int, default=42, help="kept fixed so every run sees the same test films")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--work-dir", default="/content/work")
    parser.add_argument("--out-dir", default="/content/results")
    return parser.parse_args()


def save_predictions(path: Path, records, probability: np.ndarray) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "label", "p_pneumonia"])
        for (file_path, label), score in zip(records, probability):
            writer.writerow([os.path.basename(file_path), int(label), f"{float(score):.6f}"])


def cam_grid(model, records, probability, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    if not records:
        return
    figure, axes = plt.subplots(2, len(records), figsize=(2.6 * len(records), 5.4), squeeze=False)
    for column, ((file_path, label), score) in enumerate(zip(records, probability)):
        image = np.asarray(Image.open(file_path).convert("L"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(image).unsqueeze(0).repeat(3, 1, 1)
        camera, _ = xccnet.grad_cam(model, tensor, positive=True)
        camera = np.array(Image.fromarray((camera * 255).astype(np.uint8)).resize(image.shape[::-1]))
        truth = "normal" if label == 0 else "pneumonia"
        axes[0, column].imshow(image, cmap="gray")
        axes[0, column].set_title(f"true {truth}\np={score:.2f}", fontsize=8)
        axes[1, column].imshow(image, cmap="gray")
        axes[1, column].imshow(camera, cmap="jet", alpha=0.45)
        axes[0, column].axis("off")
        axes[1, column].axis("off")
    figure.suptitle(title, fontsize=10)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    started = time.time()
    out = Path(args.out_dir) / args.name
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu", flush=True)

    records = xccnet.collect_kermany_records(xccnet.resolve_kermany_root())
    split = xccnet.stratified_split(records, seed=args.split_seed)
    counts = xccnet.split_counts(split)
    print("split:", counts, flush=True)

    use_mask = not args.no_lung_mask
    cache = Path(args.work_dir) / ("cache_clahe_lung256" if use_mask else "cache_clahe256")
    segmenter = xccnet.LungSegmenter(device) if use_mask else None
    prepared = {
        name: xccnet.preprocess_records(rows, cache, segmenter=segmenter, batch_size=16)
        for name, rows in split.items()
    }
    del segmenter
    print(f"preprocessed into {cache} ({time.time() - started:.0f}s)", flush=True)

    xccnet.set_seed(args.seed)
    workers = 2 if device.type == "cuda" else 0
    loaders = {
        name: xccnet.make_loader(rows, train=(name == "train"), batch_size=args.batch_size, num_workers=workers)
        for name, rows in prepared.items()
    }
    model = xccnet.build_clip(device, freeze=True)
    fit = xccnet.fit(
        model,
        loaders["train"],
        loaders["val"],
        device,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=1e-5,
        temperature=0.07,
        l1_lambda=1e-6,
        patience=args.patience,
        checkpoint_path=Path(args.work_dir) / "checkpoints" / f"{args.name}.pt",
        loss_type=args.loss,
        gamma=args.gamma,
    )

    val_p, val_y = xccnet.predict_loader(model, loaders["val"], device)
    test_p, test_y = xccnet.predict_loader(model, loaders["test"], device)
    threshold = xccnet.pick_threshold(val_p, val_y)
    result = {
        "name": args.name,
        "config": vars(args) | {"lung_mask": use_mask},
        "counts": counts,
        "val": xccnet.binary_metrics(val_p, val_y) | {"auc": xccnet.roc_auc(val_p, val_y)},
        "test": xccnet.binary_metrics(test_p, test_y) | {"auc": xccnet.roc_auc(test_p, test_y)},
        "val_threshold": threshold,
        "test_at_val_threshold": xccnet.binary_metrics(test_p, test_y, threshold),
        "history": fit.history,
        "minutes": round((time.time() - started) / 60, 1),
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    save_predictions(out / "test_predictions.csv", prepared["test"], test_p)
    save_predictions(out / "val_predictions.csv", prepared["val"], val_p)

    wrong = [index for index, (score, label) in enumerate(zip(test_p, test_y)) if (score >= 0.5) != bool(label)]
    wrong = sorted(wrong, key=lambda index: -abs(test_p[index] - 0.5))[:10]
    cam_grid(model, [prepared["test"][i] for i in wrong], test_p[wrong], out / "failures.png",
             f"{args.name}: misclassified test films (most confident first)")
    normals = [i for i, (_, y) in enumerate(prepared["test"]) if y == 0][:2]
    pneumonias = [i for i, (_, y) in enumerate(prepared["test"]) if y == 1][:2]
    fixed = normals + pneumonias
    cam_grid(model, [prepared["test"][i] for i in fixed], test_p[fixed], out / "gradcam.png",
             f"{args.name}: same 4 test films in every run")

    test = result["test"]
    print(
        f"DONE {args.name}: acc {test['accuracy']:.2f}  sens {test['recall']:.2f}  "
        f"spec {test['specificity']:.2f}  auc {test['auc']:.4f}  fp {test['fp']}  fn {test['fn']}  "
        f"({result['minutes']} min)",
        flush=True,
    )


if __name__ == "__main__":
    main()
