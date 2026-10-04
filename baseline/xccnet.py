"""XCCNet baseline for pediatric pneumonia on the Kermany chest X-rays.

Paper
-----
Chandra Sekhar Raghaw, Parth Shirish Bhore, Mohammad Zia Ur Rehman, and
Nagendra Kumar. "An Explainable Contrastive-based Dilated Convolutional
Network with Transformer for Pediatric Pneumonia Detection." arXiv:2410.16143,
2024. https://arxiv.org/abs/2410.16143

The network follows the published classifier: a five-block dilated CNN (SFx),
a CLIP ViT-B/32 global encoder (CoTFx), cosine-similarity contrastive loss
between the two branches, and a two-layer head on the concatenated features
(Eq. 16: 2304 + 512 = 2816) with a sigmoid output.

Two published stages are not runnable from the paper alone. The ADA generator
is described only as a style-vector GAN (Eqs. 3-6), with no layer widths,
resolution schedule, or released weights. The rib suppressor is an image-to-image
ResNet that needs paired bone-suppressed targets, which the Guangzhou set does
not include. Lung segmentation is run with the public ChestX-Det PSPNet
(left lung + right lung) because Kermany images have no lung masks for the
paper's ResUNet++. Those substitutions are recorded in the notebook.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

IMAGE_SIZE = 256
SFX_DIM = 2304
CLIP_DIM = 512
FUSED_DIM = SFX_DIM + CLIP_DIM  # Eq. 16
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
PAPER_METRICS = {
    "no_augmentation": {"accuracy": 96.29, "precision": 94.31, "recall": 97.48, "f1": 95.87},
    "traditional_augmentation": {"accuracy": 97.18, "precision": 94.52, "recall": 98.65, "f1": 96.54},
    "full_stack": {"accuracy": 99.76, "precision": 99.75, "recall": 99.75, "f1": 99.75},
}


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DilatedBlock(nn.Module):
    """One SFx block: dilated 3x3 (rate 2), batch norm, ReLU, 2x2 pool, dropout 0.1."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            dilation=2,
            padding=2,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn(self.conv(x)), inplace=True))
        return self.drop(x)


class SpatialFeatureExtractor(nn.Module):
    """Five dilated blocks, filters 32-64-128-256-512.

    Five stride-2 pools on a 256 input produce an 8x8x512 map (32,768 values).
    The paper flattens SFx to 2,304 (Eq. 16). A 1x1 convolution to 36 channels
    makes 8x8x36 = 2,304, which keeps the spatial grid without a 75M-parameter
    dense layer the paper does not describe.
    """

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        widths = (32, 64, 128, 256, 512)
        blocks = []
        channels = in_channels
        for width in widths:
            blocks.append(DilatedBlock(channels, width))
            channels = width
        self.blocks = nn.ModuleList(blocks)
        self.project = nn.Conv2d(512, 36, kernel_size=1, bias=False)
        self.project_bn = nn.BatchNorm2d(36)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        x = F.relu(self.project_bn(self.project(x)), inplace=True)
        return torch.flatten(x, 1)


def _unfreeze_last_clip_blocks(clip_model: nn.Module, n_blocks: int) -> None:
    """Train the last ViT blocks, plus the final norm and projection."""
    visual = clip_model.visual
    blocks = list(visual.transformer.resblocks)
    if n_blocks > len(blocks):
        raise ValueError(f"ViT has {len(blocks)} blocks, cannot unfreeze {n_blocks}")
    for block in blocks[-n_blocks:]:
        for parameter in block.parameters():
            parameter.requires_grad = True
    if getattr(visual, "ln_post", None) is not None:
        for parameter in visual.ln_post.parameters():
            parameter.requires_grad = True
    projection = getattr(visual, "proj", None)
    if isinstance(projection, nn.Parameter):
        projection.requires_grad = True


class XCCNet(nn.Module):
    """SFx + CLIP ViT-B/32, fused into a two-layer classifier."""

    def __init__(self, clip_model: nn.Module, freeze_clip: bool = True, unfreeze_last_blocks: int = 0) -> None:
        super().__init__()
        self.sfx = SpatialFeatureExtractor()
        self.clip = clip_model
        self.unfreeze_last_blocks = unfreeze_last_blocks
        self.freeze_clip = bool(freeze_clip) and unfreeze_last_blocks <= 0
        if freeze_clip or unfreeze_last_blocks > 0:
            for parameter in self.clip.parameters():
                parameter.requires_grad = False
            self.clip.eval()
        if unfreeze_last_blocks > 0:
            _unfreeze_last_clip_blocks(self.clip, unfreeze_last_blocks)
            self.freeze_clip = False
        self.head_spatial = nn.Linear(SFX_DIM, 256)
        self.head_global = nn.Linear(CLIP_DIM, 256)
        self.fc1 = nn.Linear(FUSED_DIM, 512)
        self.fc2 = nn.Linear(512, 1)
        self.classifier_drop = nn.Dropout(0.1)
        mean = torch.tensor(CLIP_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(CLIP_STD).view(1, 3, 1, 1)
        self.register_buffer("clip_mean", mean)
        self.register_buffer("clip_std", std)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.unfreeze_last_blocks > 0:
            self.clip.eval()
            if mode:
                blocks = list(self.clip.visual.transformer.resblocks)
                for block in blocks[-self.unfreeze_last_blocks :]:
                    block.train()
                if getattr(self.clip.visual, "ln_post", None) is not None:
                    self.clip.visual.ln_post.train()
        elif self.freeze_clip:
            self.clip.eval()
        return self

    def encode_clip(self, images: torch.Tensor) -> torch.Tensor:
        resized = F.interpolate(images, size=224, mode="bicubic", align_corners=False)
        resized = (resized - self.clip_mean) / self.clip_std
        if self.freeze_clip:
            with torch.no_grad():
                features = self.clip.encode_image(resized)
            features = features.detach()
        else:
            features = self.clip.encode_image(resized)
        return features.float()

    def forward(self, images: torch.Tensor):
        spatial = self.sfx(images)
        if spatial.shape[-1] != SFX_DIM:
            raise RuntimeError(f"SFx produced {tuple(spatial.shape)}, expected (*, {SFX_DIM})")
        global_features = self.encode_clip(images)
        if global_features.shape[-1] != CLIP_DIM:
            raise RuntimeError(
                f"CLIP produced {tuple(global_features.shape)}, expected (*, {CLIP_DIM})"
            )
        fused = torch.cat([spatial, global_features], dim=1)
        hidden = self.classifier_drop(F.relu(self.fc1(fused), inplace=False))
        logit = self.fc2(hidden).squeeze(1)
        probability = torch.sigmoid(logit)
        return probability, logit, self.head_spatial(spatial), self.head_global(global_features)


def info_nce(spatial: torch.Tensor, global_features: torch.Tensor, temperature: float) -> torch.Tensor:
    """Symmetric InfoNCE. The positive pair is the two views of the same radiograph."""
    if spatial.shape[0] < 2:
        raise ValueError("Contrastive loss needs a batch of at least 2")
    left = F.normalize(spatial, dim=-1)
    right = F.normalize(global_features, dim=-1)
    logits = left @ right.T / temperature
    targets = torch.arange(left.shape[0], device=left.device)
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets))


def combined_loss(
    probability: torch.Tensor,
    labels: torch.Tensor,
    spatial: torch.Tensor,
    global_features: torch.Tensor,
    *,
    phi: float,
    temperature: float,
    l1_lambda: float,
    parameters: Iterable[torch.nn.Parameter],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Eq. 15: phi * contrastive + (1 - phi) * binary cross-entropy, plus an L1 term."""
    contrastive = info_nce(spatial, global_features, temperature)
    classification = F.binary_cross_entropy(probability, labels)
    l1_penalty = torch.zeros((), device=probability.device)
    if l1_lambda > 0:
        l1_penalty = torch.stack([parameter.abs().mean() for parameter in parameters]).mean()
    total = phi * contrastive + (1.0 - phi) * classification + l1_lambda * l1_penalty
    parts = {
        "loss": float(total.detach()),
        "contrastive": float(contrastive.detach()),
        "classification": float(classification.detach()),
        "phi": phi,
    }
    return total, parts


def phi_for_epoch(epoch: int) -> float:
    """Phi is inversely proportional to the epoch number (paper, Eq. 15).

    Epochs are counted from 1. Phi starts at 1/2 so the classification term is
    never switched off, then falls toward 0.
    """
    return 1.0 / (epoch + 1)


def binary_metrics(probability: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Pneumonia is the positive class, matching Table 2 of the paper."""
    predicted = (probability >= threshold).astype(np.int64)
    truth = labels.astype(np.int64)
    true_positive = int(np.sum((predicted == 1) & (truth == 1)))
    true_negative = int(np.sum((predicted == 0) & (truth == 0)))
    false_positive = int(np.sum((predicted == 1) & (truth == 0)))
    false_negative = int(np.sum((predicted == 0) & (truth == 1)))
    total = max(len(truth), 1)
    precision_den = true_positive + false_positive
    recall_den = true_positive + false_negative
    precision = true_positive / precision_den if precision_den else 0.0
    recall = true_positive / recall_den if recall_den else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "accuracy": 100.0 * (true_positive + true_negative) / total,
        "precision": 100.0 * precision,
        "recall": 100.0 * recall,
        "f1": 100.0 * f1,
        "tp": true_positive,
        "tn": true_negative,
        "fp": false_positive,
        "fn": false_negative,
        "n": int(len(truth)),
    }


@torch.no_grad()
def predict_loader(model: XCCNet, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for images, batch_labels in loader:
        images = images.to(device, non_blocking=True)
        probability, _, _, _ = model(images)
        probabilities.append(probability.float().cpu().numpy())
        labels.append(batch_labels.numpy())
    return np.concatenate(probabilities), np.concatenate(labels)


def evaluate(model: XCCNet, loader: DataLoader, device: torch.device) -> dict[str, float]:
    probability, labels = predict_loader(model, loader, device)
    return binary_metrics(probability, labels)


def trainable_state(model: XCCNet) -> dict[str, torch.Tensor]:
    """Save every non-CLIP tensor, plus any CLIP weights that are being fine-tuned."""
    clip_trainable = {
        name
        for name, parameter in model.named_parameters()
        if name.startswith("clip.") and parameter.requires_grad
    }
    state = {}
    for name, tensor in model.state_dict().items():
        if name.startswith("clip.") and name not in clip_trainable:
            continue
        state[name] = tensor.detach().cpu().clone()
    return state


def load_trainable(model: XCCNet, path: str | Path, device: torch.device) -> None:
    state = torch.load(path, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [name for name in missing if not name.startswith("clip.")]
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch. missing={missing} unexpected={unexpected}")


def train_one_epoch(
    model: XCCNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    epoch: int,
    temperature: float,
    l1_lambda: float,
    grad_clip: float = 1.0,
) -> dict[str, float]:
    model.train()
    phi = phi_for_epoch(epoch)
    totals = {"loss": 0.0, "contrastive": 0.0, "classification": 0.0}
    seen = 0
    use_cuda = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_cuda):
            probability, _, spatial, global_features = model(images)
        loss, parts = combined_loss(
            probability.float(),
            labels,
            spatial.float(),
            global_features.float(),
            phi=phi,
            temperature=temperature,
            l1_lambda=l1_lambda,
            parameters=parameters,
        )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(parameters, grad_clip)
        scaler.step(optimizer)
        scaler.update()
        batch = images.shape[0]
        seen += batch
        for key in totals:
            totals[key] += parts[key] * batch
    return {key: value / max(seen, 1) for key, value in totals.items()} | {"phi": phi}


@dataclass
class FitResult:
    best_path: Path
    best_val: dict[str, float]
    history: list[dict[str, float]]


def fit(
    model: XCCNet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    *,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    temperature: float,
    l1_lambda: float,
    patience: int,
    checkpoint_path: str | Path,
    clip_learning_rate: float | None = None,
) -> FitResult:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    main_params = []
    clip_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("clip."):
            clip_params.append(parameter)
        else:
            main_params.append(parameter)
    groups = [{"params": main_params, "lr": learning_rate}]
    if clip_params:
        groups.append({"params": clip_params, "lr": clip_learning_rate or learning_rate})
    optimizer = torch.optim.Adam(groups, weight_decay=weight_decay)
    # The paper multiplies the learning rate by 0.3 when validation accuracy stalls.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.3, patience=2, min_lr=1e-7
    )
    history: list[dict[str, float]] = []
    best_accuracy = -1.0
    best_val: dict[str, float] = {}
    stale = 0
    for epoch in range(1, epochs + 1):
        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch=epoch,
            temperature=temperature,
            l1_lambda=l1_lambda,
        )
        val_stats = evaluate(model, val_loader, device)
        scheduler.step(val_stats["accuracy"])
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "clip_lr": optimizer.param_groups[-1]["lr"],
            **{f"train_{key}": value for key, value in train_stats.items()},
            **{f"val_{key}": value for key, value in val_stats.items() if key in {"accuracy", "precision", "recall", "f1"}},
        }
        history.append(row)
        print(
            f"epoch {epoch:02d}  loss {train_stats['loss']:.4f}  "
            f"val acc {val_stats['accuracy']:.2f}  prec {val_stats['precision']:.2f}  "
            f"rec {val_stats['recall']:.2f}  f1 {val_stats['f1']:.2f}  "
            f"lr {row['lr']:.2e}  clip lr {row['clip_lr']:.2e}  phi {train_stats['phi']:.3f}"
        )
        if val_stats["accuracy"] > best_accuracy:
            best_accuracy = val_stats["accuracy"]
            best_val = val_stats
            stale = 0
            _save_checkpoint(trainable_state(model), checkpoint_path)
        else:
            stale += 1
            if stale >= patience:
                print(f"early stop at epoch {epoch} (no validation-accuracy gain for {patience} epochs)")
                break
    load_trainable(model, checkpoint_path, device)
    return FitResult(best_path=checkpoint_path, best_val=best_val, history=history)


def collect_kermany_records(root: str | Path) -> list[tuple[str, int]]:
    """Pool the Kaggle train, val, and test films into one list.

    The paper uses all 1,583 normal and 4,273 pneumonia radiographs, then draws
    a fresh 80/10/10 split. The Kaggle archive also nests a second copy of those
    same files under ``chest_xray/chest_xray``. Each filename is kept once.
    ``__MACOSX`` resource forks are ignored.
    """
    chosen: dict[str, tuple[str, int]] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name != "__MACOSX" and not name.startswith(".")]
        label_name = os.path.basename(dirpath).lower()
        if label_name not in {"normal", "pneumonia"}:
            continue
        label = 0 if label_name == "normal" else 1
        for filename in filenames:
            if filename.startswith("._") or not filename.lower().endswith((".jpeg", ".jpg", ".png")):
                continue
            path = os.path.join(dirpath, filename)
            previous = chosen.get(filename.lower())
            if previous is None or len(path) < len(previous[0]):
                chosen[filename.lower()] = (path, label)
    if not chosen:
        raise FileNotFoundError(f"No NORMAL/PNEUMONIA images found under {root}")
    return list(chosen.values())


def _save_checkpoint(state: dict, path: str | Path) -> None:
    """Write the checkpoint on local disk, then copy it. Drive uploads fail if torch writes them directly."""
    import shutil

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path("/tmp") / f"{path.name}.partial"
    torch.save(state, temporary)
    shutil.copy2(temporary, path)
    temporary.unlink(missing_ok=True)


def _kagglehub_version_dirs() -> list[Path]:
    roots = [
        Path.home() / ".cache" / "kagglehub",
        Path("/root/.cache/kagglehub"),
    ]
    cache_env = os.environ.get("KAGGLEHUB_CACHE", "").strip()
    if cache_env:
        roots.insert(0, Path(cache_env))
    bases = [
        root / "datasets" / "paultimothymooney" / "chest-xray-pneumonia" / "versions"
        for root in roots
    ]
    versions: list[Path] = []
    for base in bases:
        if not base.is_dir():
            continue
        versions.extend(path for path in base.iterdir() if path.is_dir())
    return sorted(versions)


def resolve_kermany_root(explicit: str = "") -> str:
    """Return the folder from ``kagglehub.dataset_download`` when it is already cached.

    A fresh download runs only when no cached copy contains the class folders.
    """
    if explicit.strip():
        return explicit.strip()
    env = os.environ.get("DATA_DIR", "").strip()
    if env:
        return env
    for version in _kagglehub_version_dirs():
        try:
            records = collect_kermany_records(version)
        except FileNotFoundError:
            continue
        normal = sum(label == 0 for _, label in records)
        pneumonia = sum(label == 1 for _, label in records)
        if normal >= 1500 and pneumonia >= 4000:
            return str(version)
    import kagglehub

    return kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")


def stratified_split(
    records: list[tuple[str, int]],
    seed: int = 42,
) -> dict[str, list[tuple[str, int]]]:
    """Stratified 80/10/10. Real images are split before any oversampling.

    Within each class the images are shuffled, then 10% is held out for
    validation and 10% for test. The remainder is the training split.
    """
    rng = np.random.RandomState(seed)
    grouped: dict[int, list[tuple[str, int]]] = {0: [], 1: []}
    for record in records:
        grouped[record[1]].append(record)
    split: dict[str, list[tuple[str, int]]] = {"train": [], "val": [], "test": []}
    for rows in grouped.values():
        rows = rows.copy()
        rng.shuffle(rows)
        count = len(rows)
        n_val = int(round(count * 0.1))
        n_test = int(round(count * 0.1))
        if n_val + n_test >= count:
            n_val = max(count // 10, 1)
            n_test = max(count // 10, 1)
        n_train = count - n_val - n_test
        split["train"].extend(rows[:n_train])
        split["val"].extend(rows[n_train : n_train + n_val])
        split["test"].extend(rows[n_train + n_val :])
    for rows in split.values():
        rng.shuffle(rows)
    return split


def split_counts(split: dict[str, list[tuple[str, int]]]) -> dict[str, dict[str, int]]:
    summary = {}
    for name, rows in split.items():
        normal = sum(label == 0 for _, label in rows)
        pneumonia = sum(label == 1 for _, label in rows)
        summary[name] = {"normal": normal, "pneumonia": pneumonia, "total": len(rows)}
    return summary


def resize_bicubic(image: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    import cv2

    return cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    """CLAHE with an 8x8 window and clip limit 2.0, as stated for the CXR enhancer."""
    import cv2

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    return clahe.apply(image)


def _load_gray(path: str) -> np.ndarray:
    image = Image.open(path).convert("L")
    return np.asarray(image, dtype=np.uint8)


class LungSegmenter:
    """ChestX-Det PSPNet stand-in for the paper's ResUNet++ lung mask.

    Targets index 4 and 5 are the left and right lung. Pixels outside the mask
    are set to 0, which is the crop the paper applies after the logical AND.
    """

    def __init__(self, device: torch.device) -> None:
        import torchxrayvision as xrv

        self.xrv = xrv
        self.device = device
        self.model = xrv.baseline_models.chestx_det.PSPNet().to(device)
        self.model.eval()

    @torch.no_grad()
    def mask_batch(self, images: list[np.ndarray]) -> list[np.ndarray]:
        import cv2

        batch = []
        for image in images:
            normalized = np.asarray(self.xrv.datasets.normalize(image, 255), dtype=np.float32)
            batch.append(torch.from_numpy(normalized.copy()))
        tensor = torch.stack(batch).unsqueeze(1).to(self.device)
        output = self.model(tensor)
        left = torch.sigmoid(output[:, 4])
        right = torch.sigmoid(output[:, 5])
        lung = ((left > 0.5) | (right > 0.5)).cpu().numpy().astype(np.uint8)
        masks = []
        for index, image in enumerate(images):
            mask = cv2.resize(lung[index], (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            if int(mask.sum()) < 64:
                mask = np.ones_like(image, dtype=np.uint8)
            masks.append(mask)
        return masks


def _cache_name(source: str, label: int) -> str:
    import hashlib

    digest = hashlib.sha1(os.path.abspath(source).encode()).hexdigest()[:20]
    return f"{digest}_{label}.png"


def preprocess_records(
    records: list[tuple[str, int]],
    cache_dir: str | Path,
    *,
    segmenter: LungSegmenter | None = None,
    batch_size: int = 8,
    progress: Callable[[int, int], None] | None = None,
) -> list[tuple[str, int]]:
    """Resize, CLAHE, optional lung mask. Writes 8-bit PNGs and skips existing files."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached: list[tuple[str, int]] = []
    pending: list[tuple[np.ndarray, Path]] = []

    def flush() -> None:
        if not pending:
            return
        images = [item[0] for item in pending]
        if segmenter is None:
            masks = [np.ones_like(image, dtype=np.uint8) for image in images]
        else:
            masks = segmenter.mask_batch(images)
        for (image, destination), mask in zip(pending, masks):
            Image.fromarray((image * mask).astype(np.uint8), mode="L").save(destination)
        pending.clear()

    for index, (source, label) in enumerate(records, start=1):
        destination = cache_dir / _cache_name(source, label)
        cached.append((str(destination), label))
        if destination.exists():
            if progress:
                progress(index, len(records))
            continue
        gray = apply_clahe(resize_bicubic(_load_gray(source)))
        pending.append((gray, destination))
        if len(pending) >= batch_size:
            flush()
        if progress:
            progress(index, len(records))
    flush()
    return cached


class ChestXrayDataset(Dataset):
    def __init__(self, records: list[tuple[str, int]], train: bool = False) -> None:
        self.records = records
        self.train = train
        if train:
            from torchvision import transforms

            self.augment = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(10),
                    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                ]
            )
        else:
            self.augment = None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        path, label = self.records[index]
        image = Image.open(path).convert("L")
        if self.augment is not None:
            image = self.augment(image)
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).unsqueeze(0).repeat(3, 1, 1)
        return tensor, torch.tensor(label, dtype=torch.long)


def make_loader(
    records: list[tuple[str, int]],
    *,
    train: bool,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    dataset = ChestXrayDataset(records, train=train)
    sampler = None
    if train:
        labels = np.array([label for _, label in records])
        class_count = np.bincount(labels, minlength=2).astype(np.float64)
        weights = 1.0 / class_count[labels]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=train,
    )


def grad_cam(model: XCCNet, image: torch.Tensor, positive: bool = True) -> tuple[np.ndarray, float]:
    """Grad-CAM on the last dilated convolution. ``image`` is 3x256x256 in [0, 1]."""
    model.eval()
    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}
    layer = model.sfx.blocks[-1].conv

    def save_activation(_module, _inputs, output):
        activations["value"] = output

    def save_gradient(_module, _grad_input, grad_output):
        gradients["value"] = grad_output[0]

    forward_hook = layer.register_forward_hook(save_activation)
    backward_hook = layer.register_full_backward_hook(save_gradient)
    try:
        batch = image.unsqueeze(0).to(next(model.parameters()).device)
        probability, logit, _, _ = model(batch)
        score = logit[0] if positive else -logit[0]
        model.zero_grad(set_to_none=True)
        score.backward()
        activation = activations["value"].detach()[0]
        gradient = gradients["value"].detach()[0]
        weights = gradient.mean(dim=(1, 2))
        camera = torch.relu((weights[:, None, None] * activation).sum(dim=0))
        camera = camera - camera.min()
        camera = camera / (camera.max() + 1e-8)
        return camera.cpu().numpy(), float(probability[0].detach())
    finally:
        forward_hook.remove()
        backward_hook.remove()


def build_clip(device: torch.device, freeze: bool = True, unfreeze_last_blocks: int = 0) -> XCCNet:
    import open_clip

    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model = XCCNet(
        clip_model,
        freeze_clip=freeze,
        unfreeze_last_blocks=unfreeze_last_blocks,
    ).to(device)
    return model


def smoke_test() -> None:
    """One forward and backward step on random tensors, using a stub in place of CLIP."""

    class StubClip(nn.Module):
        def encode_image(self, images: torch.Tensor) -> torch.Tensor:
            return torch.zeros(images.shape[0], CLIP_DIM, device=images.device)

    device = torch.device("cpu")
    model = XCCNet(StubClip(), freeze_clip=True).to(device)
    images = torch.rand(4, 3, IMAGE_SIZE, IMAGE_SIZE)
    labels = torch.tensor([0, 1, 1, 0], dtype=torch.float32)
    probability, _, spatial, global_features = model(images)
    loss, _ = combined_loss(
        probability,
        labels,
        spatial,
        global_features,
        phi=0.5,
        temperature=0.07,
        l1_lambda=1e-6,
        parameters=[parameter for parameter in model.parameters() if parameter.requires_grad],
    )
    loss.backward()
    if probability.shape != (4,):
        raise AssertionError(probability.shape)
    if spatial.shape != (4, 256) or global_features.shape != (4, 256):
        raise AssertionError((spatial.shape, global_features.shape))
    print(
        f"smoke ok  prob {tuple(probability.shape)}  contrastive {tuple(spatial.shape)}  "
        f"loss {float(loss.detach()):.4f}"
    )


if __name__ == "__main__":
    smoke_test()
