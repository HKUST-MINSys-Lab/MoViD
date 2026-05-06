#!/usr/bin/env python3
"""Visualize orthogonality between MoViD motion and view features.

Typical usage:

1. Dump features after a forward pass:

   torch.save({
       "motion_context": network.motion_context.detach().cpu(),
       "view_feat": network.view_feat.detach().cpu(),
   }, "output/feature_dump.pt")

2. Plot orthogonality:

   python scripts/visualize_feature_orthogonality.py \
       --motion-feature output/feature_dump.pt --motion-key motion_context \
       --view-feature output/feature_dump.pt --view-key view_feat \
       --output-dir output/feature_orthogonality
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EPS = 1e-8


def _load_torch(path: Path) -> Any:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif isinstance(value, (list, tuple)):
        value = np.asarray(value)
    elif not isinstance(value, np.ndarray):
        value = np.asarray(value)
    return value.astype(np.float32, copy=False)


def _resolve_key(container: Any, key: Optional[str], path: Path) -> Any:
    if key is None:
        if isinstance(container, np.lib.npyio.NpzFile):
            keys = list(container.keys())
            if len(keys) == 1:
                return container[keys[0]]
            raise ValueError(f"{path} contains keys {keys}; pass the key explicitly.")
        if isinstance(container, dict):
            if len(container) == 1:
                return next(iter(container.values()))
            raise ValueError(f"{path} contains keys {list(container.keys())}; pass the key explicitly.")
        return container

    value = container
    for part in key.replace("/", ".").split("."):
        if isinstance(value, np.lib.npyio.NpzFile):
            value = value[part]
        elif isinstance(value, dict):
            value = value[part]
        elif isinstance(value, (list, tuple)):
            value = value[int(part)]
        else:
            value = getattr(value, part)
    return value


def load_feature(path: str, key: Optional[str] = None) -> np.ndarray:
    feature_path = Path(path).expanduser()
    suffix = feature_path.suffix.lower()

    if suffix == ".npy":
        value = np.load(feature_path, allow_pickle=True)
    elif suffix == ".npz":
        value = np.load(feature_path, allow_pickle=True)
    elif suffix in {".pt", ".pth", ".tar"} or feature_path.name.endswith(".pth.tar"):
        value = _load_torch(feature_path)
    elif suffix in {".pkl", ".pickle"}:
        with feature_path.open("rb") as handle:
            value = pickle.load(handle)
    else:
        raise ValueError(f"Unsupported feature file type: {feature_path}")

    if isinstance(value, np.ndarray) and value.dtype == object and value.shape == ():
        value = value.item()

    return _to_numpy(_resolve_key(value, key, feature_path))


def parse_slice(spec: Optional[str]) -> Optional[slice]:
    if spec is None:
        return None
    if ":" not in spec:
        idx = int(spec)
        return slice(idx, idx + 1)
    start, stop = spec.split(":", 1)
    return slice(int(start) if start else None, int(stop) if stop else None)


def flatten_feature(feature: np.ndarray, feature_slice: Optional[slice] = None) -> np.ndarray:
    if feature.ndim == 0:
        raise ValueError("Feature tensor must have at least one dimension.")
    if feature_slice is not None:
        feature = feature[..., feature_slice]
    if feature.ndim == 1:
        feature = feature[None, :]
    elif feature.ndim > 2:
        feature = feature.reshape(-1, feature.shape[-1])
    mask = np.isfinite(feature).all(axis=1)
    feature = feature[mask]
    norms = np.linalg.norm(feature, axis=1)
    return feature[norms > EPS]


def normalize_rows(feature: np.ndarray) -> np.ndarray:
    return feature / np.maximum(np.linalg.norm(feature, axis=1, keepdims=True), EPS)


def sample_rows(feature: np.ndarray, max_samples: int, rng: np.random.Generator) -> np.ndarray:
    if feature.shape[0] <= max_samples:
        return feature
    idx = rng.choice(feature.shape[0], size=max_samples, replace=False)
    return feature[idx]


def paired_cosine(motion: np.ndarray, view: np.ndarray) -> Optional[np.ndarray]:
    if motion.shape != view.shape:
        return None
    motion_n = normalize_rows(motion)
    view_n = normalize_rows(view)
    return np.sum(motion_n * view_n, axis=1)


def cross_cosine_matrix(
    motion: np.ndarray,
    view: np.ndarray,
    max_samples: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    motion_sample = normalize_rows(sample_rows(motion, max_samples, rng))
    view_sample = normalize_rows(sample_rows(view, max_samples, rng))
    return motion_sample @ view_sample.T


def compute_metrics(motion: np.ndarray, view: np.ndarray, cross_cosine: np.ndarray) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "num_motion_vectors": int(motion.shape[0]),
        "num_view_vectors": int(view.shape[0]),
        "feature_dim": int(motion.shape[1]),
        "cross_mean_cosine": float(np.mean(cross_cosine)),
        "cross_mean_abs_cosine": float(np.mean(np.abs(cross_cosine))),
        "cross_rms_cosine": float(np.sqrt(np.mean(np.square(cross_cosine)))),
    }

    paired = paired_cosine(motion, view)
    if paired is not None:
        abs_paired = np.abs(paired)
        metrics.update(
            {
                "paired_mean_cosine": float(np.mean(paired)),
                "paired_mean_abs_cosine": float(np.mean(abs_paired)),
                "paired_median_abs_cosine": float(np.median(abs_paired)),
                "paired_p95_abs_cosine": float(np.percentile(abs_paired, 95)),
                "paired_max_abs_cosine": float(np.max(abs_paired)),
                "paired_rms_cosine": float(np.sqrt(np.mean(np.square(paired)))),
                "paired_view_leakage_energy_percent": float(np.mean(np.square(paired)) * 100.0),
            }
        )
    return metrics


def save_heatmap(cosine: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(cosine, vmin=-1.0, vmax=1.0, cmap="coolwarm", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("View feature sample")
    ax.set_ylabel("Motion feature sample")
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cosine similarity")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_histogram(cosine: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.hist(cosine, bins=50, color="#4C78A8", alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.axvline(0.0, color="#111827", linestyle="--", linewidth=1.4, label="orthogonal target")
    ax.axvline(float(np.mean(cosine)), color="#D62728", linewidth=1.4, label="mean")
    ax.set_title("Paired Motion-View Cosine Distribution")
    ax.set_xlabel("cos(motion feature, view feature)")
    ax.set_ylabel("Count")
    ax.set_xlim(-1.0, 1.0)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_dashboard(
    cross_cosine: np.ndarray,
    paired: Optional[np.ndarray],
    metrics: Dict[str, Any],
    path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))
    heat_ax, hist_ax, line_ax, bar_ax = axes.ravel()

    im = heat_ax.imshow(cross_cosine, vmin=-1.0, vmax=1.0, cmap="coolwarm", aspect="auto")
    heat_ax.set_title("Cross-Sample Cosine Heatmap")
    heat_ax.set_xlabel("View feature")
    heat_ax.set_ylabel("Motion feature")
    heat_ax.set_xticks([])
    heat_ax.set_yticks([])
    fig.colorbar(im, ax=heat_ax, fraction=0.046, pad=0.04)

    if paired is not None:
        hist_ax.hist(paired, bins=50, color="#4C78A8", alpha=0.85, edgecolor="white", linewidth=0.5)
        hist_ax.axvline(0.0, color="#111827", linestyle="--", linewidth=1.2)
        hist_ax.axvline(float(np.mean(paired)), color="#D62728", linewidth=1.2)
        hist_ax.set_title("Paired Cosine Distribution")
        hist_ax.set_xlabel("cos(motion, view)")
        hist_ax.set_ylabel("Count")
        hist_ax.set_xlim(-1.0, 1.0)

        abs_paired = np.abs(paired)
        line_ax.plot(abs_paired, color="#2E86AB", linewidth=1.1)
        line_ax.axhline(float(np.mean(abs_paired)), color="#D62728", linestyle="--", linewidth=1.2)
        line_ax.set_title("Absolute Paired Cosine by Sample")
        line_ax.set_xlabel("Flattened sample index")
        line_ax.set_ylabel("|cos(motion, view)|")
        line_ax.set_ylim(0.0, min(1.0, max(0.05, float(np.percentile(abs_paired, 99)) * 1.2)))
    else:
        hist_ax.axis("off")
        hist_ax.text(
            0.5,
            0.5,
            "Paired cosine skipped\nmotion and view shapes differ",
            ha="center",
            va="center",
            fontsize=12,
        )
        line_ax.axis("off")

    names = ["cross |cos|", "cross RMS"]
    values = [metrics["cross_mean_abs_cosine"], metrics["cross_rms_cosine"]]
    if paired is not None:
        names += ["paired |cos|", "paired p95 |cos|"]
        values += [metrics["paired_mean_abs_cosine"], metrics["paired_p95_abs_cosine"]]

    bar_ax.bar(names, values, color=["#72B7B2", "#F58518", "#4C78A8", "#D62728"][: len(names)])
    bar_ax.set_title("Orthogonality Summary")
    bar_ax.set_ylabel("Lower is more orthogonal")
    bar_ax.set_ylim(0.0, min(1.0, max(values) * 1.25 if values else 1.0))
    bar_ax.tick_params(axis="x", rotation=20)
    bar_ax.grid(axis="y", alpha=0.25, linestyle="--")
    for i, value in enumerate(values):
        bar_ax.text(i, value, f"{value:.4f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Motion-View Feature Orthogonality", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_raw_vs_projected(raw_motion: np.ndarray, projected_motion: np.ndarray, view: np.ndarray, path: Path) -> Dict[str, float]:
    raw = paired_cosine(raw_motion, view)
    projected = paired_cosine(projected_motion, view)
    if raw is None or projected is None:
        raise ValueError("Raw/projected comparison requires raw motion, projected motion, and view tensors with matching shapes.")

    raw_abs = np.abs(raw)
    projected_abs = np.abs(projected)
    values = [np.mean(raw_abs), np.mean(projected_abs), np.percentile(raw_abs, 95), np.percentile(projected_abs, 95)]

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    labels = ["raw mean |cos|", "projected mean |cos|", "raw p95 |cos|", "projected p95 |cos|"]
    colors = ["#B279A2", "#4C78A8", "#F58518", "#72B7B2"]
    ax.bar(labels, values, color=colors)
    ax.set_title("Orthogonality Before and After Projection")
    ax.set_ylabel("Lower is more orthogonal")
    ax.set_ylim(0.0, min(1.0, max(values) * 1.25))
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    for i, value in enumerate(values):
        ax.text(i, value, f"{value:.4f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "raw_paired_mean_abs_cosine": float(np.mean(raw_abs)),
        "projected_paired_mean_abs_cosine": float(np.mean(projected_abs)),
        "raw_paired_p95_abs_cosine": float(np.percentile(raw_abs, 95)),
        "projected_paired_p95_abs_cosine": float(np.percentile(projected_abs, 95)),
    }


def validate_features(motion: np.ndarray, view: np.ndarray) -> None:
    if motion.shape[1] != view.shape[1]:
        raise ValueError(
            "Motion and view feature dimensions differ: "
            f"{motion.shape[1]} vs {view.shape[1]}. "
            "Pass --motion-slice/--view-slice if the saved tensor includes extra channels."
        )
    if motion.shape[0] == 0 or view.shape[0] == 0:
        raise ValueError("No valid feature vectors remain after flattening/filtering.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-feature", required=True, help="Path to projected motion feature tensor.")
    parser.add_argument("--view-feature", required=True, help="Path to view feature tensor.")
    parser.add_argument("--motion-key", default=None, help="Key for motion tensor inside dict/npz files.")
    parser.add_argument("--view-key", default=None, help="Key for view tensor inside dict/npz files.")
    parser.add_argument("--raw-motion-feature", default=None, help="Optional raw motion tensor before projection.")
    parser.add_argument("--raw-motion-key", default=None, help="Key for the optional raw motion tensor.")
    parser.add_argument("--motion-slice", default=None, help="Optional feature-dimension slice, e.g. 0:128.")
    parser.add_argument("--view-slice", default=None, help="Optional feature-dimension slice, e.g. 0:128.")
    parser.add_argument("--raw-motion-slice", default=None, help="Optional raw motion feature-dimension slice.")
    parser.add_argument("--output-dir", default="output/feature_orthogonality", help="Directory for figures and metrics.")
    parser.add_argument("--max-samples", type=int, default=128, help="Maximum vectors per side in the heatmap.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for heatmap sampling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    motion = flatten_feature(load_feature(args.motion_feature, args.motion_key), parse_slice(args.motion_slice))
    view = flatten_feature(load_feature(args.view_feature, args.view_key), parse_slice(args.view_slice))
    validate_features(motion, view)

    cross_cosine = cross_cosine_matrix(motion, view, args.max_samples, args.seed)
    paired = paired_cosine(motion, view)
    metrics = compute_metrics(motion, view, cross_cosine)

    save_heatmap(cross_cosine, output_dir / "motion_view_cross_cosine_heatmap.png", "Motion-View Cross Cosine Similarity")
    if paired is not None:
        save_histogram(paired, output_dir / "motion_view_paired_cosine_histogram.png")
    save_dashboard(cross_cosine, paired, metrics, output_dir / "motion_view_orthogonality_dashboard.png")

    if args.raw_motion_feature is not None:
        raw_motion = flatten_feature(
            load_feature(args.raw_motion_feature, args.raw_motion_key),
            parse_slice(args.raw_motion_slice),
        )
        if raw_motion.shape != motion.shape:
            raise ValueError(f"Raw motion shape {raw_motion.shape} does not match projected motion shape {motion.shape}.")
        metrics.update(save_raw_vs_projected(raw_motion, motion, view, output_dir / "raw_vs_projected_orthogonality.png"))

    with (output_dir / "orthogonality_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)

    print(f"Saved orthogonality visualization to {output_dir}")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
