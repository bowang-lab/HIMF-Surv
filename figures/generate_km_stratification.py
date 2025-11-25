"""
KM stratification figure generation.

This script generates Kaplan-Meier survival curves for a single model
using a fixed quantile threshold (q=0.2) to split patients into high-risk
and low-risk groups.
"""

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from torch.utils.data import DataLoader

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent if script_dir.name == "figures" else script_dir
sys.path.insert(0, str(project_root))

from model import HIMFSurvLightningModule
from dataset import HIMFSurvDataset, custom_collate_fn

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger("generate_km_stratification")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate KM stratification figure")
    parser.add_argument("--ckpt-dir", type=str, required=True, help="Directory containing checkpoint files (will search for fold_* checkpoints)")
    parser.add_argument("--labels-file", type=str, required=True, help="Path to labels CSV file")
    parser.add_argument("--wsi-feature-dir", type=str, required=True, help="Directory containing WSI aggregated feature files")
    parser.add_argument("--mri-feature-dir", type=str, required=True, help="Directory containing MRI aggregated feature files")
    parser.add_argument("--clinical-feature-dir", type=str, required=True, help="Directory containing clinical embedding feature files")
    parser.add_argument("--output-dir", type=str, default=".", help="Output directory")
    parser.add_argument("--q", type=float, default=0.2, help="Quantile threshold (default: 0.2)")
    return parser.parse_args()


def compute_survival_function(logits: torch.Tensor) -> np.ndarray:
    """Compute survival function from logits."""
    hazards = torch.sigmoid(logits)
    survival = torch.cumprod(1 - hazards, dim=1)
    return survival.cpu().numpy()


def predict_multimodal(
    ckpt: str,
    test_ids: List[str],
    config: Dict[str, Union[str, int, List[str]]],
    labels_df: pd.DataFrame,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """Load model checkpoint and generate predictions for test patients."""
    # Load model from checkpoint
    lit = HIMFSurvLightningModule.load_from_checkpoint(ckpt, map_location="cpu")
    lit.eval()
    lit.to(device)
    
    # Create dataset and dataloader
    ds = HIMFSurvDataset(test_ids, config, labels_df)
    dl = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=custom_collate_fn, num_workers=0)

    survs, hazards_all = [], []
    with torch.no_grad():
        for batch in dl:
            if batch is None:
                continue
            logits = lit.model(batch)
            surv = compute_survival_function(logits)
            survs.append(surv)
            hazards_all.append(torch.sigmoid(logits).cpu().numpy())
    
    if not survs:
        return np.array([]), np.array([])
    
    # Concatenate all batches
    return np.concatenate(survs, 0), np.concatenate(hazards_all, 0)


def make_risk_scores(
    survival: np.ndarray
) -> np.ndarray:
    """Compute risk scores from survival predictions."""
    return -survival.sum(axis=1)


def add_tiny_noise(x: np.ndarray, eps: float = 1e-8, seed: int = 42) -> np.ndarray:
    """Add tiny random noise to break ties in quantile calculation."""
    rng = np.random.default_rng(seed)
    return x + eps * rng.standard_normal(size=x.shape)


def standard_split_by_quantile(
    risk: np.ndarray,
    q: float
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Standard quantile split."""
    if not (0 < q < 1):
        return None

    r = add_tiny_noise(risk)
    thr_high = np.quantile(r, 1 - q)

    high = r >= thr_high
    low = r < thr_high

    if (high.sum() + low.sum()) != len(r) or np.any(high & low):
        return None

    return high, low


def km_with_ci(
    times: np.ndarray,
    events: np.ndarray,
    timeline: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute Kaplan-Meier survival curve with confidence intervals at specified time points."""
    kmf = KaplanMeierFitter()
    kmf.fit(times, events)
    surv = kmf.survival_function_at_times(timeline).values.astype(float).reshape(-1)
    
    ci_base = kmf.confidence_interval_
    idx = pd.Index(timeline, name=ci_base.index.name)
    comb = ci_base.index.union(idx)
    ci_df = ci_base.reindex(comb).sort_index().ffill().reindex(idx)
    
    lo = ci_df.iloc[:, 0].to_numpy(dtype=float).reshape(-1)
    hi = ci_df.iloc[:, 1].to_numpy(dtype=float).reshape(-1)
    return timeline, surv, lo, hi


def plot_km_stratification(
    times_low: np.ndarray,
    events_low: np.ndarray,
    times_high: np.ndarray,
    events_high: np.ndarray,
    p_logrank: float,
    output_path: Path,
    model_name: str = "HIMF-Surv"
) -> None:
    """Plot KM survival curves for high-risk and low-risk groups."""
    max_time = max(times_low.max() if len(times_low) > 0 else 0,
                   times_high.max() if len(times_high) > 0 else 0)
    shared_tl = np.linspace(0.0, float(max_time), 15)

    tl_lo, km_lo, lo_lo, lo_hi = km_with_ci(times_low, events_low, shared_tl)
    tl_hi, km_hi, hi_lo, hi_hi = km_with_ci(times_high, events_high, shared_tl)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(tl_lo, km_lo, color="green", linewidth=2, label="Low Risk")
    ax.fill_between(tl_lo, lo_lo, lo_hi, color="green", alpha=0.2)
    ax.plot(tl_hi, km_hi, color="red", linewidth=2, label="High Risk")
    ax.fill_between(tl_hi, hi_lo, hi_hi, color="red", alpha=0.2)
    ax.set_xlabel("Time (Months)", fontsize=16)
    ax.set_ylabel("Survival Probability", fontsize=16)
    ax.set_title(model_name, fontsize=20, pad=24)
    ax.text(0.5, 1.02, f"log-rank p = {p_logrank:.2e}", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=16)
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved KM stratification plot: %s", output_path)


def find_checkpoints(ckpt_dir: Path) -> Dict[int, Path]:
    """Find checkpoint files for each fold in the directory."""
    fold_pat = re.compile(r"fold[_-](\d+)", re.IGNORECASE)
    checkpoints: Dict[int, Path] = {}
    
    # Search for .ckpt files
    for ckpt_file in ckpt_dir.glob("*.ckpt"):
        match = fold_pat.search(ckpt_file.stem) or fold_pat.search(str(ckpt_file))
        if match:
            fold_num = int(match.group(1))
            if fold_num not in checkpoints:
                checkpoints[fold_num] = ckpt_file
                LOGGER.info("Found checkpoint for fold %d: %s", fold_num, ckpt_file)
    
    if not checkpoints:
        raise ValueError(f"No checkpoint files with fold_* pattern found in {ckpt_dir}")
    
    return checkpoints


def main() -> None:
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels_df = pd.read_csv(args.labels_file)
    
    fold_min = labels_df["fold"].min()
    fold_max = labels_df["fold"].max()
    LOGGER.info("Labels fold range: %d to %d", fold_min, fold_max)
    
    ckpt_dir = Path(args.ckpt_dir)
    if not ckpt_dir.exists():
        raise ValueError(f"Checkpoint directory does not exist: {ckpt_dir}")
    
    checkpoints = find_checkpoints(ckpt_dir)
    LOGGER.info("Found %d fold checkpoints", len(checkpoints))
    
    adjusted_checkpoints = {}
    for ckpt_fold, ckpt_path in checkpoints.items():
        adjusted_fold = ckpt_fold - 1
        if adjusted_fold >= 0:
            adjusted_checkpoints[adjusted_fold] = ckpt_path
    checkpoints = adjusted_checkpoints

    config = {
        "modalities": ["clinical", "mri", "wsi"],
        "wsi_feature_dir": args.wsi_feature_dir,
        "mri_feature_dir": args.mri_feature_dir,
        "wsi_feature_dim": 2048,
        "mri_feature_dim": 4096,
        "clinical_feature_dir": args.clinical_feature_dir,
        "labels_file": args.labels_file,
        "num_time_bins": 15,
    }

    # Predict for each fold using its corresponding checkpoint
    hi_times_all, hi_events_all = [], []
    lo_times_all, lo_events_all = [], []
    
    for fold_num in sorted(checkpoints.keys()):
        ckpt_path = checkpoints[fold_num]
        fold_df = labels_df[labels_df["fold"] == fold_num].copy()
        
        if len(fold_df) == 0:
            LOGGER.warning("No patients found for fold %d, skipping", fold_num)
            continue
        
        # Get patient IDs for this fold
        test_ids = [str(pid) for pid in fold_df["patient_id"].tolist()]
        LOGGER.info("Predicting for %d patients in fold %d using %s", len(test_ids), fold_num, ckpt_path.name)
        
        # Generate predictions using fold-specific checkpoint
        surv_pred, haz_pred = predict_multimodal(str(ckpt_path), test_ids, config, labels_df, device)
        
        if surv_pred.size == 0:
            LOGGER.warning("No predictions generated for fold %d, skipping", fold_num)
            continue

        # Calculate risk scores from survival predictions
        t = fold_df["time_to_follow-up/BCR"].to_numpy()
        e = fold_df["BCR"].to_numpy().astype(bool)
        risk = make_risk_scores(surv_pred)
        
        # Perform quantile split within this fold
        split = standard_split_by_quantile(risk, args.q)
        
        if split is None:
            LOGGER.warning("Failed to create valid split for fold %d with q=%.4f, skipping", fold_num, args.q)
            continue
        
        hi_mask, lo_mask = split
        
        # Collect high/low groups from each fold separately
        hi_times_all.append(t[hi_mask])
        hi_events_all.append(e[hi_mask])
        lo_times_all.append(t[lo_mask])
        lo_events_all.append(e[lo_mask])
    
    if not hi_times_all or not lo_times_all:
        raise RuntimeError("No valid splits generated for any fold")
    
    # Combine high/low groups across all folds
    combined_times_high = np.concatenate(hi_times_all)
    combined_events_high = np.concatenate(hi_events_all)
    combined_times_low = np.concatenate(lo_times_all)
    combined_events_low = np.concatenate(lo_events_all)
    
    LOGGER.info("Combined predictions from %d folds: %d high-risk, %d low-risk patients", 
                len(hi_times_all), len(combined_times_high), len(combined_times_low))

    # Calculate log-rank test
    lr = logrank_test(combined_times_low, combined_times_high, 
                      event_observed_A=combined_events_low, 
                      event_observed_B=combined_events_high)
    p_lr = float(lr.p_value)

    # Plot
    output_path = output_dir / f"km_stratification.png"
    plot_km_stratification(
        combined_times_low, combined_events_low,
        combined_times_high, combined_events_high,
        p_lr,
        output_path,
        model_name="HIMF-Surv"
    )

    LOGGER.info("Done. Output saved to: %s", output_path)


if __name__ == "__main__":
    main()

