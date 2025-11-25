"""
ABMIL attention visualization for WSI patches.

This script visualizes ABMIL attention scores overlaid on WSI thumbnails
and extracts top-k high/low attention patches.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import openslide
from scipy.ndimage import gaussian_filter

Image.MAX_IMAGE_PIXELS = None

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent if script_dir.name == "figures" else script_dir
sys.path.insert(0, str(project_root))

from model import HIMFSurvLightningModule, HIMFSurv

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger("visualize_abmil_attention")

PATCH_SIZE = 224
PATCH_STRIDE = 224


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize ABMIL attention for WSI")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--patient-id", type=str, required=True, help="Patient ID to visualize")
    parser.add_argument("--wsi-raw-dir", type=str, required=True, help="Directory containing raw WSI files")
    parser.add_argument("--wsi-feature-dir", type=str, required=True, help="Directory containing WSI aggregated feature files")
    parser.add_argument("--output-dir", type=str, default=".", help="Output directory")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top/bottom patches to extract")
    return parser.parse_args()


def load_patient_batch(patient_id: str, config: dict, device: torch.device) -> dict:
    """Load patient WSI features from pre-extracted feature files."""
    wsi_feature_dir = Path(config['wsi_feature_dir'])
    wsi_files = list(wsi_feature_dir.glob(f"{patient_id}_*_agg_layers.npy"))
    
    if not wsi_files:
        raise ValueError(f"WSI features not found for patient {patient_id}")
    
    wsi_features = np.load(wsi_files[0])
    
    batch = {
        "patient_id": [patient_id],
        "wsi": [torch.from_numpy(wsi_features).to(device)]
    }

    return batch


@torch.no_grad()
def extract_abmil_attention(model: HIMFSurv, batch: dict) -> Optional[np.ndarray]:
    """Extract ABMIL attention scores for WSI patches."""
    device = next(model.parameters()).device

    if 'wsi' not in batch:
        return None

    wsi_feat = batch['wsi'][0].to(device).float()
    N = wsi_feat.shape[0]
    wsi_flat = wsi_feat.reshape(N, -1)

    A = model.abmil_wsi.attention(wsi_flat)
    A = torch.softmax(A, dim=0)
    wsi_attention_weights = A.squeeze().cpu().numpy()

    return wsi_attention_weights


def get_patch_coordinates(wsi_path: Path, tissue_mask_path: Optional[Path] = None) -> List[Tuple[int, int]]:
    """Extract valid patch coordinates from WSI."""
    slide = openslide.OpenSlide(str(wsi_path))
    tissue_mask = None

    # Load tissue mask if provided
    if tissue_mask_path and tissue_mask_path.exists():
        mask_img = Image.open(tissue_mask_path).convert("L")
        tissue_mask = np.array(mask_img)

    width, height = slide.level_dimensions[0]
    coords: List[Tuple[int, int]] = []

    # Generate patch coordinates with stride
    for y in range(0, height - PATCH_SIZE + 1, PATCH_STRIDE):
        for x in range(0, width - PATCH_SIZE + 1, PATCH_STRIDE):
            # Filter by tissue mask if available
            if tissue_mask is not None:
                # Scale coordinates to mask resolution
                mask_scale_x = tissue_mask.shape[1] / width
                mask_scale_y = tissue_mask.shape[0] / height
                mask_x, mask_y = int(x * mask_scale_x), int(y * mask_scale_y)
                mask_w = max(1, int(PATCH_SIZE * mask_scale_x))
                mask_h = max(1, int(PATCH_SIZE * mask_scale_y))
                mask_region = tissue_mask[
                    mask_y: min(mask_y + mask_h, tissue_mask.shape[0]),
                    mask_x: min(mask_x + mask_w, tissue_mask.shape[1]),
                ]
                # Skip patches with < 20% tissue content
                if mask_region.size > 0:
                    tissue_ratio = np.sum(mask_region > 0) / mask_region.size
                    if tissue_ratio < 0.2:
                        continue
            coords.append((x, y))

    slide.close()
    return coords


def visualize_abmil_attention(
    patient_id: str,
    attention_scores: np.ndarray,
    wsi_raw_dir: str,
    output_dir: Path,
    top_k: int = 5,
) -> None:
    """Visualize ABMIL attention on WSI"""
    wsi_path = Path(wsi_raw_dir) / patient_id / f"{patient_id}_1.tif"
    tissue_mask_path = Path(wsi_raw_dir) / patient_id / f"{patient_id}_1_tissue.tif"

    if not wsi_path.exists():
        LOGGER.warning("WSI file not found: %s", wsi_path)
        return

    coords = get_patch_coordinates(wsi_path, tissue_mask_path if tissue_mask_path.exists() else None)

    if len(coords) != len(attention_scores):
        LOGGER.warning("Coordinate count (%d) and attention score count (%d) don't match",
                       len(coords), len(attention_scores))
        min_len = min(len(coords), len(attention_scores))
        coords = coords[:min_len]
        attention_scores = attention_scores[:min_len]

    if len(attention_scores) == 0:
        LOGGER.warning("No attention scores. Skipping WSI visualization.")
        return

    # Resize WSI to fixed size for visualization
    slide = openslide.OpenSlide(str(wsi_path))
    width, height = slide.level_dimensions[0]
    thumbnail_size = 2000 
    scale = thumbnail_size / max(width, height)
    thumb_w, thumb_h = int(width * scale), int(height * scale)
    thumbnail = slide.get_thumbnail((thumb_w, thumb_h))

    # Map attention scores to thumbnail coordinates
    heatmap = np.zeros((thumb_h, thumb_w), dtype=np.float32)
    counts = np.zeros((thumb_h, thumb_w), dtype=np.int32)

    # Accumulate attention scores for each patch location
    for (x, y), score in zip(coords, attention_scores):
        # Scale patch coordinates to thumbnail size
        x_thumb = int(x * scale)
        y_thumb = int(y * scale)
        patch_w = max(1, int(PATCH_SIZE * scale))
        patch_h = max(1, int(PATCH_SIZE * scale))

        y_end = min(y_thumb + patch_h, thumb_h)
        x_end = min(x_thumb + patch_w, thumb_w)

        # Accumulate scores and count overlaps
        heatmap[y_thumb:y_end, x_thumb:x_end] += score
        counts[y_thumb:y_end, x_thumb:x_end] += 1

    # Average overlapping patches
    mask = counts > 0
    heatmap[mask] /= counts[mask]
    heatmap[~mask] = 0.0

    # Apply Gaussian filter for better visualization
    heatmap_to_draw = heatmap.copy()
    sigma = max(1, int(max(heatmap.shape) * 0.01))
    heatmap_to_draw = gaussian_filter(heatmap_to_draw, sigma=sigma, mode="nearest")

    # Normalize to [0, 1]
    if np.max(heatmap_to_draw) > 0:
        heatmap_to_draw = heatmap_to_draw / np.max(heatmap_to_draw)

    # Top / Bottom patch indices
    top_k = min(top_k, len(attention_scores))
    top_k_indices = np.argsort(attention_scores)[-top_k:][::-1]
    bottom_k_indices = np.argsort(attention_scores)[:top_k]

    # Create patient_id folder structure
    patient_dir = output_dir / patient_id
    abmil_dir = patient_dir / "abmil"
    abmil_dir.mkdir(parents=True, exist_ok=True)
    
    # Save heatmap
    fig_heat, ax_heat = plt.subplots(figsize=(6, 6))
    ax_heat.imshow(thumbnail)
    ax_heat.imshow(heatmap_to_draw, cmap="jet", alpha=0.55, vmin=0.0, vmax=1.0)
    ax_heat.axis("off")
    heatmap_path = abmil_dir / "heatmap.png"
    fig_heat.savefig(heatmap_path, bbox_inches="tight", pad_inches=0, dpi=150)
    plt.close(fig_heat)
    LOGGER.info("ABMIL heatmap saved: %s", heatmap_path)
    
    # Save individual patches
    for rank, idx in enumerate(top_k_indices, start=1):
        x, y = coords[idx]
        patch = slide.read_region((x, y), 0, (PATCH_SIZE, PATCH_SIZE)).convert("RGB")
        patch_path = abmil_dir / f"high_patch_{rank:02d}.png"
        patch.save(patch_path)
        LOGGER.info("High patch %d saved: %s", rank, patch_path)
    
    for rank, idx in enumerate(bottom_k_indices, start=1):
        x, y = coords[idx]
        patch = slide.read_region((x, y), 0, (PATCH_SIZE, PATCH_SIZE)).convert("RGB")
        patch_path = abmil_dir / f"low_patch_{rank:02d}.png"
        patch.save(patch_path)
        LOGGER.info("Low patch %d saved: %s", rank, patch_path)

    slide.close()


def main():
    args = parse_args()

    # Auto-detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "modalities": ["wsi"],
        "wsi_feature_dir": args.wsi_feature_dir,
        "wsi_feature_dim": 2048,
        "num_time_bins": 15,
    }

    # Load model
    LOGGER.info("Loading checkpoint: %s", args.ckpt)
    lit_module = HIMFSurvLightningModule.load_from_checkpoint(args.ckpt, map_location="cpu")
    lit_module.eval()
    lit_module.to(device)
    model: HIMFSurv = lit_module.model

    # Load patient data
    batch = load_patient_batch(args.patient_id, config, device)

    # Extract ABMIL attention
    LOGGER.info("Extracting ABMIL attention scores...")
    attention_scores = extract_abmil_attention(model, batch)

    if attention_scores is None:
        raise RuntimeError("Failed to extract ABMIL attention scores")

    # Visualize
    visualize_abmil_attention(
        args.patient_id,
        attention_scores,
        args.wsi_raw_dir,
        output_dir,
        top_k=args.top_k,
    )

    LOGGER.info("Visualization complete. Output directory: %s", output_dir)


if __name__ == "__main__":
    main()

