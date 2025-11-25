#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
UNI (WSI) patch internal attention visualization.

This script visualizes internal ViT attention within the top-1 ABMIL attention patch
using the UNI model.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import openslide

Image.MAX_IMAGE_PIXELS = None

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent if script_dir.name == "figures" else script_dir
sys.path.insert(0, str(project_root))

from model import HIMFSurvLightningModule, HIMFSurv
from feature_extractors.wsi import WSI_FeatureExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger("visualize_uni_attention")

PATCH_SIZE = 224


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize UNI internal attention for WSI patch")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--patient-id", type=str, required=True, help="Patient ID to visualize")
    parser.add_argument("--wsi-raw-dir", type=str, required=True, help="Directory containing raw WSI files")
    parser.add_argument("--wsi-feature-dir", type=str, required=True, help="Directory containing WSI aggregated feature files")
    parser.add_argument("--output-dir", type=str, default=".", help="Output directory")
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
def get_top_patch_coord(model: HIMFSurv, batch: dict, patient_id: str, wsi_raw_dir: str) -> Tuple[int, int]:
    """Get coordinate of top-1 ABMIL attention patch."""
    device = next(model.parameters()).device

    if 'wsi' not in batch:
        raise ValueError("WSI data not available")

    wsi_feat = batch['wsi'][0].to(device).float()
    N = wsi_feat.shape[0]
    wsi_flat = wsi_feat.reshape(N, -1)

    # ABMIL attention calculation to find most important patch
    A = model.abmil_wsi.attention(wsi_flat)
    A = torch.softmax(A, dim=0)
    attention_scores = A.squeeze().cpu().numpy()

    # Get top-1 patch index
    top_idx = np.argmax(attention_scores)

    # Get patch coordinates from WSI
    wsi_path = Path(wsi_raw_dir) / patient_id / f"{patient_id}_1.tif"
    tissue_mask_path = Path(wsi_raw_dir) / patient_id / f"{patient_id}_1_tissue.tif"
    
    if not wsi_path.exists():
        raise FileNotFoundError(f"WSI file not found: {wsi_path}")

    slide = openslide.OpenSlide(str(wsi_path))
    tissue_mask = None

    # Load tissue mask if provided
    if tissue_mask_path.exists():
        mask_img = Image.open(tissue_mask_path).convert("L")
        tissue_mask = np.array(mask_img)

    width, height = slide.level_dimensions[0]
    PATCH_STRIDE = 224
    coords = []

    # Generate patch coordinates with same filtering as feature extraction
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

    if top_idx >= len(coords):
        raise ValueError(f"Top patch index {top_idx} exceeds coordinate count {len(coords)}")

    return coords[top_idx]


def visualize_uni_patch_attention(
    wsi_path: Path,
    top_patch_coord: Tuple[int, int],
    patient_id: str,
    output_dir: Path,
    device: torch.device
) -> None:
    """Visualize UNI internal attention for top-1 patch"""
    LOGGER.info("Loading UNI model...")

    try:
        device_str = 'cuda' if device.type == 'cuda' else 'cpu'
        extractor = WSI_FeatureExtractor(
            wsi_dir="",
            device=device_str
        )
        model = extractor.model
        transform = extractor.transform
    except Exception as e:
        LOGGER.error(f"UNI model loading failed: {e}", exc_info=True)
        return

    # Load top-1 patch
    try:
        slide = openslide.OpenSlide(str(wsi_path))
        patch_img = slide.read_region(top_patch_coord, 0, (extractor.patch_size, extractor.patch_size)).convert("RGB")
        tensor = transform(patch_img).unsqueeze(0).to(device)
        slide.close()
    except Exception as e:
        LOGGER.error(f"Top-1 WSI patch loading failed: {e}", exc_info=True)
        return

    # Attention map extraction via forward hooks
    attn_maps = []

    def make_attn_hook_uni(attn_module):
        """Create a forward hook to extract attention weights from UNI ViT layers."""
        def hook(module, input, output):
            try:
                x = input[0]
                B, N, C = x.shape
                qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)
                attn = (q @ k.transpose(-2, -1)) * module.scale
                attn = attn.softmax(dim=-1)
                attn_maps.append(attn.detach().cpu())
            except Exception as e:
                LOGGER.warning(f"UNI attention hook internal error: {e}")
        return hook

    hooks = []
    try:
        num_layers = len(model.blocks)
        for block in model.blocks:
            hooks.append(block.attn.register_forward_hook(make_attn_hook_uni(block.attn)))

        with torch.no_grad():
            _ = model(tensor)

    except Exception as e:
        LOGGER.error(f"UNI attention hook execution failed: {e}", exc_info=True)
        return
    finally:
        for h in hooks:
            h.remove()

    if not attn_maps:
        LOGGER.warning("UNI attention maps not extracted.")
        return

    # Define layers to plot
    half_idx = min(max(num_layers // 2, 0), num_layers - 1)
    three_quarter_idx = min(max((3 * num_layers) // 4, 0), num_layers - 1)
    layers_to_plot = {
        'midpoint': half_idx,
        'three_quarter': three_quarter_idx,
        'final': num_layers - 1,
    }
    LOGGER.info(f"UNI (total {num_layers} layers) visualization targets: {layers_to_plot}")

    patch_dim = 14
    num_patches = patch_dim * patch_dim

    # Create patient_id folder structure
    patient_dir = output_dir / patient_id
    wsi_dir = patient_dir / "wsi"
    wsi_dir.mkdir(parents=True, exist_ok=True)
    
    # Save individual heatmaps for each layer
    for layer_name, layer_idx in layers_to_plot.items():
        if layer_idx >= len(attn_maps):
            LOGGER.warning(f"UNI layer index {layer_idx} out of range. Skipping.")
            continue

        attn_tensor = attn_maps[layer_idx].squeeze(0)
        if attn_tensor.shape[1] != (num_patches + 1):
            LOGGER.warning(f"UNI layer {layer_idx} token count mismatch ({attn_tensor.shape[1]} vs {num_patches+1}). Skipping.")
            continue

        # Extract CLS-to-patch attention
        cls_to_patches = attn_tensor[:, 0, 1:]
        patch_scores = cls_to_patches.mean(dim=0)

        # Reshape to 2D grid and normalize
        heatmap = patch_scores.reshape(patch_dim, patch_dim).cpu().numpy()
        if heatmap.max() > heatmap.min():
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

        # Upscale from 14x14 grid to original patch size
        heat_tensor = torch.from_numpy(heatmap).float().unsqueeze(0).unsqueeze(0)
        heat_resized = F.interpolate(
            heat_tensor, size=(PATCH_SIZE, PATCH_SIZE), mode="bicubic", align_corners=False
        ).squeeze().numpy()
        
        # Save individual heatmap
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(patch_img)
        ax.imshow(heat_resized, cmap="jet", alpha=0.4, vmin=0.0, vmax=1.0)
        ax.axis("off")
        output_path = wsi_dir / f"uni_attention_{layer_name}.png"
        fig.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=150)
        plt.close(fig)
        LOGGER.info(f"UNI {layer_name} layer attention saved: {output_path}")


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

    # Get top-1 patch coordinate
    LOGGER.info("Extracting top-1 ABMIL attention patch...")
    top_patch_coord = get_top_patch_coord(model, batch, args.patient_id, args.wsi_raw_dir)

    # Get WSI path
    wsi_path = Path(args.wsi_raw_dir) / args.patient_id / f"{args.patient_id}_1.tif"
    if not wsi_path.exists():
        raise FileNotFoundError(f"WSI file not found: {wsi_path}")

    # Visualize UNI attention
    visualize_uni_patch_attention(
        wsi_path,
        top_patch_coord,
        args.patient_id,
        output_dir,
        device
    )

    LOGGER.info("Visualization complete. Output directory: %s", output_dir)


if __name__ == "__main__":
    main()

