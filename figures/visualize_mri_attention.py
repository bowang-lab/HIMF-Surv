"""
MRI-PTPCa internal attention visualization.

This script visualizes internal ViT attention within MRI slices
using the MIMS-ViT model.
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import SimpleITK as sitk

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent if script_dir.name == "figures" else script_dir
sys.path.insert(0, str(project_root))

from feature_extractors.mri import MRI_FeatureExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger("visualize_mri_attention")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize MRI-PTPCa internal attention")
    parser.add_argument("--patient-id", type=str, required=True, help="Patient ID to visualize")
    parser.add_argument("--mri-raw-dir", type=str, required=True, help="Directory containing raw MRI files")
    parser.add_argument("--output-dir", type=str, default=".", help="Output directory")
    return parser.parse_args()


def visualize_mri_internal_attention(
    patient_id: str,
    mri_raw_dir: str,
    output_dir: Path,
    device: torch.device
) -> None:
    """Visualize MIMS-ViT internal ViT layer attention."""
    LOGGER.info("Loading MIMS-ViT model...")

    try:
        device_str = 'cuda' if device.type == 'cuda' else 'cpu'
        extractor = MRI_FeatureExtractor(
            mri_dir="",
            device=device_str
        )
        model = extractor.model
        model_device = next(model.parameters()).device
    except Exception as e:
        LOGGER.error(f"MRI_FeatureExtractor loading failed: {e}", exc_info=True)
        return

    mri_id = f"{patient_id}_0001"
    mri_dir = Path(mri_raw_dir) / patient_id
    t2_path = mri_dir / f"{mri_id}_t2w.mha"
    adc_path = mri_dir / f"{mri_id}_adc.mha"
    mask_path = mri_dir / f"{mri_id}_mask.mha"

    if not t2_path.exists() or not adc_path.exists():
        LOGGER.warning("MRI original files not found: %s or %s", t2_path, adc_path)
        return

    applied_mask_path = mask_path if mask_path.exists() else None

    # Load and preprocess MRI tensors
    try:
        t2_tensor = extractor.load_and_preprocess_mri(t2_path, applied_mask_path)
        adc_tensor = extractor.load_and_preprocess_mri(adc_path, applied_mask_path)
        t2_tensor = t2_tensor.to(model_device)
        adc_tensor = adc_tensor.to(model_device)
        dwi_tensor = torch.zeros_like(adc_tensor)
        combined_input = torch.cat((t2_tensor, adc_tensor, dwi_tensor), dim=1)
    except Exception as e:
        LOGGER.error(f"MRI data preprocessing failed: {e}", exc_info=True)
        return

    # Extract attention map via hooks
    attn_maps = []

    def make_attn_hook(attn_module):
        """Create a forward hook to extract attention weights from MIMS-ViT layers."""
        def hook(module, module_input, module_output):
            x = module_input[0]
            x_norm = module.norm(x)
            qkv = module.to_qkv(x_norm).chunk(3, dim=-1)
            q, k, v = qkv
            B, N, _ = q.shape
            head = module.heads
            dim_head = q.shape[-1] // head
            q = q.reshape(B, N, head, dim_head).permute(0, 2, 1, 3)
            k = k.reshape(B, N, head, dim_head).permute(0, 2, 1, 3)
            dots = torch.matmul(q, k.transpose(-1, -2)) * module.scale
            attn = module.attend(dots)
            attn_maps.append(attn.detach().cpu())
        return hook

    hooks = []
    try:
        vit_transformer = model.myViT_MM.transformer
        num_layers = len(vit_transformer.layers)
        for block in vit_transformer.layers:
            attn_module = block[0]
            hooks.append(attn_module.register_forward_hook(make_attn_hook(attn_module)))

        with torch.no_grad():
            LOGGER.info(f"Combined input shape: {combined_input.shape}")
            _ = model(combined_input)

    except Exception as e:
        LOGGER.error(f"MIMS-ViT attention hook execution failed: {e}", exc_info=True)
        return
    finally:
        for h in hooks:
            h.remove()

    if not attn_maps:
        LOGGER.warning("MIMS-ViT attention maps not extracted.")
        return

    if num_layers == 0:
        LOGGER.error("MIMS-ViT layer count cannot be detected.")
        return
    
    # Debug: Check attention map shapes
    LOGGER.info(f"First attention map shape: {attn_maps[0].shape}")
    LOGGER.info(f"Last attention map shape: {attn_maps[-1].shape}")

    half_idx = min(max(num_layers // 2, 0), num_layers - 1)
    three_quarter_idx = min(max((3 * num_layers) // 4, 0), num_layers - 1)
    layers_to_plot = {
        'midpoint': half_idx,
        'three_quarter': three_quarter_idx,
        'final': num_layers - 1,
    }
    LOGGER.info(f"MIMS-ViT (total {num_layers} layers) visualization targets: {layers_to_plot}")

    bbox_dict = {}
    slice_data_dict = {}
    modality_to_path = {"T2W": t2_path, "ADC": adc_path}

    for modality_name, img_path in modality_to_path.items():
        # Load MRI image and extract middle slice
        img = sitk.ReadImage(str(img_path))
        img_data = sitk.GetArrayFromImage(img)
        slice_idx = img_data.shape[0] // 2
        slice_img = img_data[slice_idx]

        # Normalize slice to [0, 1]
        slice_norm = slice_img.astype(np.float32)
        min_val, max_val = slice_norm.min(), slice_norm.max()
        slice_norm = (slice_norm - min_val) / (max_val - min_val) if (max_val > min_val) else np.zeros_like(slice_norm)

        # Crop to tissue region if mask is available
        bbox = None
        if applied_mask_path:
            mask_img = sitk.ReadImage(str(applied_mask_path))
            mask_data = sitk.GetArrayFromImage(mask_img)
            # Resize mask if dimensions don't match
            if mask_data.shape != img_data.shape:
                mask_tensor = torch.from_numpy(mask_data).float().unsqueeze(0).unsqueeze(0)
                resized_mask_tensor = F.interpolate(mask_tensor, size=img_data.shape, mode='nearest')
                mask_data = resized_mask_tensor.squeeze().numpy()

            # Find bounding box of tissue region
            mask_binary_slice = (mask_data[slice_idx] > 0).astype(np.uint8)
            nz = np.nonzero(mask_binary_slice)
            if len(nz[0]) > 0:
                y_idx, x_idx = nz
                y_min, y_max = int(y_idx.min()), int(y_idx.max() + 1)
                x_min, x_max = int(x_idx.min()), int(x_idx.max() + 1)
                if y_max > y_min and x_max > x_min:
                    bbox = (y_min, y_max, x_min, x_max) 

        bbox_dict[modality_name] = bbox
        slice_data_dict[modality_name] = slice_norm

    # Create patient_id folder structure
    patient_dir = output_dir / patient_id
    mri_dir = patient_dir / "mri"
    mri_dir.mkdir(parents=True, exist_ok=True)
    
    # Save individual heatmaps for each modality and layer
    for layer_name, layer_idx in layers_to_plot.items():
        layer_attn = attn_maps[layer_idx].squeeze(0) # (Heads, N, N)
        num_tokens = layer_attn.shape[-1]
        
        # --- [CORRECTION START: Adaptive Token Handling] ---
        # 48 patches (16 T2 + 16 ADC + 16 DWI)
        # If num_tokens == 48, it means NO CLS token (GAP is used).
        # If num_tokens == 49, it means 1 CLS token + 48 patches.
        
        if num_tokens == 48:
            # Case: No CLS token.
            # We calculate the average attention received by each patch from all other patches.
            # layer_attn: (Heads, N, N) -> mean(0) -> (N, N)
            # mean(0) over rows -> (N,) "Average attention paid to this patch by everyone"
            patch_scores = layer_attn.mean(dim=0).mean(dim=0).cpu().numpy()
            
        elif num_tokens == 49:
            # Case: With CLS token.
            # We take attention from CLS token (index 0) to all patches (index 1:).
            cls_to_patches = layer_attn[:, 0, 1:] # (Heads, 48)
            patch_scores = cls_to_patches.mean(dim=0)
            patch_scores = torch.softmax(patch_scores, dim=0).cpu().numpy()
            
        else:
            LOGGER.warning(f"Layer {layer_idx}: Unexpected token count {num_tokens}. Expected 48 or 49. Skipping.")
            continue
            
        num_patches = len(patch_scores)
        LOGGER.info(f"Layer {layer_idx} ({layer_name}): {num_patches} patches (derived from {num_tokens} tokens)")

        if num_patches != 48:
             LOGGER.warning(f"Layer {layer_idx}: Expected 48 patches, got {num_patches}. Skipping.")
             continue
             
        # Split by index (T2: 0-15, ADC: 16-31, DWI: 32-47)
        t2_scores = patch_scores[0:16]
        adc_scores = patch_scores[16:32]
        # dwi_scores = patch_scores[32:48]

        modalities_data = [
            ("T2W", t2_scores),
            ("ADC", adc_scores)
        ]
        # --- [CORRECTION END] ---

        for modality_name, modality_scores in modalities_data:
            # Already 16 scores, reshape to 4x4 grid (representing 16 depth slices/patches)
            grid = modality_scores.reshape(4, 4)
            
            # Normalize for visualization
            if grid.max() > grid.min():
                grid = (grid - grid.min()) / (grid.max() - grid.min())

            # Upscale from 4x4 grid to model's internal processing size
            grid_tensor = torch.tensor(grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            upscale = F.interpolate(
                grid_tensor,
                size=(extractor.img_size[1], extractor.img_size[2]),
                mode="bicubic", align_corners=False
            ).squeeze().numpy()

            # Resize to original slice size for overlay
            slice_norm = slice_data_dict[modality_name]
            heat_tensor = torch.from_numpy(upscale).float().unsqueeze(0).unsqueeze(0)
            heat_resized_tensor = F.interpolate(
                heat_tensor, size=slice_norm.shape, mode="bicubic", align_corners=False
            )
            heatmap_resized = heat_resized_tensor.squeeze().numpy()
            heatmap_resized = np.clip(heatmap_resized, 0.0, 1.0)

            # Crop with BBox
            bbox = bbox_dict[modality_name]
            if bbox:
                y_min, y_max, x_min, x_max = bbox
                slice_display = slice_norm[y_min:y_max, x_min:x_max]
                heat_display = heatmap_resized[y_min:y_max, x_min:x_max]
            else:
                slice_display = slice_norm
                heat_display = heatmap_resized

            # Save individual heatmap
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(slice_display, cmap="gray")
            ax.imshow(heat_display, cmap="jet", alpha=0.25, vmin=0.0, vmax=1.0)
            ax.axis("off")
            output_path = mri_dir / f"{modality_name.lower()}_{layer_name}.png"
            fig.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=150)
            plt.close(fig)
            LOGGER.info(f"MRI {modality_name} {layer_name} layer attention saved: {output_path}")


def main():
    args = parse_args()

    # Auto-detect device with fallback to CPU if CUDA initialization fails
    try:
        if torch.cuda.is_available():
            # Try to actually initialize CUDA
            _ = torch.zeros(1).cuda()
            device = torch.device("cuda")
            LOGGER.info("Using device: cuda")
        else:
            device = torch.device("cpu")
            LOGGER.info("Using device: cpu")
    except RuntimeError as e:
        LOGGER.warning(f"CUDA initialization failed: {e}")
        LOGGER.info("Falling back to CPU")
        device = torch.device("cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Visualize MRI attention
    visualize_mri_internal_attention(
        args.patient_id,
        args.mri_raw_dir,
        output_dir,
        device
    )

    LOGGER.info("Visualization complete. Output directory: %s", output_dir)


if __name__ == "__main__":
    main()