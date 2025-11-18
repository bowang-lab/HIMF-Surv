"""
WSI feature extraction pipeline
"""

import logging
import typing as tp
import gc
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Union

import numpy as np
import pydantic
import torch
from PIL import Image
from torch import nn
from tqdm import tqdm
import openslide

import timm
from timm.data import create_transform, resolve_data_config

# Disable PIL image size limit to prevent decompression bombs
Image.MAX_IMAGE_PIXELS = None

logger = logging.getLogger(__name__)

class WSI_FeatureExtractor(pydantic.BaseModel):
    """
    This class handles loading, preprocessing, and feature extraction from WSI data using the UNI model.
    """
    name: tp.Literal["WSI_FeatureExtractor"] = "WSI_FeatureExtractor"
    device: tp.Literal["auto", "cpu", "cuda"] = "auto"

    wsi_dir: str  # Directory containing WSI data files
    layers: list[float] = [0.5, 0.75, 1.0]  # Layer indices (as fractions) to extract features from
    
    # WSI processing parameters
    patch_level: int = 0
    patch_size: int = 224
    patch_stride: int = 224
    
    # Feature extraction parameters
    batch_size: int = 512

    model_config = pydantic.ConfigDict(protected_namespaces=(), extra="forbid")
    _model: nn.Module = pydantic.PrivateAttr()
    _transform: tp.Any = pydantic.PrivateAttr()

    def model_post_init(self, log__: tp.Any) -> None:
        """
        Initialize device after model creation.
        If device is "auto", automatically select CUDA if available, otherwise CPU.
        """
        super().model_post_init(log__)
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Device set to '{self.device}'.")

    @property
    def model(self) -> nn.Module:
        """
        Load UNI model and apply pretrained weights.
        """
        if "_model" not in self.__pydantic_private__:
            try:
                logger.info("Loading UNI model from HuggingFace Hub...")
                
                self._model = timm.create_model(
                    "hf-hub:MahmoodLab/UNI",
                    pretrained=True,
                    init_values=1e-5,
                    dynamic_img_size=True,
                )
                
                self._model.to(self.device)
                self._model.eval()

                config = resolve_data_config({}, model=self._model)
                self._transform = create_transform(**config)

                logger.info(f"UNI model loaded successfully on {self.device}.")
            except Exception as e:
                logger.error(f"Failed to load UNI model: {e}", exc_info=True)
                raise
        return self._model

    @property
    def transform(self):
        """
        Get the image transform for preprocessing.
        """
        if not hasattr(self, "_transform"):
            _ = self.model
        return self._transform
    
    def _aggregate_layers(self, latents: np.ndarray, layers: list[float] = None) -> np.ndarray:
        """
        Aggregate intermediate layer features from ViT using group_mean.
        
        Args:
            latents: Array of shape (L, D) where L is number of layers, D is feature dimension
            layers: List of layer indices as fractions (e.g., [0.5, 0.75, 1.0] for middle, 3/4, and final layers)
        """
        if layers is None:
            layers = self.layers
            
        layer_indices = np.unique([int(i * (latents.shape[0] - 1)) for i in layers]).tolist()
        if len(layer_indices) == 1:
            return latents[layer_indices[0]]
        
        # Group layers and take mean within each group
        groups = []
        layer_indices[-1] += 1
        for l1, l2 in zip(layer_indices[:-1], layer_indices[1:]):
            groups.append(latents[l1:l2].mean(0))
        return np.stack(groups)

    def extract_wsi_raw_features(
        self, 
        wsi_id: str, 
        raw_feature_dir: Union[str, Path],
        save_output: bool = True
    ) -> Path | None:
        """
        Extract and save raw layer features for a single WSI case.

        Args:
            wsi_id: WSI identifier
            raw_feature_dir: Directory to save raw layer features
            save_output: Whether to save the extracted features to disk
        """
        
        raw_feature_dir = Path(raw_feature_dir)
        raw_feature_path = raw_feature_dir / f"{wsi_id}_all_layers.npy"

        if save_output and raw_feature_path.exists():
            logger.info(f"Raw layer features already exist for {wsi_id}. Skipping extraction.")
            return raw_feature_path

        wsi_dir = Path(self.wsi_dir)
        patient_id = wsi_id.split("_")[0]

        wsi_path = wsi_dir / patient_id / f"{wsi_id}.tif"
        if not wsi_path.exists():
            raise FileNotFoundError(f"WSI file not found: {wsi_path}")

        tissue_mask_path = wsi_dir / patient_id / f"{wsi_id}_tissue.tif"
        if not tissue_mask_path.exists():
            tissue_mask_path = None
        
        # Extract all layer features (N, L, D) in streaming mode and save directly to disk via memmap
        try:
            success, final_shape = self._extract_all_layers_memmap(wsi_path, tissue_mask_path, raw_feature_path)
        except Exception as e:
            logger.error(f"Failed during memmap extraction for {wsi_id}: {e}", exc_info=True)
            return None

        if not success:
            logger.warning(f"No features were extracted for {wsi_id}. Nothing to save.")
            return None

        logger.info(f"Saved raw layer features with shape {final_shape} to {raw_feature_path}.")
        return raw_feature_path

    def _extract_all_layers_memmap(self, wsi_path: Path, tissue_mask_path: Path | None, raw_feature_path: Path) -> tuple[bool, tuple[int, int, int] | None]:
        """
        Extract all intermediate layer features from the UNI model.
        
        Args:
            wsi_path: Path to the WSI file
            tissue_mask_path: Path to the tissue mask file (optional, for filtering tissue regions)
            raw_feature_path: Path to save raw layer features as memmap file
        """
        slide = openslide.OpenSlide(str(wsi_path))

        tissue_mask = None
        if tissue_mask_path and tissue_mask_path.exists():
            mask_img = Image.open(tissue_mask_path).convert("L")
            tissue_mask = np.array(mask_img)

        width, height = slide.level_dimensions[self.patch_level]

        # Build coords list
        coords: list[tuple[int, int]] = []
        for y in range(0, height - self.patch_size + 1, self.patch_stride):
            for x in range(0, width - self.patch_size + 1, self.patch_stride):
                if tissue_mask is not None:
                    mask_scale_x = tissue_mask.shape[1] / width
                    mask_scale_y = tissue_mask.shape[0] / height
                    mask_x, mask_y = int(x * mask_scale_x), int(y * mask_scale_y)
                    mask_w = max(1, int(self.patch_size * mask_scale_x))
                    mask_h = max(1, int(self.patch_size * mask_scale_y))
                    mask_region = tissue_mask[
                        mask_y : min(mask_y + mask_h, tissue_mask.shape[0]),
                        mask_x : min(mask_x + mask_w, tissue_mask.shape[1]),
                    ]
                    if mask_region.size > 0:
                        tissue_ratio = np.sum(mask_region > 0) / mask_region.size
                        if tissue_ratio < 0.2:
                            continue
                coords.append((x, y))

        if len(coords) == 0:
            slide.close()
            return False, None

        # Probe one mini-batch to infer (L, D)
        with torch.no_grad():
            probe_imgs = [
                slide.read_region(coords[0], self.patch_level, (self.patch_size, self.patch_size)).convert("RGB")
            ]
            probe_tensor = torch.stack([self.transform(img) for img in probe_imgs]).to(self.device)

            intermediate_features = []
            def hook_fn(module, input, output):
                intermediate_features.append(output.detach().cpu())

            hooks = []
            if hasattr(self.model, "blocks"):
                for block in self.model.blocks:
                    hooks.append(block.register_forward_hook(hook_fn))

            _ = self.model(probe_tensor)

            for h in hooks:
                h.remove()

            if not intermediate_features:
                slide.close()
                return False, None

            layer_outputs = torch.stack(intermediate_features)
            cls_tokens = layer_outputs[:, :, 0, :]  # (L, 1, D)
            L, D = cls_tokens.shape[0], cls_tokens.shape[-1]

        N = len(coords)
        raw_feature_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to temporary file first to prevent incomplete files on kill
        temp_path = raw_feature_path.with_suffix('.tmp.npy')
        mmap = np.lib.format.open_memmap(temp_path, mode="w+", dtype=np.float32, shape=(N, L, D))

        write_idx = 0
        try:
            with torch.no_grad():
                for i in tqdm(range(0, N, self.batch_size), desc="Extracting (memmap)"):
                    batch_coords = coords[i:i + self.batch_size]
                    # Read patches from WSI
                    image_batch = [
                        slide.read_region((x, y), self.patch_level, (self.patch_size, self.patch_size)).convert("RGB")
                        for (x, y) in batch_coords
                    ]
                    # Preprocess patches using UNI model transform
                    batch_tensor = torch.stack([self.transform(img) for img in image_batch]).to(self.device)

                    # Register hooks to capture intermediate layer outputs
                    intermediate_features = []
                    def hook_fn2(module, input, output):
                        intermediate_features.append(output.detach().cpu())

                    hooks2 = []
                    if hasattr(self.model, "blocks"):
                        for block in self.model.blocks:
                            hooks2.append(block.register_forward_hook(hook_fn2))

                    # Forward pass
                    _ = self.model(batch_tensor)

                    # Remove hooks after forward pass
                    for h in hooks2:
                        h.remove()

                    # Stack all layer outputs and extract CLS tokens
                    if intermediate_features:
                        layer_outputs = torch.stack(intermediate_features)
                        cls_tokens = layer_outputs[:, :, 0, :]  # (L, B, D)
                        B = cls_tokens.shape[1]
                        end_idx = min(write_idx + B, N)
                        actual_B = end_idx - write_idx
                        if actual_B > 0:
                            mmap[write_idx:end_idx, :, :] = cls_tokens.permute(1, 0, 2).numpy()[:actual_B]
                            write_idx = end_idx
                        if write_idx >= N:
                            break

                    del batch_tensor, image_batch, intermediate_features

            slide.close()
            
            # Move to final location if completed successfully
            mmap.flush()
            del mmap
            gc.collect()
            temp_path.rename(raw_feature_path)

            return True, (N, L, D)
            
        except Exception as e:
            slide.close()
            # Clean up temporary file on error
            if temp_path.exists():
                temp_path.unlink()
            raise e

    def aggregate_wsi_features(
        self, 
        wsi_id: str, 
        layers: list[float], 
        raw_feature_dir: Union[str, Path],
        aggregated_feature_dir: Union[str, Path],
        save_output: bool = True
    ) -> None:
        """
        Aggregate raw layer features with specific layer configuration.

        Args:
            wsi_id: WSI identifier 
            layers: List of layer indices as fractions (e.g., [0.5, 0.75, 1.0])
            raw_feature_dir: Directory containing raw layer features
            aggregated_feature_dir: Directory to save aggregated features
            save_output: Whether to save the aggregated features to disk
        """
        raw_feature_dir = Path(raw_feature_dir)
        raw_feature_path = raw_feature_dir / f"{wsi_id}_all_layers.npy"
        
        if not raw_feature_path.exists():
            logger.warning(f"Raw layer features not found for {wsi_id}. Run extract_wsi_raw_features first.")
            return
        
        # Load raw layer features
        raw_features = np.load(raw_feature_path)  # (N, L, D)
        
        # Create output path
        aggregated_feature_dir = Path(aggregated_feature_dir)
        aggregated_feature_path = aggregated_feature_dir / f"{wsi_id}_agg_layers.npy"
        
        if save_output and aggregated_feature_path.exists():
            logger.info(f"Aggregated features already exist for {wsi_id}. Skipping.")
            return
        
        # Aggregate features for each patch using group_mean
        aggregated_features = []
        for patch_layers in raw_features:  # (L, D)
            aggregated = self._aggregate_layers(patch_layers, layers=layers)
            if aggregated.ndim == 1:
                aggregated = aggregated.reshape(1, -1)
            aggregated_features.append(aggregated)
        
        if not aggregated_features:
            logger.warning(f"No features were aggregated for {wsi_id}.")
            return
        
        aggregated_features = np.stack(aggregated_features)  # (N, G, D)
        
        if save_output:
            aggregated_feature_dir.mkdir(parents=True, exist_ok=True)
            # Save to temporary file first to prevent incomplete files on kill
            temp_path = aggregated_feature_path.with_suffix('.tmp.npy')
            try:
                np.save(temp_path, aggregated_features)
                # Move to final location if save was successful
                temp_path.rename(aggregated_feature_path)
                logger.info(
                    f"Saved {aggregated_features.shape[0]} patch features (shape: {aggregated_features.shape}) to {aggregated_feature_path}."
                )
            except Exception as e:
                # Clean up temporary file on error
                if temp_path.exists():
                    temp_path.unlink()
                raise e

    def extract_all_wsis_raw_features(
        self, 
        raw_feature_dir: Union[str, Path],
        save_output: bool = True, 
        num_workers: int = 1
    ) -> None:
        """
        Extract raw layer features for all WSI files.
        
        Args:
            raw_feature_dir: Directory to save raw layer features
            save_output: Whether to save the extracted features to disk
            num_workers: Number of parallel workers for processing (use 1 for GPU)
        """
        raw_feature_dir = Path(raw_feature_dir)
        wsi_dir = Path(self.wsi_dir)
        wsi_files = list(wsi_dir.glob("*/*_1.tif"))
        logger.info(f"Found {len(wsi_files)} WSI files to extract raw features")

        wsi_ids = [wsi_path.stem for wsi_path in wsi_files]
        
        if num_workers > 1:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                list(tqdm(
                    executor.map(
                        lambda wsi_id: self._extract_wsi_raw_features_wrapper(wsi_id, raw_feature_dir, save_output),
                        wsi_ids
                    ),
                    total=len(wsi_ids),
                    desc="Extracting raw layer features"
                ))
        else:
            for wsi_id in tqdm(wsi_ids, desc="Extracting raw layer features"):
                try:
                    self.extract_wsi_raw_features(wsi_id, raw_feature_dir, save_output=save_output)
                except Exception as e:
                    logger.error(f"Failed to extract raw features for {wsi_id}: {e}", exc_info=True)
                    continue

    def _extract_wsi_raw_features_wrapper(
        self, 
        wsi_id: str, 
        raw_feature_dir: Union[str, Path], 
        save_output: bool
    ) -> None:
        """
        Wrapper for parallel processing of WSI raw feature extraction.
        
        Args:
            wsi_id: WSI identifier
            raw_feature_dir: Directory to save raw layer features
            save_output: Whether to save the extracted features to disk
        """
        try:
            self.extract_wsi_raw_features(wsi_id, raw_feature_dir, save_output=save_output)
        except Exception as e:
            logger.error(f"Failed to extract raw features for {wsi_id}: {e}", exc_info=True)

    def aggregate_all_wsis_features(
        self, 
        raw_feature_dir: Union[str, Path],
        aggregated_feature_dir: Union[str, Path],
        save_output: bool = True,
        num_workers: int = 4
    ) -> None:
        """
        Aggregate raw layer features for all WSIs with layer configuration [0.5, 0.75, 1.0].
        
        Args:
            raw_feature_dir: Directory containing raw layer features
            aggregated_feature_dir: Base directory for saving aggregated features
            save_output: Whether to save output files
            num_workers: Number of parallel workers for processing
        """
        layers = [0.5, 0.75, 1.0]
        
        raw_feature_dir = Path(raw_feature_dir)
        aggregated_feature_dir = Path(aggregated_feature_dir)
        
        wsi_dir = Path(self.wsi_dir)
        wsi_files = list(wsi_dir.glob("*/*_1.tif"))
        wsi_ids = [wsi_path.stem for wsi_path in wsi_files]
        
        logger.info(f"Aggregating features for {len(wsi_ids)} WSIs with layer config {layers}.")
        
        # Process in parallel
        if num_workers > 1:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                list(tqdm(
                    executor.map(
                        lambda wsi_id: self.aggregate_wsi_features(
                            wsi_id, layers, raw_feature_dir, aggregated_feature_dir, save_output
                        ),
                        wsi_ids
                    ),
                    total=len(wsi_ids),
                    desc="Aggregating features"
                ))
        else:
            for wsi_id in tqdm(wsi_ids, desc="Aggregating features"):
                try:
                    self.aggregate_wsi_features(
                        wsi_id, layers, raw_feature_dir, aggregated_feature_dir, save_output=save_output
                    )
                except Exception as e:
                    logger.error(f"Failed to aggregate features for {wsi_id}: {e}", exc_info=True)
                    continue


def main():
    """
    1. Extract raw layer features for all WSIs
    2. Aggregate raw features with layer configuration [0.5, 0.75, 1.0]
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract WSI features using UNI model")
    parser.add_argument("--wsi_dir", type=str, required=True,
                        help="Directory containing WSI data files")
    parser.add_argument("--raw_feature_dir", type=str, required=True,
                        help="Directory to save raw layer features")
    parser.add_argument("--aggregated_feature_dir", type=str, required=True,
                        help="Directory to save aggregated features")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                        help="Device to use for computation")
    parser.add_argument("--num_workers_extract", type=int, default=1,
                        help="Number of workers for layer extraction")
    parser.add_argument("--num_workers_aggregate", type=int, default=4,
                        help="Number of workers for aggregation")
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    feature_extractor = WSI_FeatureExtractor(
        wsi_dir=args.wsi_dir,
        device=args.device
    )

    logger.info("Step 1: Extracting raw layer features for all WSIs...")
    feature_extractor.extract_all_wsis_raw_features(
        raw_feature_dir=args.raw_feature_dir,
        save_output=True, 
        num_workers=args.num_workers_extract
    )
    
    logger.info("Step 2: Aggregating features with layer configuration [0.5, 0.75, 1.0]...")
    
    feature_extractor.aggregate_all_wsis_features(
        raw_feature_dir=args.raw_feature_dir,
        aggregated_feature_dir=args.aggregated_feature_dir,
        save_output=True,
        num_workers=args.num_workers_aggregate
    )

    print("Processing complete.")
    print(f"Raw layer features saved to: {args.raw_feature_dir}")
    print(f"Aggregated features saved to: {args.aggregated_feature_dir}")


if __name__ == "__main__":
    main()