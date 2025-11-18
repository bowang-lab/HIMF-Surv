"""
MRI feature extraction pipeline adapted from the MRI-PTPCa implementation:
https://github.com/StandWisdom/MRI-based-Predicted-Transformer-for-Prostate-cancer.git
"""

import logging
import typing as tp
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Union

import numpy as np
import pydantic
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
import SimpleITK as sitk

from model.MRI_PTCa import myCNNViT_MM

logger = logging.getLogger(__name__)

class MRI_FeatureExtractor(pydantic.BaseModel):
    """
    This class handles loading, preprocessing, and feature extraction from MRI data using the MRI-PTCa model.
    """
    name: tp.Literal["MRI_FeatureExtractor"] = "MRI_FeatureExtractor"
    device: tp.Literal["auto", "cpu", "cuda"] = "auto"

    mri_dir: str  # Directory containing MRI data files
    t2_model_path: tp.Optional[str] = None  # Path to T2 extractor model weights
    adc_model_path: tp.Optional[str] = None  # Path to ADC extractor model weights
    vit_model_path: tp.Optional[str] = None  # Path to ViT fusion model weights
    
    img_size: tp.Tuple[int, int, int] = (16, 200, 200)  # (slices, height, width)
    layers: list[float] = [0.5, 0.75, 1.0]  # Layer indices (as fractions) to extract features from

    model_config = pydantic.ConfigDict(protected_namespaces=(), extra="forbid")
    _model: nn.Module = pydantic.PrivateAttr()

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
        Load MRI_PTCa model and apply pretrained weights correctly.
        """
        if "_model" not in self.__pydantic_private__:
            try:
                logger.info("Loading MRI_PTCa model from MRI_PTCa.py...")
                model = myCNNViT_MM()
                
                model.batchsize = 1
                model.net_t2.batchsize = 1
                model.net_adc.batchsize = 1
                model.net_dwi.batchsize = 1
                
                # Interpolate position embedding from 16 patches to 48 patches (T2(16) + ADC(16) + DWI(16))
                old_pos_embed = model.myViT_MM.pos_embedding.data 
                if old_pos_embed.shape[0] == 16:
                    old_h, old_w = 4, 4
                    new_h, new_w = 4, 12
                    embed_dim = old_pos_embed.shape[1]
                    
                    pos_reshaped = old_pos_embed.reshape(1, old_h, old_w, embed_dim).permute(0, 3, 1, 2)
                    pos_interpolated = F.interpolate(pos_reshaped, size=(new_h, new_w), 
                                                     mode='bicubic', align_corners=False)
                    new_pos_embed = pos_interpolated.permute(0, 2, 3, 1).reshape(new_h * new_w, embed_dim)
                    
                    model.myViT_MM.pos_embedding = new_pos_embed

                if self.t2_model_path and Path(self.t2_model_path).exists():
                    logger.info(f"Loading T2 weights into model.net_t2")
                    state_dict = torch.load(self.t2_model_path, map_location=self.device)
                    model.net_t2.load_state_dict(state_dict)

                if self.adc_model_path and Path(self.adc_model_path).exists():
                    logger.info(f"Loading ADC weights into model.net_adc.myCNN")
                    state_dict = torch.load(self.adc_model_path, map_location=self.device)
                    model.net_adc.myCNN.load_state_dict(state_dict)

                if self.vit_model_path and Path(self.vit_model_path).exists():
                    logger.info(f"Loading ViT Fusion weights into model.myViT_MM")
                    state_dict = torch.load(self.vit_model_path, map_location=self.device)
                    model.myViT_MM.load_state_dict(state_dict, strict=False)

                model = model.to(self.device)
                model.eval()
                self._model = model
                logger.info(f"MRI_PTCa model loaded successfully on {self.device}.")
            except Exception as e:
                logger.error(f"Failed to load MRI_PTCa model: {e}", exc_info=True)
                raise
        return self._model

    def load_and_preprocess_mri(self, mri_path: Path, mask_path: tp.Optional[Path] = None) -> torch.Tensor:
        """
        Load and preprocess a single MRI file (.mha).
        
        Args:
            mri_path: Path to MRI file
            mask_path: Optional path to mask file
        """
        if not mri_path.exists():
            raise FileNotFoundError(f"MRI file not found: {mri_path}")
        
        img = sitk.ReadImage(str(mri_path))
        data = sitk.GetArrayFromImage(img).astype(np.float32)

        # 1. Load and resize mask to match data shape
        if mask_path and mask_path.exists():
            mask_img = sitk.ReadImage(str(mask_path))
            mask_data = sitk.GetArrayFromImage(mask_img)
            
            if mask_data.shape != data.shape:
                mask_tensor = torch.from_numpy(mask_data).float().unsqueeze(0).unsqueeze(0)
                resized_mask_tensor = F.interpolate(mask_tensor, size=data.shape, mode='nearest')
                mask_data = resized_mask_tensor.squeeze().numpy()
            
            # 2. Find bounding box of mask region
            mask_binary = (mask_data > 0).astype(np.uint8)
            nonzero_indices = np.nonzero(mask_binary)
            
            if len(nonzero_indices[0]) > 0:
                z_min, z_max = nonzero_indices[0].min(), nonzero_indices[0].max() + 1
                y_min, y_max = nonzero_indices[1].min(), nonzero_indices[1].max() + 1
                x_min, x_max = nonzero_indices[2].min(), nonzero_indices[2].max() + 1
                
                # 3. Crop data and mask to bounding box
                data = data[z_min:z_max, y_min:y_max, x_min:x_max]
                mask_data = mask_data[z_min:z_max, y_min:y_max, x_min:x_max]
                
                # 4. Apply mask to cropped data
                data = data * (mask_data > 0)
            else:
                logger.warning(f"Empty mask for {mri_path}, using full image")

        # 5. Resize to target size
        tensor = torch.from_numpy(data).float().unsqueeze(0)  # (1, D, H, W)
        if tuple(tensor.shape[1:]) != self.img_size:
            tensor = F.interpolate(tensor.unsqueeze(0), size=self.img_size, mode='trilinear', align_corners=False).squeeze(0)
        
        data = tensor.squeeze(0).numpy() # (1, D, H, W) -> (D, H, W)
        
        # 6. RescaleIntensity to [-1, 1]
        data_min = data.min()
        data_max = data.max()
        if data_max - data_min > 1e-8:
            data = 2.0 * (data - data_min) / (data_max - data_min) - 1.0
        else:
            data = np.zeros_like(data)
        
        # 7. Z-score normalization
        mean = data.mean()
        std = data.std()
        if std > 1e-8:
            data = (data - mean) / std
        
        # 8. 3-channel replication
        tensor = torch.from_numpy(data).float()
        tensor = tensor.unsqueeze(1).repeat(1, 3, 1, 1) # (D, H, W) -> (D, 3, H, W)
        
        return tensor.unsqueeze(0) # (D, 3, H, W) -> (1, D, 3, H, W)

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

    def extract_all_layer_features(self, t2_tensor: torch.Tensor, adc_tensor: torch.Tensor) -> np.ndarray:
        """
        Extract all intermediate layer features from ViT Fusion model.
        
        Args:
            t2_tensor: Preprocessed T2W MRI tensor of shape (1, D, 3, H, W)
            adc_tensor: Preprocessed ADC MRI tensor of shape (1, D, 3, H, W)
        """
        _ = self.model  # Ensure model is loaded
        t2_tensor = t2_tensor.to(self.device)
        adc_tensor = adc_tensor.to(self.device)
        dwi_tensor = torch.zeros_like(adc_tensor)
        combined_input = torch.cat((t2_tensor, adc_tensor, dwi_tensor), dim=1)

        # Register hooks to capture intermediate layer outputs
        intermediate_features = []
        def hook_fn(module, input, output):
            """Hook function to capture transformer layer outputs."""
            intermediate_features.append(output.detach().cpu())

        hooks = []
        vit_transformer = self.model.myViT_MM.transformer
        for layer_block in vit_transformer.layers:
            hooks.append(layer_block[0].register_forward_hook(hook_fn))
            
        # Forward pass
        with torch.no_grad():
            _ = self.model(combined_input)

        # Remove hooks after forward pass
        for hook in hooks:
            hook.remove()
            
        if not intermediate_features:
            logger.warning("Could not extract intermediate features.")
            return np.array([])
            
        # Stack all layer outputs and extract CLS tokens
        layer_outputs = torch.stack(intermediate_features)  # (L, B, N, D)
        cls_tokens = layer_outputs[:, :, 0, :].permute(1, 0, 2)  # (B, L, D)
        sample_layers = cls_tokens[0].cpu().numpy()  # (L, D)
        return sample_layers

    def extract_features(self, t2_tensor: torch.Tensor, adc_tensor: torch.Tensor) -> np.ndarray:
        """
        Extract and aggregate features from intermediate layers of ViT Fusion model.
        
        Args:
            t2_tensor: Preprocessed T2-weighted MRI tensor of shape (1, D, 3, H, W)
            adc_tensor: Preprocessed ADC MRI tensor of shape (1, D, 3, H, W)
        """
        raw_features = self.extract_all_layer_features(t2_tensor, adc_tensor)
        if raw_features.size == 0:
            return np.array([])
        return self._aggregate_layers(raw_features)

    def extract_mri_raw_features(
        self, 
        mri_id: str, 
        raw_feature_dir: Union[str, Path],
        save_output: bool = True
    ) -> Path | None:
        """
        Extract and save raw layer features for a single MRI case.

        Args:
            mri_id: MRI identifier
            raw_feature_dir: Directory to save raw layer features
            save_output: Whether to save the extracted features to disk
        """
        raw_feature_dir = Path(raw_feature_dir)
        raw_feature_path = raw_feature_dir / f"{mri_id}_all_layers.npy"
        
        if save_output and raw_feature_path.exists():
            logger.info(f"Raw layer features already exist for {mri_id}. Skipping extraction.")
            return raw_feature_path
        
        mri_dir = Path(self.mri_dir)
        patient_id = mri_id.split("_")[0]
        t2_path = mri_dir / patient_id / f"{mri_id}_t2w.mha"
        adc_path = mri_dir / patient_id / f"{mri_id}_adc.mha"
        mask_path = mri_dir / patient_id / f"{mri_id}_mask.mha"
        if not mask_path.exists():
            mask_path = None
        
        t2_tensor = self.load_and_preprocess_mri(t2_path, mask_path)
        adc_tensor = self.load_and_preprocess_mri(adc_path, mask_path)
        
        raw_features = self.extract_all_layer_features(t2_tensor, adc_tensor)
        if raw_features.size == 0:
            logger.warning(f"Feature extraction failed for {mri_id}.")
            return None
            
        if save_output:
            raw_feature_dir.mkdir(parents=True, exist_ok=True)
            np.save(raw_feature_path, raw_features)
            logger.info(f"Saved raw layer features with shape {raw_features.shape} to {raw_feature_path}")
        
        return raw_feature_path

    def aggregate_mri_features(
        self, 
        mri_id: str, 
        layers: list[float], 
        raw_feature_dir: Union[str, Path],
        aggregated_feature_dir: Union[str, Path],
        save_output: bool = True
    ) -> None:
        """
        Aggregate raw layer features with specific layer configuration.

        Args:
            mri_id: MRI identifier 
            layers: List of layer indices as fractions (e.g., [0.5, 0.75, 1.0])
            raw_feature_dir: Directory containing raw layer features
            aggregated_feature_dir: Directory to save aggregated features
            save_output: Whether to save the aggregated features to disk
        """
        raw_feature_dir = Path(raw_feature_dir)
        raw_feature_path = raw_feature_dir / f"{mri_id}_all_layers.npy"
        
        if not raw_feature_path.exists():
            logger.warning(f"Raw layer features not found for {mri_id}. Run extract_mri_raw_features first.")
            return
        
        # Load raw layer features
        raw_features = np.load(raw_feature_path)  # (L, D)
        
        # Create output path
        aggregated_feature_dir = Path(aggregated_feature_dir)
        aggregated_feature_path = aggregated_feature_dir / f"{mri_id}_agg_layers.npy"
            
        if save_output and aggregated_feature_path.exists():
            logger.info(f"Aggregated features already exist for {mri_id} with layers {layers}. Skipping.")
            return
        
        # Aggregate features using group_mean
        aggregated_features = self._aggregate_layers(raw_features, layers=layers)
        
        if aggregated_features.size == 0:
            logger.warning(f"Feature aggregation failed for {mri_id}.")
            return
        
        if save_output:
            aggregated_feature_dir.mkdir(parents=True, exist_ok=True)
            np.save(aggregated_feature_path, aggregated_features)
            logger.info(f"Saved aggregated features with shape {aggregated_features.shape} to {aggregated_feature_path}")

    def extract_all_mris_raw_features(
        self, 
        raw_feature_dir: Union[str, Path],
        save_output: bool = True, 
        num_workers: int = 1
    ) -> None:
        """
        Extract raw layer features for all MRI files.
        
        Args:
            raw_feature_dir: Directory to save raw layer features
            save_output: Whether to save the extracted features to disk
            num_workers: Number of parallel workers for processing 
        """
        mri_dir = Path(self.mri_dir)
        mri_files = list(mri_dir.glob("*/*_t2w.mha"))
        logger.info(f"Found {len(mri_files)} T2W MRI files to extract raw features")
        
        mri_ids = []
        for mri_path in mri_files:
            mri_filename = mri_path.stem
            mri_id = "_".join(mri_filename.split("_")[:2])
            mri_ids.append(mri_id)
        
        if num_workers > 1:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                list(tqdm(
                    executor.map(
                        lambda mri_id: self._extract_mri_raw_features_wrapper(mri_id, raw_feature_dir, save_output),
                        mri_ids
                    ),
                    total=len(mri_ids),
                    desc="Extracting raw layer features"
                ))
        else:
            for mri_id in tqdm(mri_ids, desc="Extracting raw layer features"):
                try:
                    self.extract_mri_raw_features(mri_id, raw_feature_dir, save_output=save_output)
                except Exception as e:
                    logger.error(f"Failed to extract raw features for {mri_id}: {e}", exc_info=True)

    def _extract_mri_raw_features_wrapper(
        self, 
        mri_id: str, 
        raw_feature_dir: Union[str, Path], 
        save_output: bool
    ) -> None:
        """
        Wrapper for parallel processing of MRI raw feature extraction.
        
        Args:
            mri_id: MRI identifier
            raw_feature_dir: Directory to save raw layer features
            save_output: Whether to save the extracted features to disk
        """
        try:
            self.extract_mri_raw_features(mri_id, raw_feature_dir, save_output=save_output)
        except Exception as e:
            logger.error(f"Failed to extract raw features for {mri_id}: {e}", exc_info=True)

    def aggregate_all_mris_features(
        self,
        raw_feature_dir: Union[str, Path],
        aggregated_feature_dir: Union[str, Path],
        save_output: bool = True,
        num_workers: int = 4
    ) -> None:
        """
        Aggregate raw layer features for all MRIs with layer configuration [0.5, 0.75, 1.0].
        
        Args:
            raw_feature_dir: Directory containing raw layer features
            aggregated_feature_dir: Base directory for saving aggregated features
            save_output: Whether to save output files
            num_workers: Number of parallel workers for processing
        """
        layers = [0.5, 0.75, 1.0]
        
        raw_feature_dir = Path(raw_feature_dir)
        aggregated_feature_dir = Path(aggregated_feature_dir)
        
        mri_dir = Path(self.mri_dir)
        mri_files = list(mri_dir.glob("*/*_t2w.mha"))
        mri_ids = []
        for mri_path in mri_files:
            mri_filename = mri_path.stem
            mri_id = "_".join(mri_filename.split("_")[:2])
            mri_ids.append(mri_id)
        
        logger.info(f"Aggregating features for {len(mri_ids)} MRIs with layer config {layers}.")
        
        # Process in parallel
        if num_workers > 1:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                list(tqdm(
                    executor.map(
                        lambda mri_id: self.aggregate_mri_features(
                            mri_id, layers, raw_feature_dir, aggregated_feature_dir, save_output
                        ),
                        mri_ids
                    ),
                    total=len(mri_ids),
                    desc="Aggregating features"
                ))
        else:
            for mri_id in tqdm(mri_ids, desc="Aggregating features"):
                try:
                    self.aggregate_mri_features(
                        mri_id, layers, raw_feature_dir, aggregated_feature_dir, save_output=save_output
                    )
                except Exception as e:
                    logger.error(f"Failed to aggregate features for {mri_id}: {e}", exc_info=True)
                    continue

def main():
    """
    1. Extract raw layer features for all MRIs
    2. Aggregate raw features with layer configuration [0.5, 0.75, 1.0]
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract MRI features using MRI-PTCa")
    parser.add_argument("--mri_dir", type=str, required=True,
                        help="Directory containing MRI data files")
    parser.add_argument("--t2_model_path", type=str, default=None,
                        help="Path to T2 extractor model weights")
    parser.add_argument("--adc_model_path", type=str, default=None,
                        help="Path to ADC extractor model weights")
    parser.add_argument("--vit_model_path", type=str, default=None,
                        help="Path to ViT fusion model weights")
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
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    feature_extractor = MRI_FeatureExtractor(
        mri_dir=args.mri_dir,
        t2_model_path=args.t2_model_path,
        adc_model_path=args.adc_model_path,
        vit_model_path=args.vit_model_path,
        device=args.device
    )
    
    logger.info("Step 1: Extracting raw layer features for all MRIs...")
    feature_extractor.extract_all_mris_raw_features(
        raw_feature_dir=args.raw_feature_dir,
        save_output=True, 
        num_workers=args.num_workers_extract
    )
    
    logger.info("Step 2: Aggregating features with layer configuration [0.5, 0.75, 1.0]...")
    
    feature_extractor.aggregate_all_mris_features(
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