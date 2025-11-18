"""
Dataset module for HIMF-Surv.
"""

import typing as tp
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)

class HIMFSurvDataset(Dataset):
    """Dataset class for loading multimodal features (WSI, MRI, clinical)."""
    def __init__(self, patient_ids: list, config: dict, labels_df: pd.DataFrame):
        """Initialize the dataset."""
        self.config = config
        self.labels_df = labels_df.set_index('patient_id')
        self.patient_ids = patient_ids
        
        logger.info(f"Pre-caching file paths for {len(self.patient_ids)} patients...")
        self.file_paths = {}
        for patient_id in self.patient_ids:
            wsi_files = list(Path(self.config['wsi_feature_dir']).glob(f"{patient_id}_*_agg_layers.npy"))
            mri_files = list(Path(self.config['mri_feature_dir']).glob(f"{patient_id}_*_agg_layers.npy"))
            clinical_file = Path(self.config['clinical_feature_dir']) / f"{patient_id}_embedding.npy"
            
            self.file_paths[patient_id] = {
                'wsi': wsi_files[0] if wsi_files else None,
                'mri': mri_files[0] if mri_files else None,
                'clinical': clinical_file if clinical_file.exists() else None
            }
        logger.info("File paths cached.")

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> tp.Dict[str, tp.Any]:
        """Get a single sample from the dataset."""
        patient_id = self.patient_ids[idx]
        features = {'patient_id': patient_id}
        
        file_paths = self.file_paths.get(patient_id, {})
        if file_paths.get('wsi'):
            features['wsi'] = np.load(file_paths['wsi'])
        if file_paths.get('mri'):
            features['mri'] = np.load(file_paths['mri'])
        if file_paths.get('clinical'):
            features['clinical'] = np.load(file_paths['clinical'])
        
        # Add labels from the dataframe
        label_info = self.labels_df.loc[int(patient_id)]
        features['time'] = float(label_info['time_to_follow-up/BCR'])
        features['event'] = float(label_info['BCR'])

        return features

def custom_collate_fn(batch: list) -> tp.Dict[str, tp.Any] | None:
    """Custom collate function with missing feature handling."""
    # Filter out samples with missing required features
    valid_batch = []
    for item in batch:
        if 'wsi' in item and 'mri' in item and 'clinical' in item:
            valid_batch.append(item)
        else:
            missing = []
            if 'wsi' not in item: missing.append('WSI')
            if 'mri' not in item: missing.append('MRI')
            if 'clinical' not in item: missing.append('Clinical')
            logger.warning(f"Skipping patient {item.get('patient_id', 'unknown')} "
                         f"in batch due to missing: {', '.join(missing)}")
    
    # If no valid samples, return None
    if not valid_batch:
        logger.error("Entire batch has missing features")
        return None
    
    collated = {
        'patient_id': [item['patient_id'] for item in valid_batch],
        'time': torch.tensor([item['time'] for item in valid_batch], dtype=torch.float32),
        'event': torch.tensor([item['event'] for item in valid_batch], dtype=torch.float32)
    }
    
    collated['wsi'] = [torch.from_numpy(item['wsi']) for item in valid_batch]
    collated['mri'] = [torch.from_numpy(item['mri']) for item in valid_batch]
    collated['clinical'] = torch.stack([torch.from_numpy(item['clinical']) for item in valid_batch])
    
    return collated
