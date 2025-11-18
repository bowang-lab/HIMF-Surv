"""
Inference script for HIMF-Surv model.
"""

import argparse
import json
import logging
from pathlib import Path
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from model import HIMFSurvLightningModule
from dataset import HIMFSurvDataset, custom_collate_fn

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    return config

def load_model_from_checkpoint(checkpoint_path: str, config: dict) -> HIMFSurvLightningModule:
    """Load model from checkpoint."""
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    
    # Load model
    model = HIMFSurvLightningModule.load_from_checkpoint(
        checkpoint_path,
        wsi_feature_dim=config['wsi_feature_dim'],
        mri_feature_dim=config['mri_feature_dim'],
        clinical_feature_dim=config['clinical_feature_dim'],
        num_time_bins=config['num_time_bins'],
        learning_rate=config.get('learning_rate', 1e-4),
        weight_decay=config.get('weight_decay', 0.01),
        strict=False
    )
    
    model.eval()
    logger.info("Model loaded successfully")
    
    return model


def predict_batch(model: HIMFSurvLightningModule, patient_ids: list, config: dict,
                 batch_size: int = 8) -> list:
    """Predict survival for multiple patients."""
    # Create empty DataFrame for dataset (labels not needed for inference)
    empty_labels_df = pd.DataFrame(columns=['patient_id', 'time_to_follow-up/BCR', 'BCR', 'fold'])
    dataset = HIMFSurvDataset([str(pid) for pid in patient_ids], config, empty_labels_df)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                           collate_fn=custom_collate_fn, num_workers=2, pin_memory=False)
    
    results = []
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue
            
            # Move batch to device
            batch_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                           for k, v in batch.items() if k != 'patient_id'}
            batch_device['wsi'] = [w.to(device) for w in batch['wsi']]
            batch_device['mri'] = [m.to(device) for m in batch['mri']]
            
            # Forward pass
            logits = model.model(batch_device)  # (B, num_time_bins)
            
            # Compute survival probabilities and risk scores
            hazards = torch.sigmoid(logits)  # (B, num_time_bins)
            survival = torch.cumprod(1 - hazards, dim=1)  # (B, num_time_bins)
            expected_survival_time = torch.sum(survival, dim=1)  # (B,)
            predicted_risk_scores = -expected_survival_time  # (B,)
            
            # Process each patient in batch
            for i, patient_id in enumerate(batch['patient_id']):
                result = {
                    'patient_id': patient_id,
                    'risk': predicted_risk_scores[i].cpu().item(),
                    'expected_time': expected_survival_time[i].cpu().item(),
                    'survival_curve': survival[i].cpu().numpy().tolist(),
                    'hazards': hazards[i].cpu().numpy().tolist()
                }
                
                results.append(result)
    
    return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='HIMF-Surv Inference: Predict survival outcomes for prostate cancer patients'
    )
    parser.add_argument('--config', type=str, default='configs/inference_config.json',
                       help='Path to config JSON file')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    try:
        config = load_config(args.config)
        logger.info(f"Loaded config from: {args.config}")
    except FileNotFoundError as e:
        logger.error(str(e))
        return
    
    # Validate required config keys
    required_keys = ['checkpoint_path', 'wsi_feature_dir', 'mri_feature_dir', 
                     'clinical_feature_dir', 'wsi_feature_dim',
                     'mri_feature_dim', 'clinical_feature_dim', 'num_time_bins']
    for key in required_keys:
        if key not in config:
            logger.error(f"Required config key '{key}' not found in config file")
            return
    
    # Set device
    device_str = config.get('device', 'auto')
    if device_str == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = device_str
    logger.info(f"Using device: {device}")
    
    # Load checkpoint
    checkpoint_path = config['checkpoint_path']
    if not Path(checkpoint_path).exists():
        logger.error(f"Checkpoint file not found: {checkpoint_path}")
        return
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load model
    model = load_model_from_checkpoint(checkpoint_path, config)
    model = model.to(device)
    
    # Find patient IDs from feature directories
    wsi_dir = Path(config['wsi_feature_dir'])
    mri_dir = Path(config['mri_feature_dir'])
    clinical_dir = Path(config['clinical_feature_dir'])
    
    # Extract patient IDs from WSI files
    wsi_files = list(wsi_dir.glob("*_*_agg_layers.npy"))
    wsi_patient_ids = set()
    for f in wsi_files:
        patient_id = f.stem.split('_')[0]
        wsi_patient_ids.add(patient_id)
    
    # Extract patient IDs from MRI files
    mri_files = list(mri_dir.glob("*_*_agg_layers.npy"))
    mri_patient_ids = set()
    for f in mri_files:
        patient_id = f.stem.split('_')[0]
        mri_patient_ids.add(patient_id)
    
    # Extract patient IDs from clinical files
    clinical_files = list(clinical_dir.glob("*_embedding.npy"))
    clinical_patient_ids = set()
    for f in clinical_files:
        patient_id = f.stem.replace('_embedding', '')
        clinical_patient_ids.add(patient_id)

    patient_ids = sorted(list(wsi_patient_ids & mri_patient_ids & clinical_patient_ids))
    logger.info(f"Found {len(patient_ids)} patients with all three modalities")
    
    # Perform inference
    logger.info("Starting inference...")
    batch_size = config.get('batch_size', 8)
    results = predict_batch(model, patient_ids, config, batch_size)
    
    # Filter out None results
    results = [r for r in results if r is not None]
    logger.info(f"Inference complete. Predicted for {len(results)} patients")
    
    # Save results
    output_file = config.get('output_file', 'results/inference_results.json')
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Predictions saved to: {output_path}")

if __name__ == "__main__":
    main()

