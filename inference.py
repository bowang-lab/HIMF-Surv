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

from model import HIMFSurvLightningModule, concordance_index
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
                 labels_df: pd.DataFrame, batch_size: int = 8) -> list:
    """Predict survival for multiple patients."""
    dataset = HIMFSurvDataset([str(pid) for pid in patient_ids], config, labels_df)
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
                     'clinical_feature_dir', 'labels_file', 'wsi_feature_dim',
                     'mri_feature_dim', 'clinical_feature_dim', 'num_time_bins']
    for key in required_keys:
        if key not in config:
            logger.error(f"Required config key '{key}' not found in config file")
            return
    
    # Load labels file
    labels_file = config['labels_file']
    if not Path(labels_file).exists():
        logger.error(f"Labels file not found: {labels_file}")
        return
    labels_df = pd.read_csv(labels_file)
    logger.info(f"Loaded labels from: {labels_file} ({len(labels_df)} patients)")
    
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
    
    # Get patient IDs from labels file
    patient_ids = sorted([str(pid) for pid in labels_df['patient_id'].unique()])
    logger.info(f"Found {len(patient_ids)} patients in labels file")
    
    # Perform inference
    logger.info("Starting inference...")
    batch_size = config.get('batch_size', 8)
    results = predict_batch(model, patient_ids, config, labels_df, batch_size)
    
    # Filter out None results
    results = [r for r in results if r is not None]
    logger.info(f"Inference complete. Predicted for {len(results)} patients")
    
    # Calculate C-index if ground truth labels are available
    c_index_info = None
    if len(results) > 0 and 'time_to_follow-up/BCR' in labels_df.columns and 'BCR' in labels_df.columns:
        # Create a mapping from patient_id to ground truth
        labels_df_indexed = labels_df.set_index('patient_id')
        
        # Extract ground truth and predicted scores
        event_times = []
        event_observed = []
        predicted_scores = []
        
        for result in results:
            patient_id = int(result['patient_id'])
            if patient_id in labels_df_indexed.index:
                label_info = labels_df_indexed.loc[patient_id]
                event_times.append(float(label_info['time_to_follow-up/BCR']))
                event_observed.append(float(label_info['BCR']))
                predicted_scores.append(result['risk'])
        
        if len(event_times) > 0 and sum(event_observed) > 0:  # Need at least one event
            c_index = concordance_index(
                event_times=torch.tensor(event_times),
                predicted_scores=torch.tensor(predicted_scores),
                event_observed=torch.tensor(event_observed)
            )
            logger.info(f"C-index: {c_index:.4f} (calculated on {len(event_times)} patients, {int(sum(event_observed))} events)")
            c_index_info = {
                'c_index': float(c_index),
                'num_patients': len(event_times),
                'num_events': int(sum(event_observed))
            }
        else:
            logger.warning("Cannot calculate C-index: insufficient events or no matching patients")
    else:
        logger.info("C-index not calculated: ground truth labels not available or no results")
    
    # Save results
    output_file = config.get('output_file', 'results/inference_results.json')
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = results.copy()
    if c_index_info is not None:
        output_data.insert(0, c_index_info)
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"Predictions saved to: {output_path}")

if __name__ == "__main__":
    main()

