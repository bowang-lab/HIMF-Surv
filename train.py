"""
Main training script for HIMF-Surv.
"""
import json
import numpy as np
import argparse
import wandb
import logging
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint

from model import HIMFSurvLightningModule
from dataset import HIMFSurvDataset, custom_collate_fn


class WandbFoldCallback(Callback):
    """Custom callback to log metrics to wandb with fold prefix."""
    def __init__(self, fold_idx):
        super().__init__()
        self.fold_idx = fold_idx
        self.fold_prefix = f"fold{fold_idx + 1}"
        
    def on_train_epoch_end(self, trainer, pl_module):
        metrics = {}
        
        train_loss = trainer.callback_metrics.get('train_loss_epoch', None)
        
        if train_loss is not None:
            metrics[f"{self.fold_prefix}/train_loss"] = float(train_loss)
        
        if metrics:
            wandb.log(metrics, step=trainer.current_epoch)
    
    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = {}
        
        val_loss = trainer.callback_metrics.get('val_loss_epoch', None)
        val_c_index = trainer.callback_metrics.get('val_c_index_epoch', None)
        
        if val_loss is not None:
            metrics[f"{self.fold_prefix}/val_loss"] = float(val_loss)
        if val_c_index is not None:
            metrics[f"{self.fold_prefix}/val_c_index"] = float(val_c_index)
        
        if metrics:
            wandb.log(metrics, step=trainer.current_epoch)

def parse_args():
    """Parse command line arguments for HIMF-Surv training."""
    parser = argparse.ArgumentParser(
        description='HIMF-Surv: Hierarchical Intra- and Inter-Modality Fusion for Multimodal Survival Prediction in Prostate Cancer'
    )
    parser.add_argument('--config', type=str, default='configs/train_config.json',
                        help='Path to config JSON file')
    return parser.parse_args()

def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    return config

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    # Load config from file
    try:
        config = load_config(args.config)
        logger.info(f"Loaded config from: {args.config}")
    except FileNotFoundError as e:
        logger.error(f"Config file not found: {e}")
        return
    
    # Validate required config keys
    required_keys = ['wsi_feature_dir', 'mri_feature_dir', 'clinical_feature_dir', 'labels_file']
    for key in required_keys:
        if key not in config:
            logger.error(f"Required config key '{key}' not found in config file")
            return
    
    # Load data with folds
    try:
        all_patients_df = pd.read_csv(config['labels_file'])
    except FileNotFoundError:
        logger.error(f"Labels file not found: {config['labels_file']}.")
        return

    # Store all fold results
    all_results = []
    
    num_folds = all_patients_df['fold'].nunique()
    # Specify which folds to run (None = all folds, or list of fold indices like [0, 1])
    folds_to_run = None 
    if folds_to_run is None:
        folds_to_run = list(range(num_folds))
        logger.info(f"Running all {num_folds} folds based on BCR stratification")
    else:
        logger.info(f"Running specific folds: {[f+1 for f in folds_to_run]} (0-based indices: {folds_to_run})")
    
    # Initialize wandb
    wandb_project = config.get("wandb_project", "HIMF-Surv")
    run_name = config.get("wandb_run_name", "initial run")
    wandb.init(
        project=wandb_project,
        name=run_name,
        config=config,
        reinit=False
    )
    logger.info(f"Initialized wandb run: {wandb.run.name}")
    
    # Create checkpoint directory
    Path(config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    
    for fold_idx in folds_to_run:
        logger.info(f"--- Starting Fold {fold_idx + 1}/{num_folds} ---")
        
        train_df = all_patients_df[all_patients_df['fold'] != fold_idx]
        val_df = all_patients_df[all_patients_df['fold'] == fold_idx]
        
        result = run_fold(fold_idx, train_df, val_df, config, all_patients_df, logger)
        all_results.append(result)
    
    logger.info(f"{num_folds}-fold cross-validation complete.")
    
    output_file = config.get('output_file', 'results/train_results.json')
    save_results(all_results, config, logger, output_file)

def save_results(all_results, config, logger, output_file_path: str = 'results/train_results.json'):
    """Calculate statistics and save results to JSON."""
    val_c_indices = [r['best_val_c_index'] for r in all_results if r['best_val_c_index'] is not None]
    
    def calculate_stats(values):
        """Calculate statistics for a list of values."""
        if not values:
            return {}
        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
        }
    
    results = {
        'config': config,
        'num_runs': len(all_results),
        'val_c_index': calculate_stats(val_c_indices),
        'individual_results': all_results
    }
    
    output_file = Path(output_file_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info("Cross-Validation Results Summary")
    logger.info(f"{'='*60}")
    logger.info(f"Total runs: {len(all_results)}")
    logger.info(f"\nValidation C-Index:")
    for key, value in results['val_c_index'].items():
        logger.info(f"  {key}: {value:.4f}")
    
    logger.info(f"\nCheckpoint Information:")
    for result in all_results:
        if result['checkpoint_path']:
            logger.info(f"  Fold {result['fold'] + 1}: {result['checkpoint_path']}")
            if result['best_val_c_index'] is not None:
                logger.info(f"    Best Val C-Index: {result['best_val_c_index']:.4f}")
    
    logger.info(f"\nResults saved to: {output_file}")
    logger.info(f"{'='*60}")
    
    wandb.summary['val_c_index_mean'] = results['val_c_index'].get('mean', 0)
    wandb.summary['val_c_index_std'] = results['val_c_index'].get('std', 0)
    
    wandb.finish()
    logger.info("Wandb run finished.")

def run_fold(fold_idx: int, train_df, val_df, config, all_patients_df, logger):
    """Run training for a single fold."""
    train_ids = [str(pid) for pid in train_df['patient_id'].tolist()]
    val_ids = [str(pid) for pid in val_df['patient_id'].tolist()]

    max_time = float(train_df['time_to_follow-up/BCR'].max())
    logger.info(f"Using max_time={max_time:.2f} months for time binning")

    # Create datasets
    train_dataset = HIMFSurvDataset(train_ids, config, all_patients_df)
    val_dataset = HIMFSurvDataset(val_ids, config, all_patients_df)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, 
                              collate_fn=custom_collate_fn, num_workers=2, pin_memory=False,
                              persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, 
                           collate_fn=custom_collate_fn, num_workers=2, pin_memory=False,
                           persistent_workers=True)
    
    # Initialize model and Lightning module
    lit_model = HIMFSurvLightningModule(
        wsi_feature_dim=config['wsi_feature_dim'],
        mri_feature_dim=config['mri_feature_dim'],
        clinical_feature_dim=config['clinical_feature_dim'],
        num_time_bins=config['num_time_bins'],
        max_time=max_time,
        learning_rate=config.get('learning_rate', 1e-4),
        weight_decay=config.get('weight_decay', 0.01)
    )
    
    # Setup callbacks
    wandb_callback = WandbFoldCallback(fold_idx=fold_idx)
    
    # Model checkpoint based on validation C-index
    checkpoint_callback = ModelCheckpoint(
        dirpath=config['checkpoint_dir'],
        filename=f'fold_{fold_idx+1}_best_val_cindex',
        monitor='val_c_index_epoch',
        mode='max',
        save_top_k=1,
        save_last=False,
        verbose=True,
        save_weights_only=False,
        auto_insert_metric_name=False
    )
    
    # Early stopping based on validation loss
    early_stopping_callback = EarlyStopping(
        monitor='val_loss_epoch',
        patience=config['early_stopping_patience'],
        min_delta=config['early_stopping_min_delta'],
        mode='min',
        verbose=True
    )

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=config['epochs'],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[wandb_callback, checkpoint_callback, early_stopping_callback],
        enable_progress_bar=True,
        logger=False,
        enable_checkpointing=True
    )
    trainer.fit(lit_model, train_loader, val_loader)
    
    # Get best validation C-index from checkpoint callback
    best_val_c_index = None
    checkpoint_path = None
    
    if checkpoint_callback.best_model_path:
        checkpoint_path = checkpoint_callback.best_model_path
        logger.info(f"Fold {fold_idx + 1} best checkpoint saved to: {checkpoint_path}")
        best_val_c_index = checkpoint_callback.best_model_score
        if best_val_c_index is not None:
            best_val_c_index = best_val_c_index.item()
    
    if best_val_c_index is not None:
        logger.info(f"Fold {fold_idx + 1} Best Val C-Index: {best_val_c_index:.4f}")
    else:
        logger.warning(f"Fold {fold_idx + 1} No best validation C-index found")
    
    # Return results
    return {
        'fold': fold_idx,
        'best_val_c_index': best_val_c_index,
        'checkpoint_path': checkpoint_path,
        'train_size': len(train_ids),
        'val_size': len(val_ids)
    }

if __name__ == "__main__":
    main()

