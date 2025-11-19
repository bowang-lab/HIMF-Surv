import logging
import typing as tp
import torch
from torch import nn, Tensor
from torch.nn import functional as F
import pytorch_lightning as pl
from torch.optim import AdamW
from x_transformers import Encoder

logger = logging.getLogger(__name__)

def concordance_index(event_times, predicted_scores, event_observed):
    """Calculate the concordance index (C-index) for survival analysis."""
    if isinstance(event_times, torch.Tensor):
        event_times = event_times.detach().cpu().numpy()
    if isinstance(predicted_scores, torch.Tensor):
        predicted_scores = predicted_scores.detach().cpu().numpy()
    if isinstance(event_observed, torch.Tensor):
        event_observed = event_observed.detach().cpu().numpy()
    
    n = len(event_times)
    concordant = 0
    permissible = 0
    
    for i in range(n):
        if event_observed[i] == 0: # censored
            continue
        for j in range(n):
            if i == j:
                continue
            if event_times[i] < event_times[j]:
                permissible += 1
                if predicted_scores[i] > predicted_scores[j]:
                    concordant += 1
                elif predicted_scores[i] == predicted_scores[j]:
                    concordant += 0.5
    
    if permissible == 0:
        return 0.5
    return concordant / permissible

def discretize_time(time: Tensor, num_bins: int, max_time: float, device: str):
    """Discretizes continuous time into bins using a fixed max_time."""
    time = torch.as_tensor(time, dtype=torch.float32, device=device)
    bins = torch.linspace(0, max_time, num_bins + 1, device=device)
    discretized = torch.bucketize(time, bins, right=True) - 1
    discretized = torch.clamp(discretized, 0, num_bins - 1)
    return discretized

class NLLLoss(nn.Module):
    """Negative Log-Likelihood loss."""
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits, y_time, y_event):
        y_time = y_time.type(torch.int64).unsqueeze(1)
        y_event = y_event.type(torch.int64).unsqueeze(1)
        num_bins = logits.shape[1]
        y_time = torch.clamp(y_time, 0, num_bins - 1)
        
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        S_padded = torch.cat([torch.ones_like(y_event, dtype=torch.float), S], 1)
        
        s_prev = torch.gather(S_padded, 1, y_time).clamp(min=1e-7)
        h_this = torch.gather(hazards, 1, y_time).clamp(min=1e-7)
        
        log_lik_uncensored = torch.log(s_prev) + torch.log(h_this)
        y_time_next = torch.clamp(y_time + 1, 0, num_bins)
        log_lik_censored = torch.log(torch.gather(S_padded, 1, y_time_next).clamp(min=1e-7))
        
        neg_log_lik = - (y_event * log_lik_uncensored + (1 - y_event) * log_lik_censored)
        
        if self.reduction == 'mean':
            return neg_log_lik.mean()
        return neg_log_lik.sum()

class ABMIL(nn.Module):
    """ABMIL module."""
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        A = self.attention(x)
        A = F.softmax(A, dim=0)
        bag_feature = torch.sum(A * x, dim=0)
        return bag_feature

class MLPPredictionHead(nn.Module):
    """MLP prediction head."""
    def __init__(
        self,
        input_dim: int,
        num_time_bins: int = 15,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_time_bins = num_time_bins
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_time_bins)
        
    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        logits = self.fc3(x)
        return logits

class HIMFSurv(nn.Module):
    """HIMF-Surv model."""
    def __init__(self,
                 wsi_feature_dim: int = 2048,
                 mri_feature_dim: int = 4096,  
                 clinical_feature_dim: int = 22,
                 num_time_bins: int = 15,
                 **kwargs):
        super().__init__()
        
        self.num_time_bins = num_time_bins

        self.abmil_wsi = ABMIL(
            input_dim=wsi_feature_dim, 
            hidden_dim=128
        )
        
        projectors_dict = {}
        projectors_dict['wsi'] = nn.Sequential(
            nn.Linear(wsi_feature_dim, 1536),
            nn.Dropout(0.1)
        )
        projectors_dict['mri'] = nn.Sequential(
            nn.Linear(mri_feature_dim, 1536),
            nn.Dropout(0.1)
        )
        projectors_dict['clinical'] = nn.Sequential(
            nn.Linear(clinical_feature_dim, 1536),
            nn.Dropout(0.1)
        )
        
        self.projectors = nn.ModuleDict(projectors_dict)

        self.transformer = Encoder(
            dim=1536,
            depth=18,
            heads=12,
            attn_dropout=0.2,
            ff_dropout=0.2
        )
        
        self.prediction_head = MLPPredictionHead(
            input_dim=1536,
            num_time_bins=num_time_bins,
            hidden_dim=64,
        )

    def forward(self, batch: tp.Dict[str, tp.Any]) -> Tensor:
        projected_tokens = {}
        device = next(self.parameters()).device
        
        # Process WSI
        if 'wsi' in batch:
            wsi_patient_features = []
            for wsi_feat in batch['wsi']:
                wsi_feat = wsi_feat.to(device).float()  # (N, G, D)
                N = wsi_feat.shape[0]
                wsi_flat = wsi_feat.reshape(N, -1)  # (N, G*D)
                patient_feature = self.abmil_wsi(wsi_flat)  # (G*D,)
                wsi_patient_features.append(patient_feature)
            wsi_tensor = torch.stack(wsi_patient_features)  # (B, G*D)
            projected_tokens['wsi'] = self.projectors['wsi'](wsi_tensor)

        # Process MRI
        if 'mri' in batch:
            mri_features = []
            for mri_feat in batch['mri']:
                mri_feat = mri_feat.to(device).float()  # (G, D)
                mri_flat = mri_feat.reshape(-1)  # (G*D,)
                mri_features.append(mri_flat)
            mri_tensor = torch.stack(mri_features)  # (B, G*D)
            projected_tokens['mri'] = self.projectors['mri'](mri_tensor) 
        
        # Process Clinical
        if 'clinical' in batch:
            clinical_tensor = batch['clinical'].to(device).float()  # (B, D)
            projected_tokens['clinical'] = self.projectors['clinical'](clinical_tensor)

        if not projected_tokens:
            raise ValueError("No features found in the batch.")

        # Create sequence in fixed order: wsi, mri, clinical
        modality_order = ['wsi', 'mri', 'clinical']
        final_sequence_tokens = [projected_tokens[m] for m in modality_order if m in projected_tokens]
        sequence_tensor = torch.stack(final_sequence_tokens, dim=1)
        
        # Pass through transformer
        encoded_sequence = self.transformer(sequence_tensor)
        
        # Mean pooling over modalities
        fused_embedding = torch.mean(encoded_sequence, dim=1)
        
        # Pass through prediction head
        logits = self.prediction_head(fused_embedding)  # (B, num_time_bins)
        
        return logits

class HIMFSurvLightningModule(pl.LightningModule):
    """HIMF-Surv Lightning module."""
    def __init__(self, 
                 wsi_feature_dim: int = 2048,
                 mri_feature_dim: int = 4096,
                 clinical_feature_dim: int = 22,
                 num_time_bins: int = 15,
                 max_time: float = 110.0,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 0.01):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = HIMFSurv(
            wsi_feature_dim=wsi_feature_dim,
            mri_feature_dim=mri_feature_dim,
            clinical_feature_dim=clinical_feature_dim,
            num_time_bins=num_time_bins
        )
        
        self.loss_fn = NLLLoss(reduction='mean')

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_time_bins = num_time_bins
        self.max_time = max_time
        
        self.training_step_outputs = []
        self.validation_step_outputs = []

    def _process_batch(self, batch):
        y_time_continuous, y_event = batch['time'], batch['event']
        y_time_discrete = discretize_time(y_time_continuous, self.num_time_bins, self.max_time, self.device)
        
        logits = self.model(batch)
        loss = self.loss_fn(logits, y_time_discrete, y_event)
        
        hazards = torch.sigmoid(logits)
        survival = torch.cumprod(1 - hazards, dim=1)
        expected_survival_time = torch.sum(survival, dim=1)
        predicted_risk_score = -expected_survival_time

        return loss, predicted_risk_score, y_time_continuous, y_event

    def training_step(self, batch, batch_idx):
        loss, predicted_risk_score, y_time, y_event = self._process_batch(batch)
        
        output = {'loss': loss, 'predicted_risk_score': predicted_risk_score, 'y_time': y_time, 'y_event': y_event}
        self.training_step_outputs.append(output)
        self.log('train_loss_step', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return loss

    def on_train_epoch_end(self):
        if not self.training_step_outputs:
            return
        
        avg_train_loss = torch.stack([x['loss'] for x in self.training_step_outputs]).mean()
        self.log('train_loss_epoch', avg_train_loss, on_epoch=True, prog_bar=True, logger=True)
        
        predicted_risk_scores = torch.cat([x['predicted_risk_score'] for x in self.training_step_outputs])
        y_times = torch.cat([x['y_time'] for x in self.training_step_outputs])
        y_events = torch.cat([x['y_event'] for x in self.training_step_outputs])
        
        c_index = concordance_index(y_times, predicted_risk_scores, y_events)
        self.log('train_c_index_epoch', c_index, on_epoch=True, prog_bar=True, logger=True)
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        loss, predicted_risk_score, y_time, y_event = self._process_batch(batch)
        output = {'val_loss': loss, 'predicted_risk_score': predicted_risk_score, 'y_time': y_time, 'y_event': y_event}
        self.validation_step_outputs.append(output)

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return
        
        avg_val_loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean()
        self.log('val_loss_epoch', avg_val_loss, on_epoch=True, prog_bar=True, logger=True)

        predicted_risk_scores = torch.cat([x['predicted_risk_score'] for x in self.validation_step_outputs])
        y_times = torch.cat([x['y_time'] for x in self.validation_step_outputs])
        y_events = torch.cat([x['y_event'] for x in self.validation_step_outputs])

        c_index = concordance_index(y_times, predicted_risk_scores, y_events)
        self.log('val_c_index_epoch', c_index, on_epoch=True, prog_bar=True, logger=True)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }