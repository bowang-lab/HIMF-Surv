# Figures Generation Scripts

This directory contains scripts for generating various visualization figures for the HIMF-Surv model.

## Scripts

### 1. `generate_km_stratification.py`

Generates Kaplan-Meier survival curves with a fixed quantile threshold (q=0.2) to split patients into high-risk and low-risk groups. Performs Out-of-Fold prediction by using each fold's checkpoint to predict on its test set. The script splits patients per fold using fold-specific quantile thresholds, then combines high-risk groups and low-risk groups separately across all folds.

**Usage:**

```bash
python generate_km_stratification.py \
    --ckpt-dir /path/to/checkpoints \
    --labels-file /path/to/labels.csv \
    --wsi-feature-dir /path/to/wsi/aggregated/features \
    --mri-feature-dir /path/to/mri/aggregated/features \
    --clinical-feature-dir /path/to/clinical/embeddings \
    --output-dir ./output \
    [--q 0.2]
```

**Arguments:**
- `--ckpt-dir` (required): Directory containing checkpoint files (will search for files with `fold_*` or `fold-*` in filename)
- `--labels-file` (required): Path to labels CSV file (must contain the following columns: `fold`, `patient_id`, `BCR`, `time_to_follow-up/BCR`)
- `--wsi-feature-dir` (required): Directory containing WSI aggregated feature files
- `--mri-feature-dir` (required): Directory containing MRI aggregated feature files
- `--clinical-feature-dir` (required): Directory containing clinical embedding feature files
- `--output-dir` (default: `.`): Output directory for generated figures
- `--q` (default: 0.2): Quantile threshold for risk group splitting (proportion of high-risk group)

**Output:**
- `km_stratification.png`: KM survival curve plot with high-risk and low-risk groups

---

### 2. `visualize_abmil_attention.py`

Visualizes ABMIL attention scores overlaid on WSI thumbnails and extracts top-k high/low attention patches. This script only requires WSI features (MRI and clinical features are not needed for ABMIL attention visualization).

**Usage:**

```bash
python visualize_abmil_attention.py \
    --ckpt /path/to/checkpoint.ckpt \
    --patient-id <patient_id> \
    --wsi-raw-dir /path/to/raw/wsi/files \
    --wsi-feature-dir /path/to/wsi/aggregated/features \
    --output-dir ./output \
    [--top-k 5]
```

**Arguments:**
- `--ckpt` (required): Path to model checkpoint file
- `--patient-id` (required): Patient ID to visualize
- `--wsi-raw-dir` (required): Directory containing raw WSI files (expected structure: `{wsi-raw-dir}/{patient_id}/{patient_id}_1.tif`)
- `--wsi-feature-dir` (required): Directory containing WSI aggregated feature files (expected pattern: `{patient_id}_*_agg_layers.npy`)
- `--output-dir` (default: `.`): Output directory for generated figures
- `--top-k` (default: 5): Number of top/bottom patches to extract

**Output:**
- `{patient_id}/abmil/heatmap.png`: WSI heatmap with attention scores overlaid
- `{patient_id}/abmil/high_patch_{rank:02d}.png`: Individual high attention patches
- `{patient_id}/abmil/low_patch_{rank:02d}.png`: Individual low attention patches

---

### 3. `visualize_uni_attention.py`

Visualizes internal ViT attention within the top-1 ABMIL attention patch using the UNI model. First extracts the top-1 patch using ABMIL attention, then visualizes UNI's internal attention at different transformer layers.

**Usage:**
```bash
python visualize_uni_attention.py \
    --ckpt /path/to/checkpoint.ckpt \
    --patient-id <patient_id> \
    --wsi-raw-dir /path/to/raw/wsi/files \
    --wsi-feature-dir /path/to/wsi/aggregated/features \
    --output-dir ./output
```

**Arguments:**
- `--ckpt` (required): Path to model checkpoint file (used to extract top-1 patch via ABMIL attention)
- `--patient-id` (required): Patient ID to visualize
- `--wsi-raw-dir` (required): Directory containing raw WSI files (expected structure: `{wsi-raw-dir}/{patient_id}/{patient_id}_1.tif`)
- `--wsi-feature-dir` (required): Directory containing WSI aggregated feature files (expected pattern: `{patient_id}_*_agg_layers.npy`)
- `--output-dir`: Output directory for generated figures

**Output:**
- `{patient_id}/wsi/uni_attention_{layer_name}.png`: Individual UNI attention heatmaps for each layer (midpoint, three_quarter, final)

---

### 4. `visualize_mri_attention.py`

Visualizes internal ViT attention within MRI slices using the MIMS-ViT model. Extracts attention from the MIMS-ViT transformer layers and overlays them on T2W and ADC MRI slices.

**Usage:**
```bash
python visualize_mri_attention.py \
    --patient-id <patient_id> \
    --mri-raw-dir /path/to/raw/mri/files \
    --output-dir ./output
```

**Arguments:**
- `--patient-id` (required): Patient ID to visualize
- `--mri-raw-dir` (required): Directory containing raw MRI files (expected structure: `{mri-raw-dir}/{patient_id}/{patient_id}_0001_{modality}.mha` where modality is `t2w`, `adc`, or `mask`)
- `--output-dir` (default: `.`): Output directory for generated figures

**Output:**
- `{patient_id}/mri/{modality}_{layer_name}.png`: Individual MRI attention heatmaps for each modality (t2w, adc) and layer (midpoint, three_quarter, final)