"""
Clinical data preprocessing pipeline
"""

import json
import argparse
import logging
from pathlib import Path
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Feature order used for consistent embedding across all patients
FEATURE_ORDER = [
    'age_at_prostatectomy', 'primary_gleason', 'secondary_gleason',
    'tertiary_gleason', 'ISUP', 'pre_operative_PSA',
    'pT_stage_numeric',
    'earlier_therapy_none', 'earlier_therapy_cryo', 'earlier_therapy_hormones',
    'positive_lymph_nodes_0', 'positive_lymph_nodes_1', 'positive_lymph_nodes_x',
    'capsular_penetration_0', 'capsular_penetration_1', 'capsular_penetration_x',
    'lymphovascular_invasion_0', 'lymphovascular_invasion_1', 'lymphovascular_invasion_x',
    'positive_surgical_margins_0', 'positive_surgical_margins_1', 'positive_surgical_margins_x'
]

def preprocess_clinical_data(data: dict) -> np.ndarray:
    """
    Preprocess clinical data dictionary into a fixed-size embedding vector.

    Args:
        data: Dictionary containing clinical features
    """
    processed_item = data.copy()

    # Extract numeric part from pT_stage
    if 'pT_stage' in processed_item:
        pT_stage = str(processed_item['pT_stage'])
        numeric_part = ''.join(filter(str.isdigit, pT_stage))
        processed_item['pT_stage_numeric'] = float(numeric_part) if numeric_part else 0.0

    # One-hot encode earlier_therapy
    if 'earlier_therapy' in processed_item:
        therapy = processed_item['earlier_therapy']
        processed_item['earlier_therapy_none'] = 1.0 if therapy == 'none' else 0.0
        processed_item['earlier_therapy_cryo'] = 1.0 if therapy == 'radiotherapy + cryotherapy' else 0.0
        processed_item['earlier_therapy_hormones'] = 1.0 if therapy == 'radiotherapy + hormones' else 0.0

    # One-hot encode binary categorical features
    one_hot_features = [
        'positive_lymph_nodes',
        'capsular_penetration',
        'lymphovascular_invasion',
        'positive_surgical_margins'
    ]

    for key in one_hot_features:
        if key in processed_item:
            value = str(processed_item[key]).lower().strip()
            
            is_0 = value in ['0', '0.0']
            is_1 = value in ['1', '1.0']
            
            processed_item[f'{key}_0'] = 1.0 if is_0 else 0.0
            processed_item[f'{key}_1'] = 1.0 if is_1 else 0.0
            processed_item[f'{key}_x'] = 1.0 if not is_0 and not is_1 else 0.0

    # Create embedding vector following FEATURE_ORDER
    embedding = []
    for feature_name in FEATURE_ORDER:
        value = processed_item.get(feature_name, 0.0)
        try:
            embedding.append(float(value))
        except (ValueError, TypeError):
            embedding.append(0.0)

    return np.array(embedding, dtype=np.float32)

def create_embeddings_from_json_files(input_dir: str, output_dir: str) -> None:
    """
    Process all JSON files in input directory and create embedding files.
    
    Args:
        input_dir: Directory containing JSON files with clinical data
        output_dir: Directory to save embedding .npy files
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    json_files = list(input_path.glob('*.json'))
    
    if not json_files:
        logger.error(f"No JSON files found in '{input_dir}'.")
        return

    logger.info(f"Found {len(json_files)} JSON files. Starting preprocessing...")

    for file_path in tqdm(json_files, desc="Processing clinical data"):
        with open(file_path, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Could not decode JSON from {file_path}. Skipping.")
                continue

        embedding_vector = preprocess_clinical_data(data)
        
        output_file = output_path / f"{file_path.stem}_embedding.npy"
        np.save(output_file, embedding_vector)

    logger.info(f"Successfully created {len(json_files)} clinical embedding files in '{output_dir}'.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess clinical data from JSON files")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing JSON files with clinical data")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save clinical embedding .npy files")
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    create_embeddings_from_json_files(args.input_dir, args.output_dir)