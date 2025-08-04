# PreProcessing.py
import dask.dataframe as dd
import dask.array as da # <-- Import dask.array
import numpy as np
import pandas as pd
from dask_ml.preprocessing import StandardScaler
from dask.base import compute

# This version requires the dask-native data loader
from batch_processing import retdf

def preprocess_data(file_path: str) -> tuple[dd.DataFrame, dd.Series]:
    """
    Performs scalable preprocessing with Dask.
    """
    print("Starting data preprocessing with Dask...")
    
    df = retdf(file_path)

    label_map = df['label'].str.split().str[0].str.strip().map(
        {'Benign': 0, 'Malicious': 1}, 
        meta=pd.Series(dtype='int')
    )
    df['label'] = label_map.fillna(0).astype(int)
    
    y = df['label']
    X = df.drop(columns=['ts', 'uid', 'label', 'detailed-label'], errors='ignore')

    print(" - Engineering features...")
    
    # Create ratio features
    X['byte_ratio'] = X['orig_bytes'] / (X['resp_bytes'] + 1e-6)
    X['packet_ratio'] = X['orig_pkts'] / (X['resp_pkts'] + 1e-6)
    
    # One-Hot Encode the most common Connection States
    top_conn_states = ['SF', 'S0', 'REJ', 'RSTO', 'RSTR']
    for state in top_conn_states:
        X[f'conn_state_{state}'] = (X['conn_state'] == state).astype(int)

    # Manual Label Encoding -
    print("   - Performing label encoding for categorical features...")
    low_cardinality_categoricals = ['proto', 'service', 'history']
    X[low_cardinality_categoricals] = X[low_cardinality_categoricals].fillna('missing')

    # Compute the unique categories for each column in a single pass
    unique_categories_to_compute = {col: X[col].unique() for col in low_cardinality_categoricals}
    computed_uniques = compute(unique_categories_to_compute)[0]

    # Create a mapping for each column and apply it
    for col in low_cardinality_categoricals:
        category_map = {category: i for i, category in enumerate(computed_uniques[col])}
        X[col] = X[col].map(category_map, meta=(col, 'int')).astype(int)
        
    # Drop original and noisy columns
    cols_to_drop = [
        'id.orig_h', 'id.resp_h', 'conn_state',
        'orig_bytes', 'resp_bytes', 'orig_pkts', 'resp_pkts'
    ]
    X = X.drop(columns=cols_to_drop)
    
    print(" - Imputing missing values...")
    for col in X.columns:
        X[col] = dd.to_numeric(X[col], errors='coerce')
        mean_val = X[col].mean().compute()
        X[col] = X[col].fillna(mean_val)

    print(" - Scaling all features...")
    X = X.astype(np.float64)
    X_dask_array = X.to_dask_array(lengths=True)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_dask_array)
    
    # This replaces any NaN/inf values created by the scaler with 0.
    print(" - Stabilizing scaled data to prevent PCA errors...")
    X_scaled = da.nan_to_num(X_scaled)

    X_final = dd.from_dask_array(X_scaled, columns=X.columns)
    
    print("Preprocessing complete.")
    return X_final, y
