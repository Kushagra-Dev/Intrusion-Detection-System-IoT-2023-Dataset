import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask_ml.preprocessing import StandardScaler
from sklearn.feature_extraction import FeatureHasher
from dask.base import compute

from batch_processing import retdf

def preprocess_data(file_path: str) -> tuple[dd.DataFrame, dd.Series]:
    """
    Performs preprocessing using Dask
    """
    print(" Starting data preprocessing with Dask...")
    
    df = retdf(file_path)

    label_map = df['label'].str.split().str[0].str.strip().map(
        {'Benign': 0, 'Malicious': 1}, 
        meta=pd.Series(dtype='int')
    )
    df['label'] = label_map.fillna(0).astype(int)
    
    y = df['label']
    X = df.drop(columns=['ts', 'uid', 'label', 'detailed-label'], errors='ignore')

    categorical_features_list = [
        'id.orig_h', 'id.resp_h', 'proto', 'service', 
        'conn_state', 'local_orig', 'local_resp', 'history', 'tunnel_parents'
    ]
    categorical_features_list = [col for col in categorical_features_list if col in X.columns]
    numeric_features = [col for col in X.columns if col not in categorical_features_list]

    print(" - Using Feature Hashing for all categorical features...")
    X[categorical_features_list] = X[categorical_features_list].fillna('missing')
    
    hasher = FeatureHasher(n_features=10, input_type='dict')

    def hash_partition(partition):
        records = partition.to_dict('records')
        hashed_features = hasher.transform(records)
        return pd.DataFrame(
            hashed_features.toarray(), #type: ignore
            columns=[f'hashed_{i}' for i in range(10)],
            index=partition.index
        )
    
    meta_df = pd.DataFrame(columns=[f'hashed_{i}' for i in range(10)], dtype=np.float64)
    hashed_features_dd = X[categorical_features_list].map_partitions(hash_partition, meta=meta_df)
    
    X = dd.concat([X[numeric_features], hashed_features_dd], axis=1)
    all_feature_columns = numeric_features + list(meta_df.columns)
    
    print(" - Calculating all means in parallel...")

    for col in all_feature_columns:
        X[col] = dd.to_numeric(X[col], errors='coerce')
        
    means_to_compute = {col: X[col].mean() for col in all_feature_columns}
    
    computed_means = compute(means_to_compute)[0]
    
    print(" - Applying means to fill missing values...")
    X = X.fillna(computed_means)

    print(" - Scaling all features...")
    X_dask_array = X.to_dask_array(lengths=True)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_dask_array)

    X_final = dd.from_dask_array(X_scaled, columns=all_feature_columns)
    
    print(" Preprocessing complete.")
    return X_final, y
