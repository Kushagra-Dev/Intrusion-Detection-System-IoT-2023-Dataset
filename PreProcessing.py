import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(file_path):
    print("Loading and preprocessing data...")
    df = pd.read_csv(file_path, low_memory=False)

    df.dropna(axis=1, how='all', inplace=True)
    df.dropna(axis=0, how='any', inplace=True)

    if 'label' not in df.columns:
        raise ValueError("No 'label' column found")

    df['label'] = df['label'].apply(lambda x: 0 if 'Benign' in str(x) else 1)

    non_numeric_cols = df.select_dtypes(include='object').columns
    df.drop(non_numeric_cols, axis=1, inplace=True)

    scaler = StandardScaler()
    X = scaler.fit_transform(df.drop('label', axis=1))
    y = df['label'].values

    print(f"Preprocessing complete. Shape: {X.shape}")
    return X, y
