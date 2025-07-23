import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(csv_path):
    print("Reading CSV...")
    df = pd.read_csv(csv_path, low_memory=False)
    print("Initial shape:", df.shape)

    # Dropping columns with mostly missing values
    df.dropna(axis=1, thresh=int(0.5 * len(df)), inplace=True)

    df.fillna(0, inplace=True)

    label_col = 'label'
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in the dataset.")

    df[label_col] = df[label_col].astype(str).str.lower().str.strip()
    df[label_col] = df[label_col].apply(lambda x: 0 if x == 'benign' else 1)

    non_numeric_cols = df.select_dtypes(include=['object']).columns.drop(label_col, errors='ignore')
    for col in non_numeric_cols:
        try:
            df[col] = LabelEncoder().fit_transform(df[col])
        except Exception as e:
            print(f"Could not encode column {col}: {e}")

    print("Label counts after binarization:", df[label_col].value_counts())

    y = df[label_col]
    X = df.drop(columns=[label_col])

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y
