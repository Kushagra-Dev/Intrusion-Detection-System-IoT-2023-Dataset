import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data_from_folder(folder_path, force=False):
    preprocessed_path = os.path.join(folder_path, 'merged_preprocessed.csv')

    if os.path.exists(preprocessed_path) and not force:
        print("📦 Cached preprocessed file found. Loading it...")
        full_df = pd.read_csv(preprocessed_path)
    else:
        print("📂 Reading and merging CSV files...")
        csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
        all_dfs = []

        for file in csv_files:
            file_path = os.path.join(folder_path, file)
            print(f"Reading: {file_path}")
            df = pd.read_csv(file_path)
            all_dfs.append(df)

        full_df = pd.concat(all_dfs, ignore_index=True)
        full_df.to_csv(preprocessed_path, index=False)
        print(f"✅ Merged CSV saved to: {preprocessed_path}")

    y_true = full_df['label'].apply(lambda x: 'Attack' if x != 'BenignTraffic' else 'Benign')
    X = full_df.drop(columns=['label'], errors='ignore')
    X = X.select_dtypes(include=['int64', 'float64'])
    X.fillna(0, inplace=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, X.columns, y_true, full_df