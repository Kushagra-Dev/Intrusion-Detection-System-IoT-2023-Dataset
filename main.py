import numpy as np
from sklearn.model_selection import train_test_split
from dask.distributed import Client
from PreProcessing import preprocess_data
from pca import run_pca
from model import train_dbscan

def main_pipeline(file_path, pca_components, sample_size, eps, min_samples):
    """The complete pipeline execution in parallel mode."""
    
    X_dd, y_dd = preprocess_data(file_path)
    X_pca_np = run_pca(X_dd, n_components=pca_components)
    y_all = y_dd.compute()

    print(f"Performing balanced undersampling...")
    benign_indices = np.where(y_all == 0)[0]
    attack_indices = np.where(y_all == 1)[0]
    
    print(f" - Original benign count: {len(benign_indices)}")
    print(f" - Original attack count: {len(attack_indices)}")

    if len(benign_indices) == 0 or len(attack_indices) == 0:
        print("\n Warning: Data file contains only one class. Skipping modeling.")
        return

    actual_sample_size = min(sample_size, len(benign_indices), len(attack_indices))
    if actual_sample_size < sample_size:
        print(f"   - Sample size adjusted to {actual_sample_size} due to smaller class size.")

    benign_sample_indices = np.random.choice(benign_indices, size=actual_sample_size, replace=False)
    attack_sample_indices = np.random.choice(attack_indices, size=actual_sample_size, replace=False)
    
    balanced_indices = np.concatenate([benign_sample_indices, attack_sample_indices])
    np.random.shuffle(balanced_indices)
    
    y_all_np = y_all.to_numpy()
    X_sample = X_pca_np[balanced_indices]
    y_sample = y_all_np[balanced_indices]
    
    print(f" - Sampled dataset shape: {X_sample.shape}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_sample, y_sample, test_size=0.25, random_state=42, stratify=y_sample
    )
    print(" Sampling and splitting complete")
    
    train_dbscan(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
        eps=eps, min_samples=min_samples
    )
    print("\n Pipeline finished successfully")

if __name__ == "__main__":
    
    client = Client(n_workers=4, threads_per_worker=1, memory_limit='4GB')
    print(f"Dask client running. Dashboard at: {client.dashboard_link}")

    FILE_PATH = "/Users/kushagra/Downloads/opt/Malware-Project/test/*/bro/conn.log.labeled"
    PCA_COMPONENTS = 15
    SAMPLE_SIZE = 75000
    EPS = 0.5
    MIN_SAMPLES = 20
    
    main_pipeline(
        file_path=FILE_PATH,
        pca_components=PCA_COMPONENTS,
        sample_size=SAMPLE_SIZE,
        eps=EPS,
        min_samples=MIN_SAMPLES
    )
    
    client.close()
