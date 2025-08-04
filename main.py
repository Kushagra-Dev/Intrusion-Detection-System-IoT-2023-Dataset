import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
from sklearn.metrics import f1_score
from scipy.stats import mode
from dask.distributed import Client
from dask import config

from PreProcessing import preprocess_data
from pca import run_pca
from model import train_dbscan  

def main_pipeline(file_path, run_tuning, pca_components, sample_size, eps, min_samples, eps_grid, min_samples_grid):
    
    # Load, Preprocess, and run PCA
    X_dd, y_dd = preprocess_data(file_path)
    X_pca_np = run_pca(X_dd, n_components=pca_components)
    y_all = y_dd.compute()
    y_all_np = y_all.to_numpy()

    # Perform balanced undersampling
    print(f"\n Performing balanced undersampling...")
    benign_indices = np.where(y_all_np == 0)[0]
    attack_indices = np.where(y_all_np == 1)[0]

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

    X_sample = X_pca_np[balanced_indices]
    y_sample = y_all_np[balanced_indices]

    if run_tuning:
        print("\n Running grid search for DBSCAN hyperparameters...")
        best_f1 = 0
        best_params = (None, None)

        for eps_val in eps_grid:
            for min_s in min_samples_grid:
                print(f"\nTrying eps={eps_val}, min_samples={min_s}...")
                X_train, X_test, y_train, y_test = train_test_split(
                    X_sample, y_sample, test_size=0.25, random_state=42, stratify=y_sample
                )

                model = DBSCAN(eps=eps_val, min_samples=min_s, n_jobs=-1)
                dbscan_labels = model.fit_predict(np.vstack((X_train, X_test)))

                dbscan_labels_train = dbscan_labels[:len(y_train)]
                dbscan_labels_test = dbscan_labels[len(y_train):]

                # Map clusters to majority class in training
                cluster_to_label = {}
                for label in np.unique(dbscan_labels_train):
                    if label == -1:
                        continue
                    indices = np.where(dbscan_labels_train == label)[0]
                    if len(indices) == 0:
                        continue
                    majority = mode(y_train[indices], keepdims=True)[0][0]
                    cluster_to_label[label] = majority

                y_pred_test = np.array([cluster_to_label.get(l, 1) for l in dbscan_labels_test])
                score = f1_score(y_test, y_pred_test, average='binary', zero_division=0)
                print(f" - F1 Score: {score:.4f}")

                if score > best_f1:
                    best_f1 = score
                    best_params = (eps_val, min_s)

        print(f"\n Best parameters: eps={best_params[0]}, min_samples={best_params[1]} with F1 = {best_f1:.4f}")
        print(" Re-training final model with best parameters...\n")

        # Train with best parameters
        final_eps, final_min_s = best_params
        X_train, X_test, y_train, y_test = train_test_split(
            X_sample, y_sample, test_size=0.25, random_state=42, stratify=y_sample
        )
        train_dbscan(
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            eps=final_eps, min_samples=final_min_s
        )
    else:
        print(f"\n Running DBSCAN training...")
        print(f"  - Sampled dataset shape: {X_sample.shape}")
        X_train, X_test, y_train, y_test = train_test_split(
            X_sample, y_sample, test_size=0.25, random_state=42, stratify=y_sample
        )
        train_dbscan(
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            eps=eps, min_samples=min_samples
        )

    print("\n Pipeline finished successfully!")

if __name__ == "__main__":
    config.set({'distributed.worker.daemon': False})
    client = Client(n_workers=4, threads_per_worker=1, memory_limit='4GB')
    print(f"Dask client running. Dashboard at: {client.dashboard_link}")

    FILE_PATH = "/Users/kushagra/Downloads/opt/Malware-Project/test/*/bro/conn.log.labeled"
    
    # Turn tuning on/off
    RUN_TUNING = True

    # Parameters for a SINGLE RUN (when tuning is off)
    PCA_COMPONENTS = 15
    SAMPLE_SIZE = 75000
    EPS = 0.5
    MIN_SAMPLES = 20

    # Parameter Grid for TUNING
    TUNING_SAMPLE_SIZE = 50000 
    EPS_GRID = np.linspace(0.3, 1.0, 8)
    MIN_SAMPLES_GRID = [15, 20, 25, 30, 35]

    active_sample_size = TUNING_SAMPLE_SIZE if RUN_TUNING else SAMPLE_SIZE

    main_pipeline(
        file_path=FILE_PATH,
        run_tuning=RUN_TUNING,
        pca_components=PCA_COMPONENTS,
        sample_size=active_sample_size,
        eps=EPS,
        min_samples=MIN_SAMPLES,
        eps_grid=EPS_GRID,
        min_samples_grid=MIN_SAMPLES_GRID
    )

    client.close()


#best paras - 
#eps = 0.7
#min_samples = 15