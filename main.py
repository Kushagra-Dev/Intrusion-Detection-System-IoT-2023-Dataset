from PreProcessing import preprocess_data
from pca import run_incremental_pca
from model import train_dbscan
from EDA import run_eda
import numpy as np

csv_path = "/Users/kushagra/Downloads/iot23_combined_new.csv"

# Step 1: EDA
#run_eda(csv_path)

# Step 2: Preprocessing
X, y = preprocess_data(csv_path)

# Step 3: Incremental PCA
X_pca = run_incremental_pca(X, n_components=7, batch_size=10000)



benign_indices = np.where(y == 0)[0]
attack_indices = np.where(y == 1)[0]

sample_size = min(len(benign_indices), len(attack_indices), 100000)

np.random.seed(42)
benign_sample = np.random.choice(benign_indices, sample_size, replace=False)
attack_sample = np.random.choice(attack_indices, sample_size, replace=False)

balanced_indices = np.concatenate((benign_sample, attack_sample))
np.random.shuffle(balanced_indices)

X_pca_balanced = X_pca[balanced_indices]
y_balanced = y[balanced_indices]

print(f"Balanced dataset shape: {X_pca_balanced.shape}")
from tune_parameters import grid_search_dbscan  

X_subset = X_pca_balanced
y_subset = y_balanced

# Define range of hyperparameters to try
eps_values = np.arange(0.1, 1.0, 0.1)
min_samples_values = [3, 5, 7, 10]

# Run Grid Search
#best_params, all_results = grid_search_dbscan(X_subset, y_subset, eps_values, min_samples_values)
# best_params, all_results = grid_search_dbscan(X_pca_balanced, y_balanced, eps_values, min_samples_values)

# print("\nAll Grid Search Results:")
# for eps, min_s, f1, sil in all_results:
#     print(f"eps={eps:.2f}, min_samples={min_s}, f1={f1:.4f}, silhouette={sil:.4f}")




print("Subsampling 100,000 rows for DBSCAN...")
# X_pca = X_pca[:200000]
# y = y[:200000]

train_dbscan(X_pca_balanced, y_balanced, eps=0.06, min_samples=17)


#/Users/kushagra/Downloads/iot23_combined_new.csv