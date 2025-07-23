"""
Main pipeline script for IoT23 Intrusion Detection using DBSCAN clustering.
Steps:
1. EDA
2. Preprocessing
3. Dimensionality Reduction (Incremental PCA)
4. Sampling and Balancing
5. Clustering with DBSCAN
"""

from PreProcessing import preprocess_data
from pca import run_incremental_pca
from model import train_dbscan
import pandas as pd
from EDA import run_eda
import numpy as np




csv_path = "/Users/kushagra/Downloads/iot23_combined_new.csv"

#Step 1: EDA
run_eda(csv_path)

# Step 2: Preprocessing
X, y = preprocess_data(csv_path)

# Step 3: Incremental PCA
X_pca = run_incremental_pca(X, n_components=15, batch_size=10000)

#Balancing dataset
benign_indices = np.where(y == 0)[0]
attack_indices = np.where(y == 1)[0]

print("Number of benign samples:", len(benign_indices))
print("Number of attack samples:", len(attack_indices))

if len(benign_indices) == 0 or len(attack_indices) == 0:
    raise ValueError("One of the classes is missing after preprocessing. Cannot balance the dataset.")


sample_size = min(len(benign_indices), len(attack_indices), 1000000)

np.random.seed(101)
benign_sample = np.random.choice(benign_indices, sample_size, replace=False)
attack_sample = np.random.choice(attack_indices, sample_size, replace=False)

balanced_indices = np.concatenate((benign_sample, attack_sample))
np.random.shuffle(balanced_indices)

X_pca_balanced = X_pca[balanced_indices]
y_balanced = y[balanced_indices]


print(f"Balanced dataset shape: {X_pca_balanced.shape}")

#Following code is used to check best hyperparameters -
 
#from tune_parameters import grid_search_dbscan  

# X_subset = X_pca_balanced
# y_subset = y_balanced

# Define range of hyperparameters to try
# eps_values = np.arange(0.1, 1.0, 0.1)
# min_samples_values = [3, 5, 7, 10]

# Run Grid Search
#best_params, all_results = grid_search_dbscan(X_subset, y_subset, eps_values, min_samples_values)
# best_params, all_results = grid_search_dbscan(X_pca_balanced, y_balanced, eps_values, min_samples_values)

# print("\nAll Grid Search Results:")
# for eps, min_s, f1, sil in all_results:
#     print(f"eps={eps:.2f}, min_samples={min_s}, f1={f1:.4f}, silhouette={sil:.4f}")

# best_params, all_results = grid_search_dbscan(
#     X_subset, y_subset,
#     eps_values=np.arange(0.30, 0.38, 0.01),
#     min_samples_values=[10, 11, 12, 13, 14, 15]
# )

# print("\nAll Grid Search Results:")
# for eps, min_s, f1, sil in all_results:
#     print(f"eps={eps:.2f}, min_samples={min_s}, f1={f1:.4f}, silhouette={sil:.4f}")

# print(f"\nBest Params: {best_params}")

#print("Subsampling 100,000 rows for DBSCAN...")


#print("\nRunning DBSCAN Grid Search...\n")
# run_dbscan_grid_search(
#     X=X_pca_balanced,
#     y_true=y_balanced,
#     eps_values=[0.3, 0.32, 0.35, 0.37, 0.39],
#     min_samples_values=[10, 15, 17, 20]
# )

y_balanced = pd.Series(y_balanced)

train_dbscan(X_pca_balanced, y_balanced, eps=0.38, min_samples=13)


