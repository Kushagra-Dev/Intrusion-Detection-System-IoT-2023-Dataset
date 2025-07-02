import numpy as np
from sklearn.decomposition import IncrementalPCA

def run_incremental_pca(X, n_components=5, batch_size=10000):
    print("Running Incremental PCA...")
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)

    X_pca = []
    for i in range(0, X.shape[0], batch_size):
        end = i + batch_size
        X_batch = X[i:end]
        if i == 0:
            ipca.partial_fit(X_batch)
        X_pca.append(ipca.transform(X_batch))

        print(f"Processed batch {i // batch_size + 1}")

    X_pca = np.vstack(X_pca)
    print(f"PCA complete. New shape: {X_pca.shape}")
    return X_pca
