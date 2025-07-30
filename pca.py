import dask.dataframe as dd
import dask.array as da
import numpy as np
from sklearn.decomposition import IncrementalPCA

def run_pca(X_dd: dd.DataFrame, n_components: int, batch_size: int = 5000) -> np.ndarray:
    """
    Performs PCA using sklearn's memory-efficient IncrementalPCA on a Dask DataFrame.
    The transform step is parallelized for high performance.
    """
    print(f" Running Incremental PCA with batch size {batch_size}...")

    X_dd = X_dd.repartition(partition_size="25MB")
    X_da = X_dd.astype(float).to_dask_array(lengths=True)
    
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)

    print(" - Fitting Incremental PCA model in batches...")
    for chunk in X_da.blocks:
        ipca.partial_fit(chunk.compute())
    
    print(" - Transforming data in parallel...")
    X_pca_da = da.map_blocks(ipca.transform, X_da, dtype=np.float64)
    
    X_pca = X_pca_da.compute()
    
    print("Incremental PCA complete")
    return X_pca
