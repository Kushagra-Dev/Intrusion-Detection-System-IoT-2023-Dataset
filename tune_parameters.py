import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import classification_report, f1_score, silhouette_score
from tqdm import tqdm

def grid_search_dbscan(X, y, eps_values, min_samples_values):
    best_score = 0
    best_params = {}
    results = []

    for eps in tqdm(eps_values, desc="Eps Loop"):
        for min_samples in min_samples_values:
            db = DBSCAN(eps=eps, min_samples=min_samples)
            preds = db.fit_predict(X)

            # Convert cluster labels to binary: -1 = anomaly/attack = 1
            y_pred = np.array([1 if p == -1 else 0 for p in preds])

            # Evaluate
            try:
                f1 = f1_score(y, y_pred)
                sil = silhouette_score(X, preds) if len(set(preds)) > 1 else -1

                results.append((eps, min_samples, f1, sil))

                if f1 > best_score:
                    best_score = f1
                    best_params = {
                        'eps': eps,
                        'min_samples': min_samples,
                        'f1_score': f1,
                        'silhouette_score': sil
                    }
            except Exception as e:
                print(f"Failed at eps={eps}, min_samples={min_samples} → {e}")

    print("\n Best Parameters Found:")
    print(best_params)
    return best_params, results

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_f1_heatmap(results_list):
    print("Plotting F1-score heatmap...")

    # Convert list of dicts to DataFrame
    df = pd.DataFrame(results_list)

    # Round values for better visualization consistency
    df["eps"] = df["eps"].round(2)
    df["min_samples"] = df["min_samples"].astype(int)

    # Pivot table (eps as index, min_samples as columns)
    heatmap_data = df.pivot_table(index="eps", columns="min_samples", values="f1_score")

    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="viridis")
    plt.title("DBSCAN F1 Score Heatmap")
    plt.xlabel("min_samples")
    plt.ylabel("eps")
    plt.tight_layout()
    plt.savefig("dbscan_f1_heatmap.png")
    print("F1-score heatmap saved as 'dbscan_f1_heatmap.png'")

