from sklearn.cluster import DBSCAN
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, Counter
from scipy.stats import mode


def train_dbscan(X_pca_balanced, y_balanced, eps=0.38, min_samples=15):
    print("Training DBSCAN...")

    model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = model.fit_predict(X_pca_balanced)

    print("Mapping clusters to true labels...")

    def map_clusters_to_labels(predicted_labels, true_labels):
        cluster_to_true = {}
        unique_clusters = set(predicted_labels)
        for cluster_id in unique_clusters:
            if cluster_id == -1:
                continue 
            indices = np.where(predicted_labels == cluster_id)
            majority = mode(true_labels[indices], keepdims=True)[0][0]
            cluster_to_true[cluster_id] = majority

        mapped = np.array([
            cluster_to_true[label] if label != -1 else 1  # Treat noise as Attack
            for label in predicted_labels
        ])
        return mapped

    mapped_labels = map_clusters_to_labels(labels, y_balanced.to_numpy())

    print("Evaluation:")
    print(classification_report(y_balanced, mapped_labels, target_names=["Benign", "Attack"]))

def plot_clusters(X, preds):
    if X.shape[1] < 2:
        print("Not enough dimensions for plotting. PCA should retain at least 2 components.")
        return

    plt.figure(figsize=(8, 6))
    unique_labels = set(preds)
    colormap = plt.cm.get_cmap('tab10', len(unique_labels))

    for k in unique_labels:
        col = 'black' if k == -1 else colormap(k)

        class_member_mask = (preds == k)
        xy = X[class_member_mask]

        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=4, label=f'Cluster {k}' if k != -1 else 'Noise')

    # from collections import Counter
    # print(Counter(preds))

    plt.title('DBSCAN Clustering (after PCA)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(loc='best', fontsize='small')
    plt.tight_layout()
    plt.savefig("dbscan_clusters.png")
    print("Cluster plot saved as 'dbscan_clusters.png'")
    plt.close()
