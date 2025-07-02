from sklearn.cluster import DBSCAN
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, Counter

def train_dbscan(X, y, eps=0.4, min_samples=3):
    print("Training DBSCAN...")
    db = DBSCAN(eps=eps, min_samples=min_samples)
    preds = db.fit_predict(X)

    # Map clusters to majority class
    cluster_to_class = {}
    cluster_labels = np.unique(preds)
    for cluster in cluster_labels:
        if cluster == -1:
            continue
        idxs = np.where(preds == cluster)[0]
        majority_class = Counter(y[idxs]).most_common(1)[0][0]
        cluster_to_class[cluster] = majority_class

    # Assign predictions
    predicted = []
    for i, cluster in enumerate(preds):
        if cluster == -1:
            predicted.append(1)  # noise = attack
        else:
            predicted.append(cluster_to_class.get(cluster, 0))

    print("Evaluation:")
    print(classification_report(y, predicted, target_names=["Benign", "Attack"]))

    # Debug cluster contents
    print("DBSCAN raw cluster labels:", Counter(preds))





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

    plt.title('DBSCAN Clustering (after PCA)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(loc='best', fontsize='small')
    plt.tight_layout()
    plt.savefig("dbscan_clusters.png")
    print("Cluster plot saved as 'dbscan_clusters.png'")
    plt.close()

