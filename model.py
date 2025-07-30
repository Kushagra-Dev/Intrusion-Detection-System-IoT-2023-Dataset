import numpy as np
from scipy.stats import mode
from sklearn.cluster import DBSCAN
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def train_dbscan(X_train, y_train, X_test, y_test, eps, min_samples):
    """
    Trains DBSCAN, maps clusters using the TRAINING set,
    and evaluates performance on the unseen TESTING set.
    """
    print(f"Running DBSCAN with eps={eps}, min_samples={min_samples}...")
    
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    
    X_full_sample = np.vstack((X_train, X_test))
    dbscan_labels = model.fit_predict(X_full_sample)
    
    train_labels_len = len(y_train)
    dbscan_labels_train = dbscan_labels[:train_labels_len]
    dbscan_labels_test = dbscan_labels[train_labels_len:]
    
    print(" - Mapping cluster meanings using only the training data...")
    cluster_to_true_map = {}
    unique_clusters = np.unique(dbscan_labels_train[dbscan_labels_train != -1])

    for cluster_id in unique_clusters:
        indices = np.where(dbscan_labels_train == cluster_id)[0]
        if len(indices) == 0:
            continue
        majority_label = mode(y_train[indices], keepdims=True)[0][0]
        cluster_to_true_map[cluster_id] = majority_label
    
    y_pred_test = np.array([cluster_to_true_map.get(label, 1) for label in dbscan_labels_test])

    print("\n Classification Report on UNSEEN TEST DATA:")
    print(classification_report(y_test, y_pred_test, target_names=["Benign", "Attack"], zero_division=0))

    print(" Confusion Matrix on UNSEEN TEST DATA:")
    cm = confusion_matrix(y_test, y_pred_test)
    print(cm)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Benign", "Attack"], yticklabels=["Benign", "Attack"])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('DBSCAN Confusion Matrix (Test Set)')
    plt.savefig("dbscan_confusion_matrix.png")
    print("Evaluation complete.")
