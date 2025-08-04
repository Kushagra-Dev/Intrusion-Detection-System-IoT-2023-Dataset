import numpy as np
from scipy.stats import mode
from sklearn.cluster import DBSCAN
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import dask
from dask.delayed import delayed
from dask.base import compute

def _run_single_dbscan_test(X_sample, y_sample, eps, min_samples):
    """
    Internal helper function to run one DBSCAN test for the grid search.
    Returns the F1-score.
    """
    # Split data for a fair evaluation of this specific parameter set
    X_train, X_test, y_train, y_test = train_test_split(
        X_sample, y_sample, test_size=0.25, random_state=42, stratify=y_sample
    )
    y_train, y_test = np.asarray(y_train), np.asarray(y_test)

    model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    dbscan_labels = model.fit_predict(X_sample)
    
    train_labels_len = len(y_train)
    dbscan_labels_train = dbscan_labels[:train_labels_len]
    dbscan_labels_test = dbscan_labels[train_labels_len:]
    
    cluster_to_true_map = {}
    unique_clusters = np.unique(dbscan_labels_train[dbscan_labels_train != -1])
    for cluster_id in unique_clusters:
        indices = np.where(dbscan_labels_train == cluster_id)[0]
        if len(indices) == 0: continue
        majority_label = mode(y_train[indices], keepdims=True)[0][0]
        cluster_to_true_map[cluster_id] = majority_label
    
    y_pred_test = np.array([cluster_to_true_map.get(label, 1) for label in dbscan_labels_test])
    return f1_score(y_test, y_pred_test, average='weighted')

def tune_dbscan_hyperparameters(X_sample, y_sample, eps_values, min_samples_values):
    """
    Performs a parallelized grid search over DBSCAN hyperparameters using Dask.
    """
    print(" Starting Hyperparameter Grid Search...")
    print(f" - Using a sample of shape {X_sample.shape} for all tests.")

    tasks = []
    param_combinations = []
    for eps in eps_values:
        for min_s in min_samples_values:
            param_combinations.append({'eps': eps, 'min_samples': min_s})
            task = delayed(_run_single_dbscan_test)(X_sample, y_sample, eps, min_s)
            tasks.append(task)

    print(f"\n - Executing {len(tasks)} tests in parallel")
    scores = compute(*tasks)
    print("- All parallel tests complete")
    
    results = []
    for params, score in zip(param_combinations, scores):
        results.append((params['eps'], params['min_samples'], score))
    
    best_eps, best_min_s, best_score = max(results, key=lambda item: item[2])
    best_params = {'eps': best_eps, 'min_samples': best_min_s}
    
    print("\n\n- Grid Search Complete")
    print(f" Best F1-Score: {best_score:.4f}")
    print(f" Best Parameters: {best_params}")
    
    print("\nFull Results (sorted by score):")
    results.sort(key=lambda item: item[2], reverse=True)
    for eps, min_s, score in results:
        print(f"  eps={eps:.2f}, min_samples={min_s}, F1-Score={score:.4f}")

def train_dbscan(X_train, y_train, X_test, y_test, eps, min_samples):
    """
    Trains DBSCAN, maps clusters using the TRAINING set,
    and evaluates performance on the unseen TESTING set.
    """
    print(f" Running DBSCAN with eps={eps}, min_samples={min_samples}...")
    
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

    print("Confusion Matrix on UNSEEN TEST DATA:")
    cm = confusion_matrix(y_test, y_pred_test)
    print(cm)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Benign", "Attack"], yticklabels=["Benign", "Attack"])
    plt.title('DBSCAN Confusion Matrix (Test Set)')
    plt.savefig("dbscan_confusion_matrix.png")
    print("\n Evaluation complete.")
