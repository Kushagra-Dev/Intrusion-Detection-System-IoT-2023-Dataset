from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix

def run_isolation_forest(X, y_true):
    print("\n Training Isolation Forest model...")
    model = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
    model.fit(X)
    
    y_pred = model.predict(X)
    y_pred = ['Attack' if val == -1 else 'Benign' for val in y_pred]

    print("\n Evaluation Metrics:")
    print(classification_report(y_true, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))