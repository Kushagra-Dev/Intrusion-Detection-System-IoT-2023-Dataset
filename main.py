from PreProcessing import preprocess_data_from_folder
from model import run_isolation_forest
from EDA import run_eda

if __name__ == '__main__':
    folder_path = "/Users/kushagra/Downloads/opt/Malware-Project/BigDataset/IoTScenarios/"

    print(" Starting preprocessing...")
    X_scaled, columns, y_true, df = preprocess_data_from_folder(folder_path, force=False)

    print("\n Performing EDA...")
    run_eda(X_scaled, columns, y_true)

    print("\n Running model...")
    run_isolation_forest(X_scaled, y_true)

    print("\n Pipeline execution complete.")
