# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("madhavmalhotra/unb-cic-iot-dataset")

# print("Path to dataset files:", path)

import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load your CSV file
df = pd.read_csv("/Users/kushagra/.cache/kagglehub/datasets/madhavmalhotra/unb-cic-iot-dataset/versions/1/wataiData/csv/CICIoT2023/part-00000-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv")  # Replace with your actual path

# Separate features and labels
X = df.drop(columns=['label'])  # Drop label for now (used later only for evaluation)
y_true = df['label'].apply(lambda x: 'Attack' if x != 'Benign' else 'Benign')  # For evaluation

# Check for and drop non-numeric columns (if any)
X = X.select_dtypes(include=['int64', 'float64'])

# Fill missing values
X.fillna(0, inplace=True)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Preprocessing Complete")
print("Shape of input data:", X_scaled.shape)

import matplotlib.pyplot as plt
import seaborn as sns

# Step 0: Recreate a DataFrame from scaled features (optional but useful for plotting)
X_df = pd.DataFrame(X_scaled, columns=X.columns)
X_df['label'] = y_true.values

# Count of Benign vs Attack
plt.figure(figsize=(6, 4))
sns.countplot(x='label', data=X_df)
plt.title("Class Distribution (Benign vs Attack)")
plt.xlabel("Label")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Compute correlation matrix
corr_matrix = pd.DataFrame(X_scaled, columns=X.columns).corr()

# Plot correlation heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, cmap='coolwarm', linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()
