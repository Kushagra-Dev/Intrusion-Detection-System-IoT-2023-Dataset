import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import LabelEncoder

# #DATA ANALYSIS ->

# Correct function to read a Parquet file
df = pd.read_parquet('/Users/kushagra/Downloads/archive/1-Neris-20110810.binetflow.parquet')

# Show basic info
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# Preview the first few rows
df.head()


# See Class Distribution (label)

# df['label'].value_counts().plot(kind='bar', color=['green', 'red', 'blue'])
# plt.title("Label Distribution")
# plt.xlabel("Label")
# plt.ylabel("Count")
# plt.show()

# See Botnet Families

# df['Family'].value_counts().plot(kind='barh', color='orange')
# plt.title("Botnet Families in This Capture")
# plt.xlabel("Flow Count")
# plt.ylabel("Botnet Family")
# plt.show()

# Protocol Usage

# df['proto'].value_counts().plot(kind='bar', color='skyblue')
# plt.title("Protocols Used")
# plt.xlabel("Protocol")
# plt.ylabel("Count")
# plt.show()

# Compare Feature Distributions (Normal vs Botnet)

# import seaborn as sns

# sns.boxplot(x='label', y='tot_bytes', data=df)
# plt.title("Total Bytes per Flow: Botnet vs Normal")
# plt.show()

# sns.boxplot(x='label', y='dur', data=df)
# plt.title("Duration of Flows: Botnet vs Normal")
# plt.show()

# Correlation Heatmap (on Numeric Features)

# import numpy as np

# Encode categorical cols temporarily for correlation
# df_copy = df.copy()
# for col in ['proto', 'dir', 'state']:
#     df_copy[col] = LabelEncoder().fit_transform(df_copy[col])

# Drop label/family
# df_corr = df_copy.drop(columns=['label', 'Family'])

# Plot heatmap
# plt.figure(figsize=(10, 8))
# sns.heatmap(df_corr.corr(), annot=True, fmt='.2f', cmap='coolwarm')
# plt.title("Feature Correlation Heatmap")
# plt.show()

# DATA PREPROCESSING ->

# Removing columns Famlily and Label as we'll use unsupervised model 
df_clean = df.drop(columns=['label', 'Family'])

print(df_clean.columns)


# HANDLING MISSING VALUES

# ENCODING CATEGORICAL VALUES

# NORMALISATION


