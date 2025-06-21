import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def run_eda(X, columns, y_true):
    print("\n Running EDA...")
    df = pd.DataFrame(X, columns=columns)
    df['label'] = y_true

    plt.figure(figsize=(12, 6))
    sns.countplot(x='label', data=df)
    plt.title("Class Distribution")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("class_distribution.png")
    print("📊 Saved class distribution plot as class_distribution.png")
