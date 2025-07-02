import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda(file_path):
    print("Running EDA...")

    df = pd.read_csv(file_path, low_memory=False)

    print("\nTop 5 rows:")
    print(df.head())

    print("\nLabel distribution:")
    print(df['label'].value_counts())

    if 'label' in df.columns:
        plt.figure(figsize=(6, 4))
        df['label'].value_counts().plot(kind='bar', color='skyblue')
        plt.title('Label Distribution')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig("label_distribution.png")
        print("Label distribution plot saved as 'label_distribution.png'")
    else:
        print("No 'label' column found for distribution plot.")
