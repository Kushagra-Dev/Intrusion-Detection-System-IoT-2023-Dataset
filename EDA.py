import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import glob

def plot_top_n_categorical(df, column, n, title, filename):
    """Helper function to plot the top N categorical features."""
    print(f"\nGenerating Top {n} {column.title()} Plot...")
    if column not in df.columns:
        print(f"   - Warning: Column '{column}' not found. Skipping plot.")
        return

    plt.figure(figsize=(12, 7))
    top_n = df[column].value_counts().nlargest(n)
    sns.barplot(x=top_n.index, y=top_n.values, hue=top_n.index, palette="viridis", legend=False)
    plt.title(title, fontsize=16)
    plt.ylabel('Count', fontsize=12)
    plt.xlabel(column.title(), fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(filename)
    print(f"   - Saved: {filename}")
    plt.close()

def plot_correlation_heatmap(df, title, filename):
    """Helper function to plot a correlation heatmap."""
    print("\nGenerating Correlation Heatmap...")
    
    df_corr = df.copy()
    for col in df_corr.select_dtypes(include=['object', 'category']).columns:
        df_corr[col] = df_corr[col].astype('category').cat.codes
        
    df_corr['label_numeric'] = df_corr['label_clean'].map({'Benign': 0, 'Malicious': 1}).fillna(0)
    df_corr = df_corr.drop(columns=['label_clean'])

    plt.figure(figsize=(18, 15))
    correlation_matrix = df_corr.corr()
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=.5)
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"   - Saved: {filename}")
    plt.close()

def run_eda(file_path_pattern: str, sample_size: int = 150000):
    """
    Performs Exploratory Data Analysis on a random sample of the data,
    generating a set of insightful plots.
    """
    print(f"Running EDA on a sample of up to {sample_size} rows...")

    MASTER_COLUMN_LIST = [
        'ts', 'uid', 'id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p',
        'proto', 'service', 'duration', 'orig_bytes', 'resp_bytes', 'conn_state',
        'local_orig', 'local_resp', 'missed_bytes', 'history', 'orig_pkts',
        'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes', 'tunnel_parents',
        'label', 'detailed-label'
    ]
    
    all_files = glob.glob(file_path_pattern)
    if not all_files:
        print(f"Error: No files found matching pattern: {file_path_pattern}")
        return

    print(f"   - Found {len(all_files)} files to load.")
    df_list = [pd.read_csv(f, sep=r'\s+', engine='python', comment='#', header=None,
                         names=MASTER_COLUMN_LIST, na_values=['-', '(empty)'],
                         on_bad_lines='skip') for f in all_files]
    
    df = pd.concat(df_list, axis=0, ignore_index=True)
    print(f" - Successfully loaded {len(df)} total rows.")
    
    sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
    print(f" - Using {len(sample_df)} rows for plotting.")

    sample_df['label_clean'] = sample_df['label'].str.split().str[0].str.strip()
    
    for col in sample_df.select_dtypes(include='number').columns:
        sample_df[col] = pd.to_numeric(sample_df[col], errors='coerce').fillna(0)

    sns.set_theme(style="whitegrid")

    # 1. Benign vs. Attack Bargraph
    plot_top_n_categorical(sample_df, 'label_clean', 10, 'Label Distribution (Sampled)', 'eda_label_distribution.png')
    
    # 2. Top 10 Services Plot
    plot_top_n_categorical(sample_df, 'service', 10, 'Top 10 Services (Sampled)', 'eda_top_services.png')
    
    # 3. Correlation Heatmap
    plot_correlation_heatmap(sample_df.drop(columns=['ts', 'uid']), 'Feature Correlation Heatmap', 'eda_correlation_heatmap.png')

    print("\nEDA complete.")


if __name__ == "__main__":
    FILE_PATH = "/Users/kushagra/Downloads/opt/Malware-Project/test/*/bro/conn.log.labeled"
    run_eda(FILE_PATH)
