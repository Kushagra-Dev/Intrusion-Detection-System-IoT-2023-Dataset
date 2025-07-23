import pandas as pd

df = pd.read_csv("/Users/kushagra/Downloads/iot23_combined_new.csv", low_memory=False)
print(df['label'].dropna().unique())

