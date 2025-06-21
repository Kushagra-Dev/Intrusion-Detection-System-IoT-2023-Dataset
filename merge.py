import os
import pandas as pd

root_dir = "/Users/kushagra/Downloads/opt/Malware-Project/BigDataset/IoTScenarios"
output_file = "iot23_merged.csv"
output_path = os.path.join(root_dir, output_file)

valid_exts = (".csv", ".labeled", ".log", ".binetflow")

csv_files = []
for root, dirs, files in os.walk(root_dir):
    for file in files:
        if file.lower().endswith(valid_exts) and output_file not in file:
            csv_files.append(os.path.join(root, file))

print(f"📦 Found {len(csv_files)} flow files.")
first = True

for file in csv_files:
    try:
        df = pd.read_csv(file, low_memory=False)
        if df.empty:
            print(f"Skipped empty file: {file}")
            continue
        df["source_file"] = os.path.basename(file)

        # Append mode, write header only once
        df.to_csv(output_path, mode='a', index=False, header=first)
        first = False
        print(f" Written: {file}")
    except Exception as e:
        print(f" Failed to read {file}: {e}")

print(f"\n Final merged CSV saved to: {output_path}")
