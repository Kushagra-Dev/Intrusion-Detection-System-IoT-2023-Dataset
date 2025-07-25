import pandas as pd


file_path='/home/saurabh/Intrusion-Detection-System-IoT-2023-Dataset/opt/Malware-Project/BigDataset/IoTScenarios/CTU-Honeypot-Capture-4-1/bro/conn.log.labeled'


with open(file_path) as f:
    for line in f:
        if line.startswith('#fields'):
            columns= line.strip().split('\x09')[1:]
        if line.startswith('#types'):
            types=line.strip().split('\x09')[1::]
            break

print(df.head(1))
print(columns)
print(types)


df = pd.read_csv(file_path, sep='\x09', comment='#',header=columns, dtype=types)
