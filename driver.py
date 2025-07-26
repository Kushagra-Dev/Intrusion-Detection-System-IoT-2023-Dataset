#last prompt used - asiign data type for time and string and why sep='\x09' wont work?

import pandas as pd

file_path = "opt/Malware-Project/BigDataset/IoTScenarios/CTU-Honeypot-Capture-4-1/bro/conn.log.labeled"


with open(file_path, 'r') as f:
    columns = None
    zeek_types = None
    for line in f:
        if line.startswith("#fields"):
            columns = line.strip().split("\t")[1:]  
        elif line.startswith("#types"):
            zeek_types = line.strip().split("\t")[1:] 
        if columns and zeek_types:
            break


if not columns or not zeek_types or len(columns) != len(zeek_types):
    raise ValueError("Mismatch or missing #fields / #types in the log file.")


type_mapping = {
    'time': 'float64',
    'string': 'string',
    'addr': 'string',
    'port': 'Int64',        # nullable integer
    'enum': 'string',
    'interval': 'float64',
    'count': 'Int64',
    'bool': 'boolean',
    'set[string]': 'string'
}


dtype_dict = {
    name: type_mapping.get(ztype, 'string') 
    for name, ztype in zip(columns, zeek_types)
}


df = pd.read_csv(
    file_path,
    sep="\t",
    comment="#",               
    names=columns,             
    dtype=dtype_dict,           
    na_values=["-", "(empty)"], 
    engine="python"
)


print(df.head())
