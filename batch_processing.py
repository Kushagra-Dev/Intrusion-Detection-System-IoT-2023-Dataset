import dask.dataframe as dd

def retdf(file_pattern: str) -> dd.DataFrame:
    """
    Reads conn.log.labeled file(s) into a Dask DataFrame using a fixed, robust schema
    and explicit dtypes to prevent parsing errors.
    """
    MASTER_COLUMN_LIST = [
        'ts', 'uid', 'id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p',
        'proto', 'service', 'duration', 'orig_bytes', 'resp_bytes', 'conn_state',
        'local_orig', 'local_resp', 'missed_bytes', 'history', 'orig_pkts',
        'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes', 'tunnel_parents',
        'label', 'detailed-label'
    ]

    # This mapping prevents Dask from guessing dtypes incorrectly.
    DTYPE_MAPPING = {
        'ts': 'float64', 'id.orig_p': 'float64', 'id.resp_p': 'float64',
        'duration': 'float64', 'orig_bytes': 'float64', 'resp_bytes': 'float64',
        'missed_bytes': 'float64', 'orig_pkts': 'float64', 'orig_ip_bytes': 'float64',
        'resp_pkts': 'float64', 'resp_ip_bytes': 'float64',
        'service': 'object', 'label': 'object', 'detailed-label': 'object',
        'tunnel_parents': 'object'
    }

    print(f" Loading Dask DataFrame from '{file_pattern}' using robust settings...")
    
    df = dd.read_csv(
        file_pattern,
        # --- ### THE FIX: Switched from deprecated 'delim_whitespace' to 'sep' ### ---
        sep=r'\s+',          # Use the recommended regex for any whitespace
        engine='python',        # The regex separator requires the python engine
        # --- ### End of Fix ### ---
        comment='#',
        header=None,
        names=MASTER_COLUMN_LIST,
        na_values=['-', '(empty)'],
        blocksize="128MB",
        encoding_errors='ignore',
        on_bad_lines='skip',
        dtype=DTYPE_MAPPING # Use the explicit dtype mapping
    )
    
    print(f" Dask DataFrame loaded. Lazy shape: {df.shape[1]} columns.")
    return df
