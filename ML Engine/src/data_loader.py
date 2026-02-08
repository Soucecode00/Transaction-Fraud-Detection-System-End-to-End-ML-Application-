# src/data_loader.py
import pandas as pd
import os
import config

def load_data():
    """
    Load transaction data from the path specified in config.

    Returns
    -------
    pd.DataFrame
        Dataframe containing transaction data
    """
    data_path = config.DATA_PATH
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at path: {data_path}")
    
    df = pd.read_csv(data_path)
    return df   