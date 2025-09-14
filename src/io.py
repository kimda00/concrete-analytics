# src/io.py
import os
import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def print_missing_ratio(df: pd.DataFrame) -> None:
    ratio = (df.isnull().mean() * 100).round(2)
    print("결측치 비율(%) :")
    print(ratio)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)