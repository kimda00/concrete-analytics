# src/io.py
import os
import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(df)  # 원래 코드처럼 전체 출력
    return df

def print_missing_ratio(df: pd.DataFrame) -> None:
    print("결측치 비율 :")
    print(df.isnull().mean() * 100)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)