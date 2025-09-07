# src/analysis.py
import pandas as pd
from typing import List

def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df.corr(numeric_only=True)

def top_correlated_features(df: pd.DataFrame, target: str, topn: int = 3) -> List[str]:
    corr = df.corr(numeric_only=True)[target].drop(target).abs().sort_values(ascending=False)
    return corr.head(topn).index.tolist()
