# src/processing.py
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def scaling_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    # 정규화/표준화 적용
    minmax_scaled = pd.DataFrame(MinMaxScaler().fit_transform(df), columns=df.columns)
    zscore_scaled = pd.DataFrame(StandardScaler().fit_transform(df), columns=df.columns)

    def summary_stats(df_input: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            'mean': df_input.mean(),
            'median': df_input.median(),
            'std': df_input.std()
        })

    summary_original = summary_stats(df).rename(columns=lambda x: f'original_{x}')
    summary_minmax   = summary_stats(minmax_scaled).rename(columns=lambda x: f'minmax_{x}')
    summary_zscore   = summary_stats(zscore_scaled).rename(columns=lambda x: f'zscore_{x}')

    comparison_df = pd.concat([summary_original, summary_minmax, summary_zscore], axis=1)
    return comparison_df

def save_table_csv(df: pd.DataFrame, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=True)
