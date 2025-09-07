# src/outliers.py
import pandas as pd

def replace_outliers_with_bounds(df: pd.DataFrame) -> pd.DataFrame:
    """
    각 컬럼의 IQR 경계값 밖의 이상치를 경계값(lower/upper bound)으로 '대체'한다.
    (원본을 삭제하지 않고 값만 깎아/cap 한다)
    """
    df_replaced = df.copy()
    for column in df.columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df_replaced[column] = df_replaced[column].apply(
            lambda x: lower_bound if x < lower_bound else upper_bound if x > upper_bound else x
        )
    return df_replaced
