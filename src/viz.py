# src/viz.py
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_theme(style="whitegrid")

def _savefig(outdir: str, name: str):
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, name), dpi=150, bbox_inches="tight")

def hist_all_columns(df: pd.DataFrame, outdir="./plots", show=True):
    df.hist(figsize=(12, 10))
    plt.tight_layout()
    _savefig(outdir, "hist_all_columns.png")
    if show: plt.show()
    plt.close("all")

def boxplots_grid(df: pd.DataFrame, outdir="./plots", show=True, name="boxplots_grid_before.png"):
    plt.figure(figsize=(15, 10))
    n = len(df.columns)
    rows = (n + 2) // 3  # 3열 그리드에 맞춰 행 수 자동 계산
    for i, column in enumerate(df.columns):
        plt.subplot(rows, 3, i + 1)
        sns.boxplot(y=df[column])
        plt.title(f'Boxplot - {column}')
        plt.tight_layout()
    _savefig(outdir, name)
    if show: plt.show()
    plt.close("all")

def boxplots_before_after(df_before: pd.DataFrame, df_after: pd.DataFrame, outdir="./plots", show=True):
    cols = df_before.columns.tolist()
    num_cols = len(cols)
    plt.figure(figsize=(18, 2 * num_cols))
    for i, column in enumerate(cols):
        plt.subplot(num_cols, 2, 2*i + 1)
        sns.boxplot(y=df_before[column], color='skyblue')
        plt.title(f'[Before] {column}', fontsize=10)
        plt.ylabel('')
        plt.subplot(num_cols, 2, 2*i + 2)
        sns.boxplot(y=df_after[column], color='lightgreen')
        plt.title(f'[After] {column}', fontsize=10)
        plt.ylabel('')
    plt.tight_layout()
    _savefig(outdir, "boxplots_before_after.png")
    if show: plt.show()
    plt.close("all")

def heatmap_corr(corr: pd.DataFrame, outdir="./plots", show=True):
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Feature Correlation Matrix')
    _savefig(outdir, "corr_heatmap.png")
    if show: plt.show()
    plt.close("all")

def scatter_cement_water_strength(df: pd.DataFrame, outdir="./plots", show=True):
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=df,
        x="cement",
        y="water",
        hue="concrete_compressive_strength",
        palette="viridis",
        alpha=0.7
    )
    plt.title("Cement, Water, Compressive Strength (color: Strength)", fontsize=14)
    plt.xlabel("Cement (kg/m³)")
    plt.ylabel("Water (kg/m³)")
    plt.legend(title="Compressive Strength", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    _savefig(outdir, "scatter_cement_water_strength.png")
    if show: plt.show()
    plt.close("all")

def scatter_top_features_vs_target(df: pd.DataFrame, features, target: str, outdir="./plots", show=True):
    plt.figure(figsize=(15, 4))
    for i, feature in enumerate(features):
        plt.subplot(1, len(features), i + 1)
        sns.scatterplot(data=df, x=feature, y=target, alpha=0.6)
        plt.title(f"{feature} vs {target}")
        plt.xlabel(feature); plt.ylabel("Compressive Strength")
    plt.tight_layout()
    _savefig(outdir, "scatter_top3_vs_target.png")
    if show: plt.show()
    plt.close("all")
