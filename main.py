# main.py
from src.io import load_data, print_missing_ratio, ensure_dir
from src.outliers import replace_outliers_with_bounds
from src.analysis import correlation_matrix, top_correlated_features
from src.viz import (
    hist_all_columns, boxplots_grid, boxplots_before_after,
    heatmap_corr, scatter_cement_water_strength,
    scatter_top_features_vs_target
)
from src.processing import scaling_comparison_table, save_table_csv

DATA_PATH = "./data/concrete_data.csv"
PLOT_DIR  = "./plots"
TARGET    = "concrete_compressive_strength"

def main():
    ensure_dir(PLOT_DIR)

    # 1) Load & Basic prints
    df = load_data(DATA_PATH)
    print_missing_ratio(df)

    # 2) Hist & Boxplot (before)
    hist_all_columns(df, outdir=PLOT_DIR, show=True)
    boxplots_grid(df, outdir=PLOT_DIR, show=True, name="boxplots_grid_before.png")

    # 3) Outlier capping (IQR bounds)
    df_capped = replace_outliers_with_bounds(df)

    # 4) Boxplot before vs after
    boxplots_before_after(df, df_capped, outdir=PLOT_DIR, show=True)

    # 5) Correlation heatmap
    corr = correlation_matrix(df)
    heatmap_corr(corr, outdir=PLOT_DIR, show=True)

    # 6) Cement–Water–Strength scatter
    scatter_cement_water_strength(df, outdir=PLOT_DIR, show=True)

    # 7) Top-3 correlated features vs target
    top3 = top_correlated_features(df, target=TARGET, topn=3)
    scatter_top_features_vs_target(df, top3, target=TARGET, outdir=PLOT_DIR, show=True)

    # 8) Scaling comparison (MinMax vs Standard)
    comp = scaling_comparison_table(df)
    print(comp.round(3))
    # CSV로 저장하고 싶으면:
    save_table_csv(comp.round(3), f"{PLOT_DIR}/scaling_comparison_stats.csv")

if __name__ == "__main__":
    main()
