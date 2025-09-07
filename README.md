# Concrete EDA / Preprocessing

`concrete_data.csv`를 대상으로 **데이터 로드 → 결측 비율 출력 → 분포/이상치 시각화 → IQR 경계 대체 → 상관관계 분석 → 주요 특징 산점도 → MinMax/StandardScaler 비교**를 수행  
시각화 결과는 `./plots` 폴더에 저장 (화면에도 표시)

## Requirements
- Python 3.8+
- pandas, numpy
- matplotlib, seaborn
- scikit-learn

```bash
pip install pandas numpy matplotlib seaborn scikit-learn


1. Load & Missing Ratio: CSV 로드, 결측 비율 출력

2. Histograms: 모든 컬럼 히스토그램 → plots/hist_all_columns.png

3. Boxplots (Before): 컬럼별 박스플롯 그리드 → plots/boxplots_grid_before.png

4. Outlier Capping (IQR): 이상치를 IQR 경계값으로 대체

5. Boxplots (Before vs After) → plots/boxplots_before_after.png

6. Correlation Heatmap → plots/corr_heatmap.png

7. Scatter (cement-water-strength) → plots/scatter_cement_water_strength.png

8. Top-3 Features vs Target Scatter → plots/scatter_top3_vs_target.png

9. Scaling Comparison: MinMax/StandardScaler 적용 후 통계 비교 테이블 출력 (+ plots/scaling_comparison_stats.csv 저장)