import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# K-fold
def create_kfold_splits(X_train, y_train, n_splits=5, random_state=42):
    """KFold를 사용하여 데이터 분할을 생성하는 함수"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_results = []
    for train_index, val_index in kf.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
        fold_results.append((X_train_fold, X_val_fold, y_train_fold, y_val_fold))
    return fold_results

def evaluate_model(model, X_train, X_test, y_train, y_test):
    # 모델 학습
    model.fit(X_train, y_train)

    # 테스트 데이터로 예측
    y_pred = model.predict(X_test)

    # 예측 성능 평가 (평균 절대 오차, R2 스코어, 평균 제곱 오차)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    return mse, mae, r2

def add_evaluation_results(test_name, mse, mae, r2, results=None):
    new_results = pd.DataFrame([
        [test_name, mse, mae, r2]
    ], columns=['test_name', 'Mean Squared Error', 'Mean Absolute Error', 'R2 Score'])

    if results is not None:
        new_results = pd.concat([results, new_results], ignore_index=True)

    return new_results

def train_and_evaluate_models(df, target_column, test_size=0.2, random_state=42, n_splits=5):
    X, y = df.drop(columns=[target_column]), df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )

    # K-fold
    fold_results = create_kfold_splits(X_train, y_train, n_splits=n_splits, random_state=random_state)

    # 정규화 적용
    scaler = StandardScaler()

    fold_results_scaled = []
    for X_train_fold, X_val_fold, y_train_fold, y_val_fold in fold_results:
        X_train_fold_scaled = scaler.fit_transform(X_train_fold)
        X_val_fold_scaled = scaler.transform(X_val_fold)
        fold_results_scaled.append((X_train_fold_scaled, X_val_fold_scaled, y_train_fold, y_val_fold))

    models = [
        [LinearRegression(), 'LinearRegression'],
        [DecisionTreeRegressor(), 'DecisionTreeRegressor'],
        [RandomForestRegressor(random_state=42, n_estimators=200), 'random_forest'],
        [GradientBoostingRegressor(random_state=42, n_estimators=1000), 'GradientBoostingRegressor'],
    ]

    # 5-fold evaluation
    results = None
    for X_train_fold, X_val_fold, y_train_fold, y_val_fold in fold_results_scaled:
        for model, name in models:
            mse, mae, r2 = evaluate_model(model, X_train, X_test, y_train, y_test)
            results = add_evaluation_results(name, mse, mae, r2, results=results)

    return results