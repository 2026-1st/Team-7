import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from IPython.display import display
df = pd.read_csv("data/data_team7.csv")
print(df.head())
print(df.columns)
df.info()

def get_hr_data(filepath, model_type='tree'):
    """
    model_type: 'linear_xgb' (원-핫 인코딩) 또는 'lgbm_tabnet' (오디널 인코딩)
    """
    # 1. 데이터 로드 및 Feature / Target 분리
    df = pd.read_csv(filepath)
    target_col = 'Attrition'
    
    X = df.drop(target_col, axis=1)
    # Target이 'Yes'/'No'라면 1/0으로 변환
    y = df[target_col].apply(lambda x: 1 if x == 'Yes' else 0) if df[target_col].dtype == 'object' else df[target_col]

    # 2. 공정성을 위한 고정된 분할 (70:15:15)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    # 3. 변수 타입별 분류
    categorical_cols = ['BusinessTravel', 'Department','EducationField','Gender', 'JobRole', 'MaritalStatus', 'OverTime', ]
    numerical_cols = ['Age', 'DailyRate', 'DistanceFromHome', 'Education','EnvironmentSatisfaction','HourlyRate','JobInvolvement','JobLevel','JobSatisfaction','MonthlyIncome', 'MonthlyRate', 'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']

    # 4. 모델 타입에 따른 전처리기(ColumnTransformer) 구성
    if model_type == 'linear_xgb':
        # Track A: 다중공선성 방지를 위해 drop='first' 적용
        cat_transformer = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
    else:
        # Track B: 단순 정수 변환
        cat_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', cat_transformer, categorical_cols)
        ]
    )

    # 5. 전처리 적용
    X_train_processed = preprocessor.fit_transform(X_train)
    X_valid_processed = preprocessor.transform(X_valid)
    X_test_processed = preprocessor.transform(X_test)

    return X_train_processed, X_valid_processed, X_test_processed, y_train, y_valid, y_test, preprocessor

x_train, x_valid, x_test, y_train, y_valid, y_test, preprocessor = get_hr_data("data/data_team7.csv", model_type='linear_xgb')

## 평가지표 함수
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, log_loss)

def evaluate_model(y_test, y_pred, y_proba):
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    logloss = log_loss(y_test, y_proba)
    
    return precision, recall, f1, roc_auc, pr_auc, logloss

## 로지스틱 회귀 - 그리드서치
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# 테스트할 하이퍼파라미터 메뉴판(Grid) 정의
# lbfgs는 l1을 지원하지 않으므로, 에러 방지를 위해 liblinear와 lbfgs를 분리해서 정의
param_grid = [
    {
        'penalty': ['l1', 'l2'],
        'C': [0.1, 1.0, 10.0, 100.0],
        'class_weight': [None, 'balanced'],
        'solver': ['liblinear', 'saga']
    },
    {
        'penalty': ['l2'],
        'C': [0.1, 1.0, 10.0, 100.0],
        'class_weight': [None, 'balanced'],
        'solver': ['lbfgs']
    }
]

# 로지스틱 회귀 모델 생성
base_model = LogisticRegression(max_iter=3000, random_state=42)

scoring_metrics = {
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc',
    'pr_auc': 'average_precision',   
    'logloss': 'neg_log_loss'          
}

# GridSearchCV 객체 생성
grid_search = GridSearchCV(
    estimator=base_model, 
    param_grid=param_grid, 
    scoring=scoring_metrics, 
    refit=False, 
    cv=5, 
    n_jobs=-1
)

grid_search.fit(x_train, y_train)

# 전체 결과를 판다스 데이터프레임으로 변환
cv_results = pd.DataFrame(grid_search.cv_results_)

# 보기 쉽게 기본 스펙과 6대 지표 평균 점수만 정리
report_df = pd.DataFrame({
    'Class_Weight': cv_results['param_class_weight'],
    'Penalty': cv_results['param_penalty'],
    'C': cv_results['param_C'],
    'Solver': cv_results['param_solver'],
    'Precision': cv_results['mean_test_precision'].round(4),
    'Recall': cv_results['mean_test_recall'].round(4),
    'F1-Score': cv_results['mean_test_f1'].round(4),
    'ROC-AUC': cv_results['mean_test_roc_auc'].round(4),
    'PR-AUC': cv_results['mean_test_pr_auc'].round(4),
    # neg_log_loss에 다시 -1을 곱해서 원본 양수 스코어로 복원
    'LogLoss': (cv_results['mean_test_logloss'] * -1).round(4)
})

pd.set_option('display.max_columns', None)
print("\n Precision 높은 순 TOP 5")
display(report_df.sort_values(by='Precision', ascending=False).head(5))

print("\n Recall 높은 순 TOP 5")
display(report_df.sort_values(by='Recall', ascending=False).head(5))

print("F1-Score 높은 순 TOP 5")
display(report_df.sort_values(by='F1-Score', ascending=False).head(5))

print("\n ROC-AUC 높은 순 TOP 5")
display(report_df.sort_values(by='ROC-AUC', ascending=False).head(5))

print("\n PR-AUC 높은 순 TOP 5")
display(report_df.sort_values(by='PR-AUC', ascending=False).head(5))

print("\n LogLoss 낮은 순 TOP 5")
# 로그손실은 복원된 양수 기준이므로 ascending=True(낮은 순)로 정렬해야 1등이 맨 위로 옴
display(report_df.sort_values(by='LogLoss', ascending=True).head(5))

## 각 지표별 1등 조합으로 모델 학습
top_combinations = [
    {"name": "Model_Precision_Top (Index 1)", "class_weight": None, "penalty": "l1", "solver": "saga", "C": 0.1},
    {"name": "Model_Recall_Top (Index 4)", "class_weight": "balanced", "penalty": "l1", "solver": "liblinear", "C": 0.1},
    {"name": "Model_F1_Top (Index 18)", "class_weight": None, "penalty": "l2", "solver": "liblinear", "C": 10.0},
    {"name": "Model_ROCAUC_Top (Index 27)", "class_weight": None, "penalty": "l2", "solver": "saga", "C": 100.0},
    {"name": "Model_PRAUC_Top (Index 34)", "class_weight": None, "penalty": "l2", "solver": "lbfgs", "C": 1.0},
    {"name": "Model_LogLoss_Top (Index 11)", "class_weight": None, "penalty": "l2", "solver": "saga", "C": 1.0}
]

# 전처리 파이프라인에서 인코딩된 전체 변수 이름 순서대로 복원하기
feature_names = list(preprocessor.get_feature_names_out())

# 결과를 저장할 데이터프레임 생성
coef_comparison_df = pd.DataFrame({"Feature": feature_names})

# 6개의 최적 모델을 루프 돌며 통째로 재학습시키고 가중치를 뽑아 합치기
valid_results=[]

for comb in top_combinations:
    model = LogisticRegression(
        penalty=comb["penalty"],
        C=comb["C"],
        class_weight=comb["class_weight"],
        solver=comb["solver"],
        max_iter=3000,
        random_state=42
    )
    # 전체 Train 데이터셋으로 학습
    model.fit(x_train, y_train)
    
    # 검증
    y_pred_val = model.predict(x_valid)
    y_proba_val = model.predict_proba(x_valid)[:, 1]
    precision, recall, f1, roc_auc, pr_auc, logloss = evaluate_model(y_valid, y_pred_val, y_proba_val)
    
    valid_results.append({
        "Model": comb["name"].split(" ")[0],
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1-Score": round(f1, 4),
        "ROC-AUC": round(roc_auc, 4),
        "PR-AUC": round(pr_auc, 4),
        "LogLoss": round(logloss, 4)
    })
    
    
    # 해당 모델이 계산한 변수별 가중치를 컬럼으로 추가
    coef_comparison_df[comb["name"]] = model.coef_[0].round(4)

# 지표 별 가중치 크기 기준으로 정렬
# 모든 변수가 한눈에 다 보이도록 판다스 행 제한 해제 후 출력
pd.set_option('display.max_rows', 100)
print("[1] Precision 1등 모델 기준 변수 정렬")
display(coef_comparison_df[['Feature', 'Model_Precision_Top (Index 1)']].sort_values(by='Model_Precision_Top (Index 1)', ascending=False).head(10))

print("\n[2] Recall 1등 모델 기준 변수 정렬")
display(coef_comparison_df[['Feature', 'Model_Recall_Top (Index 4)']].sort_values(by='Model_Recall_Top (Index 4)', ascending=False).head(10))

print("\n [3] F1-Score 1등 모델 기준 변수 정렬")
display(coef_comparison_df[['Feature', 'Model_F1_Top (Index 18)']].sort_values(by='Model_F1_Top (Index 18)', ascending=False).head(10))

print("\n [4] ROC-AUC 1등 모델 기준 변수 정렬")
display(coef_comparison_df[['Feature', 'Model_ROCAUC_Top (Index 27)']].sort_values(by='Model_ROCAUC_Top (Index 27)', ascending=False).head(10))

print("\n [5] PR-AUC 1등 모델 기준 변수 정렬")
display(coef_comparison_df[['Feature', 'Model_PRAUC_Top (Index 34)']].sort_values(by='Model_PRAUC_Top (Index 34)', ascending=False).head(10))

print("\n [6] LogLoss 1등 모델 기준 변수 정렬")
display(coef_comparison_df[['Feature', 'Model_LogLoss_Top (Index 11)']].sort_values(by='Model_LogLoss_Top (Index 11)', ascending=False).head(10))

# 검증 결과
print("\n[검증 결과]")
display(pd.DataFrame(valid_results))

# pr-auc 1등 모델 기준 threshold 튜닝
import numpy as np

best_rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    class_weight=None,
    random_state=42,
    n_jobs=-1
)
best_rf_model.fit(x_train, y_train)
y_proba_rf_valid = best_rf_model.predict_proba(x_valid)[:, 1]

# 임계값(Threshold)을 0.1부터 0.9까지 0.05 간격으로 변경
thresholds = np.arange(0.1, 0.95, 0.05)
rf_threshold_tuning_results = []

for th in thresholds:
    # 확률이 th보다 크거나 같으면 1(퇴사), 작으면 0(잔류)으로 커스텀 판정
    y_pred_rf_custom = (y_proba_rf_valid >= th).astype(int)
    
    # 평가지표
    precision, recall, f1, roc_auc, pr_auc, logloss = evaluate_model(y_valid, y_pred_rf_custom, y_proba_rf_valid)
    
    rf_threshold_tuning_results.append({
        "Threshold": round(th, 2),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1-Score": round(f1, 4),
        "ROC-AUC": round(roc_auc, 4),
        "PR-AUC": round(pr_auc, 4),    
        "LogLoss": round(logloss, 4)   
    })

# 결과를 보기 편하게 데이터프레임으로 변환 후 출력
print("\n[PR-AUC 1등 모델 기준 Threshold 튜닝 결과]")
rf_tuning_df = pd.DataFrame(rf_threshold_tuning_results)
display(rf_tuning_df)
# PR-AUC 1등 모델 기준으로 Threshold 0.25가 가장 균형있음

## 최종 정리
# pr-auc 1등 모델로 하면 과대적합이 일어나서 max_depth를 10->4로 낮추고 min_samples_split은 5-> 10으로 튜닝
final_rf_master = RandomForestClassifier(
    n_estimators=300,
    max_depth=4,
    min_samples_split=10,
    class_weight=None,
    random_state=42,
    n_jobs=-1
)
final_rf_master.fit(x_train, y_train)

final_rf_th = 0.25

# [Train 데이터셋 채점]
y_proba_rf_train = final_rf_master.predict_proba(x_train)[:, 1]
y_pred_rf_train = (y_proba_rf_train >= final_rf_th).astype(int)
p_tr, r_tr, f_tr, roc_tr, pr_tr, log_tr = evaluate_model(y_train, y_pred_rf_train, y_proba_rf_train)

# [Valid 데이터셋 채점]
y_proba_rf_valid = final_rf_master.predict_proba(x_valid)[:, 1]
y_pred_rf_valid = (y_proba_rf_valid >= final_rf_th).astype(int)
p_va, r_va, f_va, roc_va, pr_va, log_va = evaluate_model(y_valid, y_pred_rf_valid, y_proba_rf_valid)

# [Test 데이터셋 채점]
y_proba_rf_test = final_rf_master.predict_proba(x_test)[:, 1]
y_pred_rf_test = (y_proba_rf_test >= final_rf_th).astype(int)
p_te, r_te, f_te, roc_te, pr_te, log_te = evaluate_model(y_test, y_pred_rf_test, y_proba_rf_test)

# 판다스 데이터프레임으로 3대 데이터셋 성적 통합
rf_total_performance_df = pd.DataFrame([
    {
        "데이터셋 (Dataset)": "훈련 데이터 (Train_70%)",
        "Precision": round(p_tr, 4), "Recall": round(r_tr, 4), "F1-Score": round(f_tr, 4),
        "ROC-AUC": round(roc_tr, 4), "PR-AUC": round(pr_tr, 4),"LogLoss": round(log_tr, 4)
    },
    {
        "데이터셋 (Dataset)": "검증 데이터 (Valid_15%)",
        "Precision": round(p_va, 4), "Recall": round(r_va, 4), "F1-Score": round(f_va, 4), 
        "ROC-AUC": round(roc_va, 4), "PR-AUC": round(pr_va, 4), "LogLoss": round(log_va, 4)
    },
    {
        "데이터셋 (Dataset)": "실전 데이터 (Test_15%)",
        "Precision": round(p_te, 4), "Recall": round(r_te, 4), "F1-Score": round(f_te, 4),
        "ROC-AUC": round(roc_te, 4), "PR-AUC": round(pr_te, 4), "LogLoss": round(log_te, 4)
    }
])

print("[최종 모델 성능]")
display(rf_total_performance_df)


# Feature Importance 추출 및 정리
feature_names_rf = list(preprocessor.get_feature_names_out())
final_rf_importances = pd.DataFrame({
    'Feature': feature_names_rf,
    'Importance': final_rf_master.feature_importances_.round(4)})

# 중요도가 높은 순서대로 예쁘게 정렬
final_rf_importances = final_rf_importances.sort_values(by='Importance', ascending=False).head(10)
final_rf_importances.reset_index(drop=True, inplace=True)

print("\n [최종 모델 기준] 랜덤 포레스트 변수 중요도 TOP 10")
display(final_rf_importances)