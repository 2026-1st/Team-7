import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from IPython.display import display
import numpy as np

# 데이터 로드 및 확인
df = pd.read_csv("data/data_team7.csv")
print(df.head())
print(df.columns)
df.info()

# ==========================================
# 피처 엔지니어링
# ==========================================
def apply_advanced_feature_engineering(df):
    df_new = df.copy()

    df_new['Total_Satisfaction_Score'] = (
        df_new['EnvironmentSatisfaction'] + 
        df_new['JobSatisfaction'] + 
        df_new['RelationshipSatisfaction'] + 
        df_new['WorkLifeBalance']
    )

    df_new['Income_Per_WorkingYear'] = df_new['MonthlyIncome'] / (df_new['TotalWorkingYears'] + 1)
    df_new['Income_Per_YearAtCompany'] = df_new['MonthlyIncome'] / (df_new['YearsAtCompany'] + 1)
    df_new['Income_Per_Level'] = df_new['MonthlyIncome'] / df_new['JobLevel']
    df_new['Cost_Effectiveness'] = df_new['PercentSalaryHike'] / df_new['PerformanceRating']

    is_overtime = df_new['OverTime'].apply(lambda x: 1 if x == 'Yes' or x == 1 else 0)
    
    wlb_reversed = 5 - df_new['WorkLifeBalance'] #워라벨은 높을수록 좋은거였으니 번아웃 리스크 피쳐를 위해 낮은게 좋은 것으로 변환
    df_new['Burnout_Risk'] = is_overtime + wlb_reversed
    df_new['Sat_WLB_Interaction'] = df_new['Total_Satisfaction_Score'] * df_new['WorkLifeBalance']

    df_new['Promotion_Speed_Index'] = df_new['JobLevel'] / (df_new['TotalWorkingYears'] + 1)
    df_new['Stagnation_Index'] = df_new['YearsSinceLastPromotion'] / (df_new['YearsAtCompany'] + 1) 
    df_new['Job_Hopping_Index'] = df_new['NumCompaniesWorked'] / (df_new['TotalWorkingYears'] + 1)
    
    df_new['Loyalty_Ratio'] = np.where(
        df_new['TotalWorkingYears'] > 0,
        df_new['YearsAtCompany'] / df_new['TotalWorkingYears'],
        0
    )

    return df_new

# ==========================================
# 데이터 전처리
# ==========================================
def get_hr_data(filepath, model_type='tree'):
    """
    model_type: 'linear_xgb' (원-핫 인코딩) 또는 'lgbm_tabnet' (오디널 인코딩)
    """
    # 1. 데이터 로드 및 Feature / Target 분리
    df = pd.read_csv(filepath)
    target_col = 'Attrition'
    
    df_eng = apply_advanced_feature_engineering(df)
    
    X = df_eng.drop(target_col, axis=1)
    # Target이 'Yes'/'No'라면 1/0으로 변환
    y = df_eng[target_col].apply(lambda x: 1 if x == 'Yes' else 0) if df_eng[target_col].dtype == 'object' else df_eng[target_col]

    # 2. 공정성을 위한 고정된 분할 (70:15:15)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    # 3. 변수 타입별 분류
    categorical_cols = ['BusinessTravel', 'Department','EducationField','Gender', 'JobRole', 'MaritalStatus', 'OverTime', ]
    numerical_cols = ['Age', 'DailyRate', 'DistanceFromHome', 'Education','EnvironmentSatisfaction','HourlyRate',
                      'JobInvolvement','JobLevel','JobSatisfaction','MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 
                      'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears', 
                      'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 
                      'YearsSinceLastPromotion', 'YearsWithCurrManager', 'Total_Satisfaction_Score', 'Income_Per_WorkingYear', 'Income_Per_YearAtCompany', 'Income_Per_Level',
                      'Cost_Effectiveness', 'Burnout_Risk', 'Sat_WLB_Interaction', 'Promotion_Speed_Index', 'Stagnation_Index', 'Job_Hopping_Index', 'Loyalty_Ratio']

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

# 피처 엔지니어링이 반영된 트리 전용 데이터셋을 로드
x_train_eng, x_valid_eng, x_test_eng, y_train, y_valid, y_test, preprocessor_eng = get_hr_data("data/data_team7.csv", model_type='tree')


# ==========================================
# 평가지표 함수
# ==========================================
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, log_loss)

def evaluate_model(y_test, y_pred, y_proba):
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    logloss = log_loss(y_test, y_proba)
    
    return precision, recall, f1, roc_auc, pr_auc, logloss


# ==============================================================================
# 1단계: 그리드 서치 및 지표별 최적 중요도 비교 (피처 엔지니어링 전 원본 데이터 기준)
# ==============================================================================
print("\n" + "="*35 + " [그리드 서치 및 지표별 최적 모델 탐색] " + "="*35)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# 피처 엔지니어링 전 순정 상태의 오디널 인코딩 전처리를 위해 오리지널 목록 정의
categorical_cols_pre = ['BusinessTravel', 'Department','EducationField','Gender', 'JobRole', 'MaritalStatus', 'OverTime']
numerical_cols_pre = ['Age', 'DailyRate', 'DistanceFromHome', 'Education','EnvironmentSatisfaction','HourlyRate','JobInvolvement','JobLevel','JobSatisfaction','MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']

preprocessor_pre = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols_pre),
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_cols_pre)
    ]
)

df_ori = pd.read_csv("data/data_team7.csv")
y_ori = df_ori['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
X_ori = df_ori.drop('Attrition', axis=1)

X_train_ori, X_temp_ori, y_train, y_temp = train_test_split(X_ori, y_ori, test_size=0.3, stratify=y_ori, random_state=42)
X_valid_ori, X_test_ori, y_valid, y_test = train_test_split(X_temp_ori, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

x_train_pre = preprocessor_pre.fit_transform(X_train_ori)
x_valid_pre = preprocessor_pre.transform(X_valid_ori)

param_grid = {
    'n_estimators':[10, 50, 100, 300],
    'max_depth' : [None, 5, 10, 15],
    'min_samples_split' : [2, 5, 10],
    'class_weight': [None, 'balanced']
}

base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
scoring_metrics = {
    'precision': 'precision', 'recall': 'recall', 'f1': 'f1', 
    'roc_auc': 'roc_auc', 'pr_auc': 'average_precision', 'logloss': 'neg_log_loss'
}

grid_search = GridSearchCV(estimator=base_model, param_grid=param_grid, scoring=scoring_metrics, refit=False, cv=5, n_jobs=-1)
grid_search.fit(x_train_pre, y_train)

cv_results = pd.DataFrame(grid_search.cv_results_)
report_df = pd.DataFrame({
    'Class_Weight': cv_results['param_class_weight'], 'N_Estimators': cv_results['param_n_estimators'],
    'Max_Depth': cv_results['param_max_depth'], 'Min_Samples_Split': cv_results['param_min_samples_split'],
    'Precision': cv_results['mean_test_precision'].round(4), 'Recall': cv_results['mean_test_recall'].round(4),
    'F1-Score': cv_results['mean_test_f1'].round(4), 'ROC-AUC': cv_results['mean_test_roc_auc'].round(4),
    'PR-AUC': cv_results['mean_test_pr_auc'].round(4), 'LogLoss': (cv_results['mean_test_logloss'] * -1).round(4)
})

for metric, asc in [('Precision', False), ('Recall', False), ('F1-Score', False), ('ROC-AUC', False), ('PR-AUC', False), ('LogLoss', True)]:
    print(f"\n {metric} {'낮은' if asc else '높은'} 순 TOP 5")
    display(report_df.sort_values(by=metric, ascending=asc).head(5))

# 지표별 1등 조합 복원학습 (ROC-AUC와 PR-AUC 중복으로 총 5개)
top_combinations = [
    {"name": "Model_Precision_Top (Index 34)", "class_weight": None, "n_estimators": 100, "max_depth": 10, "min_samples_split": 10},
    {"name": "Model_Recall_Top (Index 68)", "class_weight": "balanced", "n_estimators": 10, "max_depth": 5, "min_samples_split": 10},
    {"name": "Model_F1_Top (Index 70)", "class_weight": "balanced", "n_estimators": 100, "max_depth": 5, "min_samples_split": 10},
    {"name": "Model_ROCAUC_PRAUC_Top (Index 31)", "class_weight": None, "n_estimators": 300, "max_depth": 10, "min_samples_split": 5},
    {"name": "Model_LogLoss_Top (Index 3)", "class_weight": None, "n_estimators": 300, "max_depth": None, "min_samples_split": 2}
]

feature_names_pre = list(preprocessor_pre.get_feature_names_out())
importance_comparison_df = pd.DataFrame({"Feature": feature_names_pre})
valid_results = []

for comb in top_combinations:
    model = RandomForestClassifier(n_estimators=comb["n_estimators"], max_depth=comb["max_depth"], min_samples_split=comb["min_samples_split"], class_weight=comb["class_weight"], random_state=42, n_jobs=-1)
    model.fit(x_train_pre, y_train)
    y_pred_val = model.predict(x_valid_pre)
    y_proba_val = model.predict_proba(x_valid_pre)[:, 1]
    precision, recall, f1, roc_auc, pr_auc, logloss = evaluate_model(y_valid, y_pred_val, y_proba_val)
    
    valid_results.append({
        "Model": comb["name"].split(" ")[0], "Precision": round(precision, 4), "Recall": round(recall, 4),
        "F1-Score": round(f1, 4), "ROC-AUC": round(roc_auc, 4), "PR-AUC": round(pr_auc, 4), "LogLoss": round(logloss, 4)
    })
    importance_comparison_df[comb["name"]] = model.feature_importances_.round(4)

for comb in top_combinations:
    print(f"\n[{comb['name'].split(' ')[0]}] 기준 변수 정렬")
    display(importance_comparison_df[['Feature', comb["name"]]].sort_values(by=comb["name"], ascending=False).head(10))

print("\n[각 최적 조합별 검증 결과]")
display(pd.DataFrame(valid_results))


# ==========================================
# 2단계: PR-AUC 1등 모델 기준 Threshold 튜닝
# ==========================================
print("\n" + "="*40 + " [PR-AUC 1등 모델 기준 Threshold 튜닝] " + "="*40)
best_rf_model = RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_split=10, class_weight=None, random_state=42, n_jobs=-1)
best_rf_model.fit(x_train_pre, y_train)
y_proba_rf_valid = best_rf_model.predict_proba(x_valid_pre)[:, 1]

thresholds = np.arange(0.1, 0.95, 0.05)
rf_threshold_tuning_results = []

for th in thresholds:
    y_pred_rf_custom = (y_proba_rf_valid >= th).astype(int)
    precision, recall, f1, roc_auc, pr_auc, logloss = evaluate_model(y_valid, y_pred_rf_custom, y_proba_rf_valid)
    rf_threshold_tuning_results.append({
        "Threshold": round(th, 2), "Precision": round(precision, 4), "Recall": round(recall, 4),
        "F1-Score": round(f1, 4), "ROC-AUC": round(roc_auc, 4), "PR-AUC": round(pr_auc, 4), "LogLoss": round(logloss, 4)
    })
display(pd.DataFrame(rf_threshold_tuning_results))


# ==============================================================================
# 3단계: [최종 모델 성능] 데이터 분할별 랜덤 포레스트 분류 성능 비교
# ==============================================================================
print("\n" + "="*40 + " [최종 모델 성능] 데이터 분할별 랜덤 포레스트 분류 성능 비교 " + "="*40)
final_rf_master = RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_split=10, class_weight=None, random_state=42, n_jobs=-1)
final_rf_master.fit(x_train_pre, y_train)

final_rf_th = 0.25

p_tr1, r_tr1, f_tr1, roc_tr1, pr_tr1, log_tr1 = evaluate_model(y_train, (final_rf_master.predict_proba(x_train_pre)[:, 1] >= final_rf_th).astype(int), final_rf_master.predict_proba(x_train_pre)[:, 1])
p_va1, r_va1, f_va1, roc_va1, pr_va1, log_va1 = evaluate_model(y_valid, (final_rf_master.predict_proba(x_valid_pre)[:, 1] >= final_rf_th).astype(int), final_rf_master.predict_proba(x_valid_pre)[:, 1])
X_test_pre = preprocessor_pre.transform(X_test_ori)
p_te1, r_te1, f_te1, roc_te1, pr_te1, log_te1 = evaluate_model(y_test, (final_rf_master.predict_proba(X_test_pre)[:, 1] >= final_rf_th).astype(int), final_rf_master.predict_proba(X_test_pre)[:, 1])

rf_total_performance_df = pd.DataFrame([
    {"데이터셋 (Dataset)": "훈련 데이터 (Train_70%)", "Precision": round(p_tr1, 4), "Recall": round(r_tr1, 4), "F1-Score": round(f_tr1, 4), "ROC-AUC": round(roc_tr1, 4), "PR-AUC": round(pr_tr1, 4), "LogLoss": round(log_tr1, 4)},
    {"데이터셋 (Dataset)": "검증 데이터 (Valid_15%)", "Precision": round(p_va1, 4), "Recall": round(r_va1, 4), "F1-Score": round(f_va1, 4), "ROC-AUC": round(roc_va1, 4), "PR-AUC": round(pr_va1, 4), "LogLoss": round(log_va1, 4)},
    {"데이터셋 (Dataset)": "실전 데이터 (Test_15%)", "Precision": round(p_te1, 4), "Recall": round(r_te1, 4), "F1-Score": round(f_te1, 4), "ROC-AUC": round(roc_te1, 4), "PR-AUC": round(pr_te1, 4), "LogLoss": round(log_te1, 4)}
], index=['0', '1', '2'])
print("[최종 모델 성능]")
display(rf_total_performance_df)

final_rf_importances_pre = pd.DataFrame({'Feature': feature_names_pre, 'Importance': final_rf_master.feature_importances_.round(4)}).sort_values(by='Importance', ascending=False).head(10).reset_index(drop=True)
print("\n [최종 모델 기준] 랜덤 포레스트 변수 중요도 TOP 10")
display(final_rf_importances_pre)


# ==============================================================================
# 4단계: [피처 엔지니어링 반영 후] 랜덤 포레스트 성능 비교
# ==============================================================================
print("\n" + "="*40 + " [피처 엔지니어링 반영 후] 랜덤 포레스트 성능 " + "="*40)
rf_model_post = RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_split=10, class_weight=None, random_state=42, n_jobs=-1)
rf_model_post.fit(x_train_eng, y_train)

p_tr2, r_tr2, f_tr2, roc_tr2, pr_tr2, log_tr2 = evaluate_model(y_train, (rf_model_post.predict_proba(x_train_eng)[:, 1] >= final_rf_th).astype(int), rf_model_post.predict_proba(x_train_eng)[:, 1])
p_va2, r_va2, f_va2, roc_va2, pr_va2, log_va2 = evaluate_model(y_valid, (rf_model_post.predict_proba(x_valid_eng)[:, 1] >= final_rf_th).astype(int), rf_model_post.predict_proba(x_valid_eng)[:, 1])
p_te2, r_te2, f_te2, roc_te2, pr_te2, log_te2 = evaluate_model(y_test, (rf_model_post.predict_proba(x_test_eng)[:, 1] >= final_rf_th).astype(int), rf_model_post.predict_proba(x_test_eng)[:, 1])

feature_eng_perf_df = pd.DataFrame([
    {"데이터셋 (Dataset)": "훈련 데이터 (Train_70%)", "Precision": round(p_tr2, 4), "Recall": round(r_tr2, 4), "F1-Score": round(f_tr2, 4), "ROC-AUC": round(roc_tr2, 4), "PR-AUC": round(pr_tr2, 4), "LogLoss": round(log_tr2, 4)},
    {"데이터셋 (Dataset)": "검증 데이터 (Valid_15%)", "Precision": round(p_va2, 4), "Recall": round(r_va2, 4), "F1-Score": round(f_va2, 4), "ROC-AUC": round(roc_va2, 4), "PR-AUC": round(pr_va2, 4), "LogLoss": round(log_va2, 4)},
    {"데이터셋 (Dataset)": "실전 데이터 (Test_15%)", "Precision": round(p_te2, 4), "Recall": round(r_te2, 4), "F1-Score": round(f_te2, 4), "ROC-AUC": round(roc_te2, 4), "PR-AUC": round(pr_te2, 4), "LogLoss": round(log_te2, 4)}
])
print("[피처 엔지니어링 반영 후] 랜덤 포레스트 성능")
display(feature_eng_perf_df)

feature_names_eng = list(preprocessor_eng.get_feature_names_out())
coef_df = pd.DataFrame({"Feature": feature_names_eng, "Importance": rf_model_post.feature_importances_.round(4)})
print("\n [피처 엔지니어링 반영 후] 랜덤 포레스트 변수 중요도 TOP 10")
display(coef_df.sort_values(by="Importance", ascending=False).head(10).reset_index(drop=True))