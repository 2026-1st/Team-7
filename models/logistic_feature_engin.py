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
                      'JobInvolvement','JobLevel','JobSatisfaction','MonthlyIncome', 'MonthlyRate','NumCompaniesWorked', 'PercentSalaryHike', 
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

# 피처 엔지니어링이 반영된 셋 로드
x_train_eng, x_valid_eng, x_test_eng, y_train, y_valid, y_test, preprocessor_eng = get_hr_data("data/data_team7.csv", model_type='linear_xgb')


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
# 그리드 서치 및 지표별 최적 가중치 비교 (피처 엔지니어링 전 원본 데이터 기준)
# ==============================================================================
print("\n" + "="*35 + " [그리드 서치 및 지표별 최적 모델 탐색] " + "="*35)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# 원본용 전처리 별도 정의
categorical_cols_pre = ['BusinessTravel', 'Department','EducationField','Gender', 'JobRole', 'MaritalStatus', 'OverTime']
numerical_cols_pre = ['Age', 'DailyRate', 'DistanceFromHome', 'Education','EnvironmentSatisfaction','HourlyRate','JobInvolvement','JobLevel','JobSatisfaction','MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']

preprocessor_pre = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols_pre),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), categorical_cols_pre)
    ]
)

df_ori = pd.read_csv("data/data_team7.csv")
y_ori = df_ori['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
X_ori = df_ori.drop('Attrition', axis=1)

# 70:15:15로 정확하게 순차 분할을 수행
X_train_ori, X_temp_ori, y_train, y_temp = train_test_split(X_ori, y_ori, test_size=0.3, stratify=y_ori, random_state=42)
X_valid_ori, X_test_ori, y_valid, y_test = train_test_split(X_temp_ori, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

x_train_pre = preprocessor_pre.fit_transform(X_train_ori)
x_valid_pre = preprocessor_pre.transform(X_valid_ori)

param_grid = [
    {'penalty': ['l1', 'l2'], 'C': [0.1, 1.0, 10.0, 100.0], 'class_weight': [None, 'balanced'], 'solver': ['liblinear', 'saga']},
    {'penalty': ['l2'], 'C': [0.1, 1.0, 10.0, 100.0], 'class_weight': [None, 'balanced'], 'solver': ['lbfgs']}
]

base_model = LogisticRegression(max_iter=3000, random_state=42)
scoring_metrics = {
    'precision': 'precision', 'recall': 'recall', 'f1': 'f1', 
    'roc_auc': 'roc_auc', 'pr_auc': 'average_precision', 'logloss': 'neg_log_loss'
}

grid_search = GridSearchCV(estimator=base_model, param_grid=param_grid, scoring=scoring_metrics, refit=False, cv=5, n_jobs=-1)
grid_search.fit(x_train_pre, y_train)

cv_results = pd.DataFrame(grid_search.cv_results_)
report_df = pd.DataFrame({
    'Class_Weight': cv_results['param_class_weight'], 'Penalty': cv_results['param_penalty'],
    'C': cv_results['param_C'], 'Solver': cv_results['param_solver'],
    'Precision': cv_results['mean_test_precision'].round(4), 'Recall': cv_results['mean_test_recall'].round(4),
    'F1-Score': cv_results['mean_test_f1'].round(4), 'ROC-AUC': cv_results['mean_test_roc_auc'].round(4),
    'PR-AUC': cv_results['mean_test_pr_auc'].round(4), 'LogLoss': (cv_results['mean_test_logloss'] * -1).round(4)
})

print("\n [순정 데이터] PR-AUC 높은 순 TOP 5 조합")
display(report_df.sort_values(by='PR-AUC', ascending=False).head(5))

best_comb_pre = {"class_weight": None, "penalty": "l2", "solver": "lbfgs", "C": 1.0}

final_master_model_pre = LogisticRegression(
    penalty=best_comb_pre["penalty"], 
    C=best_comb_pre["C"], 
    class_weight=best_comb_pre["class_weight"], 
    solver=best_comb_pre["solver"], 
    max_iter=3000, 
    random_state=42
)
final_master_model_pre.fit(x_train_pre, y_train)

# 검증 데이터 셋 채점
y_pred_val = final_master_model_pre.predict(x_valid_pre)
y_proba_val = final_master_model_pre.predict_proba(x_valid_pre)[:, 1]
precision, recall, f1, roc_auc, pr_auc, logloss = evaluate_model(y_valid, y_pred_val, y_proba_val)

valid_results = [{
    "Model": "LR_PR_AUC_Top_Model", "Precision": round(precision, 4), "Recall": round(recall, 4),
    "F1-Score": round(f1, 4), "ROC-AUC": round(roc_auc, 4), "PR-AUC": round(pr_auc, 4), "LogLoss": round(logloss, 4)
}]

feature_names_pre = list(preprocessor_pre.get_feature_names_out())
coef_df_pre = pd.DataFrame({
    "Feature": feature_names_pre, 
    "Coefficient": final_master_model_pre.coef_[0].round(4)
})

print("\n 기준 원본 데이터 변수 영향력(Coefficient) TOP 10")
display(coef_df_pre.sort_values(by="Coefficient", ascending=False).head(10))

print("\n[최적 조합의 검증 데이터 결과]")
display(pd.DataFrame(valid_results))

print("\n[각 최적 조합별 검증 결과]")
display(pd.DataFrame(valid_results))


# ==============================================================================
# [최종 모델 성능] 데이터 분할별 로지스틱 분류 성능 비교 및 변수 가중치 TOP 10
# ==============================================================================
print("\n" + "="*40 + " [최종 모델 성능] 데이터 분할별 로지스틱 분류 성능 비교 " + "="*40)
final_master_model = LogisticRegression(penalty="l2", C=1.0, class_weight=None, solver="lbfgs", max_iter=3000, random_state=42)
final_master_model.fit(x_train_pre, y_train)

final_th = 0.5

p_tr1, r_tr1, f_tr1, roc_tr1, pr_tr1, log_tr1 = evaluate_model(y_train, (final_master_model.predict_proba(x_train_pre)[:, 1] >= final_th).astype(int), final_master_model.predict_proba(x_train_pre)[:, 1])
p_va1, r_va1, f_va1, roc_va1, pr_va1, log_va1 = evaluate_model(y_valid, (final_master_model.predict_proba(x_valid_pre)[:, 1] >= final_th).astype(int), final_master_model.predict_proba(x_valid_pre)[:, 1])
X_test_pre = preprocessor_pre.transform(X_test_ori)
p_te1, r_te1, f_te1, roc_te1, pr_te1, log_te1 = evaluate_model(y_test, (final_master_model.predict_proba(X_test_pre)[:, 1] >= final_th).astype(int), final_master_model.predict_proba(X_test_pre)[:, 1])

total_performance_df = pd.DataFrame([
    {"데이터셋 (Dataset)": "훈련 데이터 (Train_70%)", "Precision": round(p_tr1, 4), "Recall": round(r_tr1, 4), "F1-Score": round(f_tr1, 4), "ROC-AUC": round(roc_tr1, 4), "PR-AUC": round(pr_tr1, 4), "LogLoss": round(log_tr1, 4)},
    {"데이터셋 (Dataset)": "검증 데이터 (Valid_15%)", "Precision": round(p_va1, 4), "Recall": round(r_va1, 4), "F1-Score": round(f_va1, 4), "ROC-AUC": round(roc_va1, 4), "PR-AUC": round(pr_va1, 4), "LogLoss": round(log_va1, 4)},
    {"데이터셋 (Dataset)": "실전 데이터 (Test_15%)", "Precision": round(p_te1, 4), "Recall": round(r_te1, 4), "F1-Score": round(f_te1, 4), "ROC-AUC": round(roc_te1, 4), "PR-AUC": round(pr_te1, 4), "LogLoss": round(log_te1, 4)}
], index=['0', '1', '2'])
display(total_performance_df)

final_coef_df = pd.DataFrame({'인사 요인 변수 (Feature)': feature_names_pre, '퇴사 유발 영향력 (Coefficient)': final_master_model.coef_[0].round(4)}).sort_values(by='퇴사 유발 영향력 (Coefficient)', ascending=False).head(10).reset_index(drop=True)
print("\n [최종 모델 기준] 퇴사 유발 핵심 요인 TOP 10")
display(final_coef_df)


# ==============================================================================
# [피처 엔지니어링 반영 후] 로지스틱 회귀 성능 비교 및 변수 가중치 TOP 10
# ==============================================================================
print("\n" + "="*40 + " [피처 엔지니어링 반영 후] 로지스틱 회귀 성능 " + "="*40)
lr_model_post = LogisticRegression(penalty="l2", C=1.0, class_weight=None, solver="lbfgs", max_iter=3000, random_state=42)
lr_model_post.fit(x_train_eng, y_train)

p_tr2, r_tr2, f_tr2, roc_tr2, pr_tr2, log_tr2 = evaluate_model(y_train, (lr_model_post.predict_proba(x_train_eng)[:, 1] >= final_th).astype(int), lr_model_post.predict_proba(x_train_eng)[:, 1])
p_va2, r_va2, f_va2, roc_va2, pr_va2, log_va2 = evaluate_model(y_valid, (lr_model_post.predict_proba(x_valid_eng)[:, 1] >= final_th).astype(int), lr_model_post.predict_proba(x_valid_eng)[:, 1])
p_te2, r_te2, f_te2, roc_te2, pr_te2, log_te2 = evaluate_model(y_test, (lr_model_post.predict_proba(x_test_eng)[:, 1] >= final_th).astype(int), lr_model_post.predict_proba(x_test_eng)[:, 1])

feature_eng_perf_df = pd.DataFrame([
    {"데이터셋 (Dataset)": "훈련 데이터 (Train_70%)", "Precision": round(p_tr2, 4), "Recall": round(r_tr2, 4), "F1-Score": round(f_tr2, 4), "ROC-AUC": round(roc_tr2, 4), "PR-AUC": round(pr_tr2, 4), "LogLoss": round(log_tr2, 4)},
    {"데이터셋 (Dataset)": "검증 데이터 (Valid_15%)", "Precision": round(p_va2, 4), "Recall": round(r_va2, 4), "F1-Score": round(f_va2, 4), "ROC-AUC": round(roc_va2, 4), "PR-AUC": round(pr_va2, 4), "LogLoss": round(log_va2, 4)},
    {"데이터셋 (Dataset)": "실전 데이터 (Test_15%)", "Precision": round(p_te2, 4), "Recall": round(r_te2, 4), "F1-Score": round(f_te2, 4), "ROC-AUC": round(roc_te2, 4), "PR-AUC": round(pr_te2, 4), "LogLoss": round(log_te2, 4)}
])
display(feature_eng_perf_df)

feature_names_eng = list(preprocessor_eng.get_feature_names_out())
coef_df = pd.DataFrame({"Feature": feature_names_eng, "Coefficient": lr_model_post.coef_[0].round(4)})
print("\n [피처 엔지니어링 반영 후] 로지스틱 회귀 변수 가중치 TOP 10")
display(coef_df.sort_values(by="Coefficient", ascending=False).head(10))


# ==============================================================================
# [피처 엔지니어링 반영 후] 하이퍼 파라미터 그리드 서치
# ==============================================================================
print("\n" + "="*30 + " [피처 엔지니어링 반영 후] 2차 그리드 서치 탐색 시작 " + "="*30)

# 1) 파생 변수가 포함된 x_train_eng 데이터를 기반으로 그리드 서치 수행
grid_search_post = GridSearchCV(
    estimator=LogisticRegression(max_iter=3000, random_state=42),
    param_grid=param_grid, # 앞서 정의한 l1, l2, C, solver 하이퍼파라미터 그대로 활용
    scoring=scoring_metrics,
    refit=False,
    cv=5,
    n_jobs=-1
)
grid_search_post.fit(x_train_eng, y_train)

# 2) 2차 그리드 서치 결과 정리 및 출력
cv_results_post = pd.DataFrame(grid_search_post.cv_results_)
report_post_df = pd.DataFrame({
    'Class_Weight': cv_results_post['param_class_weight'],
    'Penalty': cv_results_post['param_penalty'],
    'C': cv_results_post['param_C'],
    'Solver': cv_results_post['param_solver'],
    'Precision': cv_results_post['mean_test_precision'].round(4),
    'Recall': cv_results_post['mean_test_recall'].round(4),
    'F1-Score': cv_results_post['mean_test_f1'].round(4),
    'ROC-AUC': cv_results_post['mean_test_roc_auc'].round(4),
    'PR-AUC': cv_results_post['mean_test_pr_auc'].round(4),
    'LogLoss': (cv_results_post['mean_test_logloss'] * -1).round(4)
})

# 피처 엔지니어링 데이터 기준 PR-AUC 대장 모델 조합 확인용 출력
print("\n [피처 엔지니어링 반영 후] PR-AUC 높은 순 TOP 5 조합 (새로운 최적 파라미터 후보)")
display(report_post_df.sort_values(by='PR-AUC', ascending=False).head(5))


# 3) 새로운 1등 하이퍼파라미터 조합으로 최종 모델 정의 및 학습
best_penalty_post = "l1"
best_c_post = 10.0
best_solver_post = "liblinear"
best_weight_post = None

lr_model_post = LogisticRegression(
    penalty=best_penalty_post, 
    C=best_c_post, 
    class_weight=best_weight_post, 
    solver=best_solver_post, 
    max_iter=3000, 
    random_state=42
)
lr_model_post.fit(x_train_eng, y_train)


# 4) 임계값 0.5 기준으로 최종 스코어 산출
final_th = 0.5

p_tr2, r_tr2, f_tr2, roc_tr2, pr_tr2, log_tr2 = evaluate_model(y_train, (lr_model_post.predict_proba(x_train_eng)[:, 1] >= final_th).astype(int), lr_model_post.predict_proba(x_train_eng)[:, 1])
p_va2, r_va2, f_va2, roc_va2, pr_va2, log_va2 = evaluate_model(y_valid, (lr_model_post.predict_proba(x_valid_eng)[:, 1] >= final_th).astype(int), lr_model_post.predict_proba(x_valid_eng)[:, 1])
p_te2, r_te2, f_te2, roc_te2, pr_te2, log_te2 = evaluate_model(y_test, (lr_model_post.predict_proba(x_test_eng)[:, 1] >= final_th).astype(int), lr_model_post.predict_proba(x_test_eng)[:, 1])

feature_eng_perf_df = pd.DataFrame([
    {"데이터셋 (Dataset)": "훈련 데이터 (Train_70%)", "Precision": round(p_tr2, 4), "Recall": round(r_tr2, 4), "F1-Score": round(f_tr2, 4), "ROC-AUC": round(roc_tr2, 4), "PR-AUC": round(pr_tr2, 4), "LogLoss": round(log_tr2, 4)},
    {"데이터셋 (Dataset)": "검증 데이터 (Valid_15%)", "Precision": round(p_va2, 4), "Recall": round(r_va2, 4), "F1-Score": round(f_va2, 4), "ROC-AUC": round(roc_va2, 4), "PR-AUC": round(pr_va2, 4), "LogLoss": round(log_va2, 4)},
    {"데이터셋 (Dataset)": "실전 데이터 (Test_15%)", "Precision": round(p_te2, 4), "Recall": round(r_te2, 4), "F1-Score": round(f_te2, 4), "ROC-AUC": round(roc_te2, 4), "PR-AUC": round(pr_te2, 4), "LogLoss": round(log_te2, 4)}
])
print("\n[피처 엔지니어링 및 2차 튜닝 반영 후] 로지스틱 회귀 최종 성능")
display(feature_eng_perf_df)

# 5) 새롭게 정렬된 변수 가중치 TOP 10 확인
feature_names_eng = list(preprocessor_eng.get_feature_names_out())
coef_df = pd.DataFrame({"Feature": feature_names_eng, "Coefficient": lr_model_post.coef_[0].round(4)})
print("\n [피처 엔지니어링 및 2차 튜닝 반영 후] 로지스틱 회귀 변수 가중치 TOP 10")
display(coef_df.sort_values(by="Coefficient", ascending=False).head(10))


# ==============================================================================
# [교차 검증]
# ==============================================================================
print("\n" + "="*35 + " [최종 모델 5-Fold 교차 검증 안정성 평가] " + "="*35)
from sklearn.model_selection import cross_validate, StratifiedKFold

# 타겟 비율을 공정하게 유지하며 쪼개기 위한 전략
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 평가할 5대 지표
cv_scoring = {
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc',
    'pr_auc': 'average_precision',
    'logloss': 'neg_log_loss'
}

# 그리드 서치 및 최종 결정에 수렴한 최적 규격 모델 정의 (C=1.0, penalty='l2')
final_cv_model = LogisticRegression(penalty="l2", C=1.0, class_weight=None, solver="lbfgs", max_iter=3000, random_state=42)

# 1) 피처 엔지니어링 [전] 순정 데이터 기준의 최종 모델
cv_results_pre = cross_validate(final_cv_model, x_train_pre, y_train, cv=cv_strategy, scoring=cv_scoring, n_jobs=-1)

# 2) 피처 엔지니어링 [후] 파생 변수 데이터 기준의 최종 모델 
cv_results_post = cross_validate(lr_model_post, x_train_eng, y_train, cv=cv_strategy, scoring=cv_scoring, n_jobs=-1)

# 데이터프레임으로 최종 교차 검증 대조군 출력
cv_comparison_df = pd.DataFrame([
    {
        "최종 모델 검증 트랙 (5-Fold CV Avg)": "피처 엔지니어링 [전] 최종 모델",
        "Precision": round(np.mean(cv_results_pre['test_precision']), 4),
        "Recall": round(np.mean(cv_results_pre['test_recall']), 4),
        "F1-Score": round(np.mean(cv_results_pre['test_f1']), 4),
        "ROC-AUC": round(np.mean(cv_results_pre['test_roc_auc']), 4),
        "PR-AUC": round(np.mean(cv_results_pre['test_pr_auc']), 4),
        "LogLoss": round(-1 * np.mean(cv_results_pre['test_logloss']), 4)
    },
    {
        "최종 모델 검증 트랙 (5-Fold CV Avg)": "피처 엔지니어링 [후] 최종 모델",
        "Precision": round(np.mean(cv_results_post['test_precision']), 4),
        "Recall": round(np.mean(cv_results_post['test_recall']), 4),
        "F1-Score": round(np.mean(cv_results_post['test_f1']), 4),
        "ROC-AUC": round(np.mean(cv_results_post['test_roc_auc']), 4),
        "PR-AUC": round(np.mean(cv_results_post['test_pr_auc']), 4),
        "LogLoss": round(-1 * np.mean(cv_results_post['test_logloss']), 4)
    }
])

print("\n[최종 검증] 하이퍼파라미터 고정 후 피처 엔지니어링 전/후 교차 검증 비교")
display(cv_comparison_df)


# ==============================================================================
# [시각화]
# ==============================================================================
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve
from sklearn.metrics import precision_recall_curve, confusion_matrix
import os
import numpy as np

# 한글 깨짐 방지 및 마이너스 기호 깨짐 처리 
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ==============================================================================
# [데이터셋 최종 매칭 동기화 세팅]
# ==============================================================================
target_model = lr_model_post          # 2차 튜닝이 완료된 최종 l1, C=10.0 모델
X_train_target = x_train_eng          # 1, 4번용 훈련 데이터
y_train_target = y_train
X_test_target = x_test_eng            # 2, 3번용 실전 테스트 데이터 (Test_15%)
y_test_target = y_test                # 2, 3번용 실전 테스트 정답
feature_names_target = feature_names_eng
final_th = 0.5                        # 확정된 기본 임계값


# ==============================================================================
# 1. Learning_Curve
# ==============================================================================
train_sizes, train_scores, valid_scores = learning_curve(
    estimator=target_model,
    X=X_train_target,
    y=y_train_target,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,
    scoring='average_precision',  
    n_jobs=-1,
    random_state=42
)

train_mean = np.mean(train_scores, axis=1)
valid_mean = np.mean(valid_scores, axis=1)

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Train PR-AUC')
plt.plot(train_sizes, valid_mean, 'o-', color='green', label='Validation PR-AUC')
# Best 모델 빨간 점선 표시
plt.axvline(x=train_sizes[-1], color='red', linestyle='--', linewidth=1.5, label='Best Model Point')

plt.title('로지스틱 회귀 Learning Curve (PR-AUC)', fontsize=14, fontweight='bold')
plt.xlabel('훈련 데이터 크기 (Training Samples)', fontsize=11)
plt.ylabel('PR-AUC Score', fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='best')
plt.tight_layout()
plt.show()


# ==============================================================================
# 2. Precision-Recall_Curve
# ==============================================================================
y_proba_test = target_model.predict_proba(X_test_target)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test_target, y_proba_test)

plt.figure(figsize=(7, 5))
plt.plot(recalls, precisions, color='purple', linewidth=2, label='PR Curve (Test)')

plt.title('로지스틱 회귀 Precision-Recall Curve (Test)', fontsize=14, fontweight='bold')
plt.xlabel('Recall (재현율)', fontsize=11)
plt.ylabel('Precision (정밀도)', fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()


# ==============================================================================
# 3. Confusion_Matrix
# ==============================================================================
y_pred_test_custom = (y_proba_test >= final_th).astype(int)
cm = confusion_matrix(y_test_target, y_pred_test_custom)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Stay (잔류)', 'Leave (퇴사)'],
            yticklabels=['Stay (잔류)', 'Leave (퇴사)'],
            annot_kws={'size': 14, 'weight': 'bold'})

plt.title(f'로지스틱 회귀 Confusion Matrix (Test | Th: {final_th})', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label (예측값)', fontsize=12)
plt.ylabel('True Label (실제값)', fontsize=12)
plt.tight_layout()
plt.show()


# ==============================================================================
# 4. Feature_Importance
# ==============================================================================
coef_values = target_model.coef_[0]
importance_df = pd.DataFrame({
    'Feature': feature_names_target,
    'Importance': np.abs(coef_values),  # 영향력의 순수한 크기 대조를 위해 절대값 변환
    'Original_Coef': coef_values        # 부호(+/-) 식별용 원본 값 보존
})

# 상위 20개 핵심 인사 변수 정렬
importance_df = importance_df.sort_values(by='Importance', ascending=False).head(20)

plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis', hue='Feature', legend=False)

# 바 차트 우측에 텍스트로 양수/음수 방향성이 포함된 가중치 값 명시
for index, row in enumerate(importance_df.itertuples()):
    plt.text(row.Importance + 0.02, index, f" {row.Original_Coef:+.4f}", va='center', fontsize=10, fontweight='bold')

plt.title('로지스틱 회귀 변수 중요도 (Coefficient Absolute Value - Top 20)', fontsize=14, fontweight='bold')
plt.xlabel('영향력 크기 (Absolute Coefficient)', fontsize=12)
plt.ylabel('인사 요인 변수 (Features)', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.4, axis='x')
plt.tight_layout()
plt.show()
