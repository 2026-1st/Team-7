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
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=['object', 'category']).columns.tolist()
    

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


# 베이스라인 로지스틱 회귀
from sklearn.linear_model import LogisticRegression

lr_base_model = LogisticRegression(max_iter=1000, random_state=42)
lr_base_model.fit(x_train_eng, y_train)


# 4) 임계값 0.5 기준으로 최종 스코어 산출
final_th = 0.5

p_tr2, r_tr2, f_tr2, roc_tr2, pr_tr2, log_tr2 = evaluate_model(y_train, (lr_base_model.predict_proba(x_train_eng)[:, 1] >= final_th).astype(int), lr_base_model.predict_proba(x_train_eng)[:, 1])
p_va2, r_va2, f_va2, roc_va2, pr_va2, log_va2 = evaluate_model(y_valid, (lr_base_model.predict_proba(x_valid_eng)[:, 1] >= final_th).astype(int), lr_base_model.predict_proba(x_valid_eng)[:, 1])
p_te2, r_te2, f_te2, roc_te2, pr_te2, log_te2 = evaluate_model(y_test, (lr_base_model.predict_proba(x_test_eng)[:, 1] >= final_th).astype(int), lr_base_model.predict_proba(x_test_eng)[:, 1])

feature_eng_perf_df = pd.DataFrame([
    {"데이터셋 (Dataset)": "훈련 데이터 (Train_70%)", "Precision": round(p_tr2, 4), "Recall": round(r_tr2, 4), "F1-Score": round(f_tr2, 4), "ROC-AUC": round(roc_tr2, 4), "PR-AUC": round(pr_tr2, 4), "LogLoss": round(log_tr2, 4)},
    {"데이터셋 (Dataset)": "검증 데이터 (Valid_15%)", "Precision": round(p_va2, 4), "Recall": round(r_va2, 4), "F1-Score": round(f_va2, 4), "ROC-AUC": round(roc_va2, 4), "PR-AUC": round(pr_va2, 4), "LogLoss": round(log_va2, 4)},
    {"데이터셋 (Dataset)": "실전 데이터 (Test_15%)", "Precision": round(p_te2, 4), "Recall": round(r_te2, 4), "F1-Score": round(f_te2, 4), "ROC-AUC": round(roc_te2, 4), "PR-AUC": round(pr_te2, 4), "LogLoss": round(log_te2, 4)}
])
print("\n[피처 엔지니어링 후] 로지스틱 회귀 성능")
display(feature_eng_perf_df)

# 5) 새롭게 정렬된 변수 가중치 TOP 10 확인
feature_names_eng = list(preprocessor_eng.get_feature_names_out())
coef_df = pd.DataFrame({"Feature": feature_names_eng, "Coefficient": lr_base_model.coef_[0].round(4)})
print("\n [피처 엔지니어링 후] 로지스틱 회귀 변수 가중치 TOP 10")
display(coef_df.sort_values(by="Coefficient", ascending=False).head(10))


# ==============================================================================
# [교차 검증]
# ==============================================================================
print("\n" + "="*35 + " [ 5-Fold 교차 검증 안정성 평가] " + "="*35)
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

# 피처 엔지니어링 [후] 파생 변수 데이터 기준의 최종 모델 
cv_results_post = cross_validate(lr_base_model, x_train_eng, y_train, cv=cv_strategy, scoring=cv_scoring, n_jobs=-1)

# 데이터프레임으로 최종 교차 검증 대조군 출력
cv_comparison_df = pd.DataFrame([
    {
        "최종 모델 검증 트랙 (5-Fold CV Avg)": "피처 엔지니어링 [후] 기본 모델",
        "Precision": round(np.mean(cv_results_post['test_precision']), 4),
        "Recall": round(np.mean(cv_results_post['test_recall']), 4),
        "F1-Score": round(np.mean(cv_results_post['test_f1']), 4),
        "ROC-AUC": round(np.mean(cv_results_post['test_roc_auc']), 4),
        "PR-AUC": round(np.mean(cv_results_post['test_pr_auc']), 4),
        "LogLoss": round(-1 * np.mean(cv_results_post['test_logloss']), 4)
    }
])

print("\n 교차 검증")
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
target_model = lr_base_model    
X_train_target = x_train_eng          
y_train_target = y_train
X_test_target = x_test_eng           
y_test_target = y_test               
feature_names_target = feature_names_eng
final_th = 0.5                       


# ==============================================================================
# Precision-Recall_Curve
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
# Confusion_Matrix
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
# Feature_Importance
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
