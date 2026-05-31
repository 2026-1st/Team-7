import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from IPython.display import display
import numpy as np
df = pd.read_csv("data/data_team7.csv")
print(df.head())
print(df.columns)
df.info()


## 피처 엔지니어링
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

## 전처리
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
                      'JobInvolvement','JobLevel','JobSatisfaction','MonthlyIncome', 'MonthlyRate', 'PercentSalaryHike', 
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

x_train, x_valid, x_test, y_train, y_valid, y_test, preprocessor = get_hr_data("data/data_team7.csv", model_type='tree')

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

## 랜덤 포레스트 학습(pr-auc 기준에서 튜닝한 것)
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=4,
    min_samples_split=10,
    class_weight=None,
    random_state=42,
    n_jobs=-1)

rf_model.fit(x_train, y_train)

threshold = 0.25

# [Train 데이터셋 채점]
y_proba_train = rf_model.predict_proba(x_train)[:, 1]
y_pred_train = (y_proba_train >= threshold).astype(int)
p_tr, r_tr, f_tr, roc_tr, pr_tr, log_tr = evaluate_model(y_train, y_pred_train, y_proba_train)

# [Valid 데이터셋 채점]
y_proba_valid = rf_model.predict_proba(x_valid)[:, 1]
y_pred_valid = (y_proba_valid >= threshold).astype(int)
p_va, r_va, f_va, roc_va, pr_va, log_va = evaluate_model(y_valid, y_pred_valid, y_proba_valid)

# [Test 데이터셋 채점]
y_proba_test = rf_model.predict_proba(x_test)[:, 1]
y_pred_test = (y_proba_test >= threshold).astype(int)
p_te, r_te, f_te, roc_te, pr_te, log_te = evaluate_model(y_test, y_pred_test, y_proba_test)

# 판다스 데이터프레임으로 피처 엔지니어링 후의 3대 성적 통합
feature_eng_perf_df = pd.DataFrame([
    {
        "데이터셋 (Dataset)": "훈련 데이터 (Train_70%)",
        "Precision": round(p_tr, 4), "Recall": round(r_tr, 4), "F1-Score": round(f_tr, 4),
        "ROC-AUC": round(roc_tr, 4), "PR-AUC": round(pr_tr, 4), "LogLoss": round(log_tr, 4)
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

print("[피처 엔지니어링 반영 후] 랜덤 포레스트 성능")
display(feature_eng_perf_df)


# 변수 가중치 TOP 10 확인
feature_names_eng = list(preprocessor.get_feature_names_out())
coef_df = pd.DataFrame({"Feature": feature_names_eng, "Importance": rf_model.feature_importances_.round(4)})

print("\n [피처 엔지니어링 반영 후] 랜덤 포레스트 변수 가중치 TOP 10")
display(coef_df.sort_values(by="Importance", ascending=False).head(10))