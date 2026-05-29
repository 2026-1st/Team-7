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
    
    # 피처 엔지니어링
    # 1) 워라밸 파괴 지수 (야근 여부 수치화 x 출퇴근 거리)
    df['OverTime_Numeric'] = df['OverTime'].apply(lambda x: 1 if x == 'Yes' else 0)
    df['Overwork_Fatigue_Index'] = df['OverTime_Numeric'] * df['DistanceFromHome']
    
    # 2) 체감 보상 불공정 지수 (총 겨력 대비 월 소득 비율, 분모 0 방지)
    df['Income_Per_WorkingYear'] = df['MonthlyIncome'] / (df['TotalWorkingYears'] + 1)
    
    # 3) 종합 조직 안착도 점수 ( 3대 만족도 지표 합산)
    df['Total_Satisfaction_Score'] = (df['EnvironmentSatisfaction'] + df['JobSatisfaction'] + df['RelationshipSatisfaction'])
    
    
    X = df.drop(target_col, axis=1)
    # Target이 'Yes'/'No'라면 1/0으로 변환
    y = df[target_col].apply(lambda x: 1 if x == 'Yes' else 0) if df[target_col].dtype == 'object' else df[target_col]

    # 2. 공정성을 위한 고정된 분할 (70:15:15)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    # 3. 변수 타입별 분류
    categorical_cols = ['BusinessTravel', 'Department','EducationField','Gender', 'JobRole', 'MaritalStatus', 'OverTime', ]
    numerical_cols = ['Age', 'DailyRate', 'DistanceFromHome', 'Education','EnvironmentSatisfaction','HourlyRate','JobInvolvement','JobLevel','JobSatisfaction','MonthlyIncome', 'MonthlyRate', 'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager','Overwork_Fatigue_Index', 'Income_Per_WorkingYear', 'Total_Satisfaction_Score']


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

## 로지스틱 회귀 학습(pr-auc 기준)
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression(
    penalty="l2",
    C=1.0,
    class_weight=None,
    solver="lbfgs",
    max_iter=3000,
    random_state=42
)
lr_model.fit(x_train, y_train)

threshold = 0.25

# [Train 데이터셋 채점]
y_proba_train = lr_model.predict_proba(x_train)[:, 1]
y_pred_train = (y_proba_train >= threshold).astype(int)
p_tr, r_tr, f_tr, roc_tr, pr_tr, log_tr = evaluate_model(y_train, y_pred_train, y_proba_train)

# [Valid 데이터셋 채점]
y_proba_valid = lr_model.predict_proba(x_valid)[:, 1]
y_pred_valid = (y_proba_valid >= threshold).astype(int)
p_va, r_va, f_va, roc_va, pr_va, log_va = evaluate_model(y_valid, y_pred_valid, y_proba_valid)

# [Test 데이터셋 채점]
y_proba_test = lr_model.predict_proba(x_test)[:, 1]
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

print("[피처 엔지니어링 반영 후] 로지스틱 회귀 성능")
display(feature_eng_perf_df)


# 변수 가중치 TOP 10 확인
feature_names_eng = list(preprocessor.get_feature_names_out())
coef_df = pd.DataFrame({"Feature": feature_names_eng, "Coefficient": lr_model.coef_[0].round(4)})

print("\n [피처 엔지니어링 반영 후] 로지스틱 회귀 변수 가중치 TOP 10")
display(coef_df.sort_values(by="Coefficient", ascending=False).head(10))