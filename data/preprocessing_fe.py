import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


RANDOM_STATE = 42
TARGET_COL = "Attrition"


BASE_CATEGORICAL_COLS = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "OverTime",
]

BASE_NUMERICAL_COLS = [
    "Age",
    "DailyRate",
    "DistanceFromHome",
    "Education",
    "EnvironmentSatisfaction",
    "HourlyRate",
    "JobInvolvement",
    "JobLevel",
    "JobSatisfaction",
    "MonthlyIncome",
    "MonthlyRate",
    "NumCompaniesWorked",  # 원본 변수로도 사용하고 Job_Hopping_Index 계산에도 사용합니다.
    "PercentSalaryHike",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "StockOptionLevel",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "WorkLifeBalance",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
]

FE_NUMERICAL_COLS = [
    "Income_Per_WorkingYear",
    "Income_Per_YearAtCompany",
    "Income_Per_Level",
    "Cost_Effectiveness",
    "OverTime_Binary",
    "Overwork_Fatigue_Index",
    "Burnout_Risk",
    "JobSat_WLB_Interaction",
    "Career_Velocity_Index",
    "Stagnation_Index",
    "Role_Stagnation_Index",
    "Total_Satisfaction_Score",
    "Job_Hopping_Index",
    "Organizational_Loyalty_Ratio",
]


def add_hr_feature_engineering(df):
    """HR Attrition 데이터에 설명 가능한 파생 피처를 추가합니다.

    원본 DataFrame을 직접 수정하지 않도록 copy를 만든 뒤 반환합니다.
    모든 ratio 피처는 분모가 0이 되는 것을 막기 위해 +1을 사용합니다.
    """
    df_fe = df.copy()

    # OverTime은 Yes/No 문자열이므로 상호작용 피처 계산을 위해 1/0으로 변환합니다.
    df_fe["OverTime_Binary"] = (df_fe["OverTime"] == "Yes").astype(int)

    # 1. 보상 관련 피처: 경력, 근속, 직급, 고과 대비 보상 수준을 표현합니다.
    df_fe["Income_Per_WorkingYear"] = df_fe["MonthlyIncome"] / (df_fe["TotalWorkingYears"] + 1)
    df_fe["Income_Per_YearAtCompany"] = df_fe["MonthlyIncome"] / (df_fe["YearsAtCompany"] + 1)
    df_fe["Income_Per_Level"] = df_fe["MonthlyIncome"] / (df_fe["JobLevel"] + 1)
    df_fe["Cost_Effectiveness"] = df_fe["PercentSalaryHike"] / (df_fe["PerformanceRating"] + 1)

    # 2. 번아웃 관련 피처: 야근, 통근 거리, 워라밸 저하가 겹치는 위험을 표현합니다.
    df_fe["Overwork_Fatigue_Index"] = df_fe["OverTime_Binary"] * df_fe["DistanceFromHome"]
    df_fe["Burnout_Risk"] = df_fe["OverTime_Binary"] + (5 - df_fe["WorkLifeBalance"])
    df_fe["JobSat_WLB_Interaction"] = df_fe["JobSatisfaction"] * df_fe["WorkLifeBalance"]

    # 3. 경력 정체 관련 피처: 경력 대비 직급, 승진 정체, 역할 정체를 표현합니다.
    df_fe["Career_Velocity_Index"] = df_fe["JobLevel"] / (df_fe["TotalWorkingYears"] + 1)
    df_fe["Stagnation_Index"] = df_fe["YearsSinceLastPromotion"] / (df_fe["YearsAtCompany"] + 1)
    df_fe["Role_Stagnation_Index"] = df_fe["YearsInCurrentRole"] / (df_fe["TotalWorkingYears"] + 1)

    # 4. 만족도 및 충성도 피처: 전반적 만족도와 현재 회사 근속 비중을 표현합니다.
    df_fe["Total_Satisfaction_Score"] = (
        df_fe["EnvironmentSatisfaction"]
        + df_fe["JobSatisfaction"]
        + df_fe["RelationshipSatisfaction"]
        + df_fe["WorkLifeBalance"]
    )

    # 5. 이직 성향 피처: 총 경력 대비 이전 직장 수로 직장 이동 성향을 표현합니다.
    df_fe["Job_Hopping_Index"] = df_fe["NumCompaniesWorked"] / (df_fe["TotalWorkingYears"] + 1)
    df_fe["Organizational_Loyalty_Ratio"] = df_fe["YearsAtCompany"] / (df_fe["TotalWorkingYears"] + 1)

    # 모든 행에서 값이 같아 예측에 도움이 되지 않는 컬럼은 있으면 제거합니다.
    noise_cols = ["EmployeeCount", "StandardHours", "Over18"]
    df_fe = df_fe.drop(columns=[col for col in noise_cols if col in df_fe.columns])

    return df_fe


def get_hr_data_fe(filepath, model_type="tree", return_feature_names=False):
    """공통 분할 방식에 FE를 추가한 전처리 함수입니다.

    model_type="linear_xgb"이면 원-핫 인코딩을 사용하고,
    그 외 모델은 기존 팀 코드처럼 오디널 인코딩을 사용합니다.
    """
    df = pd.read_csv(filepath)
    df = add_hr_feature_engineering(df)

    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL].apply(lambda x: 1 if x == "Yes" else 0) if df[TARGET_COL].dtype == "object" else df[TARGET_COL]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.3,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )

    categorical_cols = [col for col in BASE_CATEGORICAL_COLS if col in X.columns]
    numerical_cols = [col for col in BASE_NUMERICAL_COLS + FE_NUMERICAL_COLS if col in X.columns]

    if model_type == "linear_xgb":
        cat_transformer = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
    else:
        cat_transformer = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", cat_transformer, categorical_cols),
        ]
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_valid_processed = preprocessor.transform(X_valid)
    X_test_processed = preprocessor.transform(X_test)

    if return_feature_names:
        if model_type == "linear_xgb":
            feature_names = preprocessor.get_feature_names_out().tolist()
        else:
            feature_names = numerical_cols + categorical_cols

        return (
            X_train_processed,
            X_valid_processed,
            X_test_processed,
            y_train,
            y_valid,
            y_test,
            feature_names,
        )

    return X_train_processed, X_valid_processed, X_test_processed, y_train, y_valid, y_test
