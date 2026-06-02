import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


RANDOM_STATE = 42
TARGET_COL = "Attrition"


def get_hr_data(filepath, model_type="tree"):
    """HR Attrition 데이터를 공통 방식으로 분할하고 전처리합니다.

    기존 팀 공통 전처리 코드의 수정본입니다.
    차이점은 원본 데이터의 NumCompaniesWorked 변수를 numerical_cols에 포함했다는 점입니다.

    Parameters
    ----------
    filepath : str
        CSV 데이터 파일 경로입니다.
    model_type : str
        "linear_xgb"이면 범주형 변수에 OneHotEncoder를 사용합니다.
        그 외 값이면 LGBM/TabNet 등 트리 계열 실험용으로 OrdinalEncoder를 사용합니다.

    Returns
    -------
    X_train_processed, X_valid_processed, X_test_processed, y_train, y_valid, y_test
        70:15:15 stratified split 이후 전처리된 데이터입니다.
    """
    df = pd.read_csv(filepath)

    # 1. Feature / Target 분리
    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL].apply(lambda x: 1 if x == "Yes" else 0) if df[TARGET_COL].dtype == "object" else df[TARGET_COL]

    # 2. 팀 공통 기준인 70:15:15 stratified split을 유지합니다.
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

    categorical_cols = [
        "BusinessTravel",
        "Department",
        "EducationField",
        "Gender",
        "JobRole",
        "MaritalStatus",
        "OverTime",
    ]

    numerical_cols = [
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
        "NumCompaniesWorked",  # 기존 전처리 코드에서 빠져 있던 원본 변수입니다.
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

    # 데이터 파일에 없는 컬럼이 생겨도 바로 에러가 나지 않도록 실제 존재 컬럼만 사용합니다.
    categorical_cols = [col for col in categorical_cols if col in X.columns]
    numerical_cols = [col for col in numerical_cols if col in X.columns]

    if model_type == "linear_xgb":
        # 선형 모델/XGBoost용: 범주형 변수에 순서를 부여하지 않도록 원-핫 인코딩을 사용합니다.
        cat_transformer = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
    else:
        # LGBM/TabNet 실험용: 기존 팀 코드와 동일하게 오디널 인코딩을 사용합니다.
        cat_transformer = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", cat_transformer, categorical_cols),
        ]
    )

    # train set에만 fit하고 valid/test에는 transform만 적용해 데이터 누수를 막습니다.
    X_train_processed = preprocessor.fit_transform(X_train)
    X_valid_processed = preprocessor.transform(X_valid)
    X_test_processed = preprocessor.transform(X_test)

    return X_train_processed, X_valid_processed, X_test_processed, y_train, y_valid, y_test
