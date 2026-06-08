import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.feature_eng import add_hr_features


DROP_COLS = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]


def load_hr_dataframe(filepath, use_fe=False):
    """CSV를 읽고 불필요한 열 제거, 타깃 변환, 파생변수 생성을 처리합니다."""
    df = pd.read_csv(filepath)
    df = df.drop(columns=[col for col in DROP_COLS if col in df.columns])

    if use_fe:
        df = add_hr_features(df)

    df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0}).fillna(df["Attrition"]).astype(int)
    return df


def split_hr_data(df):
    """전체 데이터를 train/validation/test = 70/15/15 비율로 나눕니다."""
    X = df.drop(columns="Attrition")
    y = df["Attrition"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def _make_preprocessor(categorical_cols, numerical_cols, model_type="tree", scale_numeric=True):
    if model_type in ["linear", "linear_xgb"]:
        encoder = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
    else:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    numeric_transformer = StandardScaler() if scale_numeric else "passthrough"

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols),
            ("cat", encoder, categorical_cols),
        ]
    )


def _feature_names(preprocessor, categorical_cols, numerical_cols, model_type):
    if model_type in ["linear", "linear_xgb"]:
        cat_names = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_cols)
        return numerical_cols + cat_names.tolist()
    return numerical_cols + categorical_cols


def get_hr_data(filepath, model_type="tree"):
    """
    기존 팀 코드와 호환되는 공용 전처리 함수입니다.

    - 기본 데이터만 사용합니다.
    - 수치형 변수에는 StandardScaler를 적용합니다.
    - 반환값은 기존과 동일하게 6개입니다.
    """
    df = load_hr_dataframe(filepath, use_fe=False)
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_hr_data(df)

    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_cols = X_train.select_dtypes(exclude=["object", "category"]).columns.tolist()

    preprocessor = _make_preprocessor(
        categorical_cols, numerical_cols, model_type=model_type, scale_numeric=True
    )

    X_train = preprocessor.fit_transform(X_train)
    X_valid = preprocessor.transform(X_valid)
    X_test = preprocessor.transform(X_test)

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def get_lgbm_data(filepath, use_fe=False, return_feature_names=False):
    """
    LightGBM 노트북 전용 전처리 함수입니다.

    LGBM은 트리 모델이라 수치형 스케일링을 생략하고, feature importance를 보기 쉽도록
    DataFrame 형태로 반환합니다.
    """
    df = load_hr_dataframe(filepath, use_fe=use_fe)
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_hr_data(df)

    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_cols = X_train.select_dtypes(exclude=["object", "category"]).columns.tolist()
    preprocessor = _make_preprocessor(
        categorical_cols, numerical_cols, model_type="tree", scale_numeric=False
    )

    X_train_encoded = preprocessor.fit_transform(X_train)
    X_valid_encoded = preprocessor.transform(X_valid)
    X_test_encoded = preprocessor.transform(X_test)
    feature_names = _feature_names(preprocessor, categorical_cols, numerical_cols, model_type="tree")

    X_train_encoded = pd.DataFrame(X_train_encoded, columns=feature_names, index=X_train.index)
    X_valid_encoded = pd.DataFrame(X_valid_encoded, columns=feature_names, index=X_valid.index)
    X_test_encoded = pd.DataFrame(X_test_encoded, columns=feature_names, index=X_test.index)

    result = (X_train_encoded, X_valid_encoded, X_test_encoded, y_train, y_valid, y_test)
    if return_feature_names:
        return (*result, feature_names)
    return result
