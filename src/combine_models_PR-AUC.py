import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.metrics import precision_recall_curve, average_precision_score

# 모델 라이브러리
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.metrics import Metric

plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# 1. 피처 엔지니어링 및 데이터 로드
# ==========================================
def apply_advanced_feature_engineering(df):
    df_new = df.copy()
    df_new['Total_Satisfaction_Score'] = (
        df_new['EnvironmentSatisfaction'] + df_new['JobSatisfaction'] +
        df_new['RelationshipSatisfaction'] + df_new['WorkLifeBalance']
    )
    df_new['Income_Per_WorkingYear'] = df_new['MonthlyIncome'] / (df_new['TotalWorkingYears'] + 1)
    df_new['Income_Per_YearAtCompany'] = df_new['MonthlyIncome'] / (df_new['YearsAtCompany'] + 1)
    df_new['Income_Per_Level'] = df_new['MonthlyIncome'] / df_new['JobLevel']
    df_new['Cost_Effectiveness'] = df_new['PercentSalaryHike'] / df_new['PerformanceRating']
    is_overtime = df_new['OverTime'].apply(lambda x: 1 if x == 'Yes' or x == 1 else 0)
    wlb_reversed = 5 - df_new['WorkLifeBalance']
    df_new['Burnout_Risk'] = is_overtime + wlb_reversed
    df_new['Sat_WLB_Interaction'] = df_new['Total_Satisfaction_Score'] * df_new['WorkLifeBalance']
    df_new['Promotion_Speed_Index'] = df_new['JobLevel'] / (df_new['TotalWorkingYears'] + 1)
    df_new['Stagnation_Index'] = df_new['YearsSinceLastPromotion'] / (df_new['YearsAtCompany'] + 1)
    df_new['Job_Hopping_Index'] = df_new['NumCompaniesWorked'] / (df_new['TotalWorkingYears'] + 1)
    df_new['Loyalty_Ratio'] = np.where(
        df_new['TotalWorkingYears'] > 0, df_new['YearsAtCompany'] / df_new['TotalWorkingYears'], 0
    )
    return df_new

# 데이터 로드 (경로는 환경에 맞게 수정)
df = pd.read_csv("/content/drive/MyDrive/data_team7.csv")
target_col = 'Attrition'

df_eng = apply_advanced_feature_engineering(df)

X = df_eng.drop(target_col, axis=1)
y = df_eng[target_col].apply(lambda x: 1 if x == 'Yes' else 0) if df_eng[target_col].dtype == 'object' else df_eng[target_col]

# 공정성을 위한 고정 분할 (70:15:15)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# 변수 분류
categorical_cols = ['BusinessTravel', 'Department','EducationField','Gender', 'JobRole', 'MaritalStatus', 'OverTime']
numerical_cols = ['Age', 'DailyRate', 'DistanceFromHome', 'Education','EnvironmentSatisfaction','HourlyRate',
                  'JobInvolvement','JobLevel','JobSatisfaction','MonthlyIncome', 'MonthlyRate','NumCompaniesWorked',
                  'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel',
                  'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
                  'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager', 'Total_Satisfaction_Score',
                  'Income_Per_WorkingYear', 'Income_Per_YearAtCompany', 'Income_Per_Level', 'Cost_Effectiveness',
                  'Burnout_Risk', 'Sat_WLB_Interaction', 'Promotion_Speed_Index', 'Stagnation_Index',
                  'Job_Hopping_Index', 'Loyalty_Ratio']

# ==========================================
# 2. 데이터 전처리 (모델 타입별)
# ==========================================

preprocessor_ohe = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), categorical_cols)
    ]
)

X_train_ohe = preprocessor_ohe.fit_transform(X_train)
X_test_ohe = preprocessor_ohe.transform(X_test)

preprocessor_ord = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_cols)
    ]
)

X_train_ord = preprocessor_ord.fit_transform(X_train)
X_valid_ord = preprocessor_ord.transform(X_valid)
X_test_ord = preprocessor_ord.transform(X_test)

cat_idxs = list(range(len(numerical_cols), len(numerical_cols) + len(categorical_cols)))
cat_dims = [len(X[col].unique()) for col in categorical_cols]


# ==========================================
# 3. 모델 학습 및 예측 확률 추출
# ==========================================
models_proba = {}
vanilla_lr_model = LogisticRegression(max_iter=3000, random_state=42)
vanilla_lr_model.fit(X_train_ohe, y_train)
models_proba['Vanilla LR (Baseline)'] = vanilla_lr_model.predict_proba(X_test_ohe)[:, 1]
print("1. Logistic Regression 학습 중...")

lr_model = LogisticRegression(
    penalty="l1", C=10.0, class_weight=None, solver="liblinear", max_iter=3000, random_state=42
)
lr_model.fit(X_train_ohe, y_train)
models_proba['Logistic Regression'] = lr_model.predict_proba(X_test_ohe)[:, 1]

print("2. XGBoost 학습 중...")
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic', eval_metric='auc', max_depth=4,
    learning_rate=0.05, n_estimators=300, scale_pos_weight=pos_weight, random_state=42
)
xgb_model.fit(X_train_ohe, y_train)
models_proba['XGBoost'] = xgb_model.predict_proba(X_test_ohe)[:, 1]

print("3. Random Forest 학습 중...")

rf_model = RandomForestClassifier(
    n_estimators=300, max_depth=None, min_samples_split=2, class_weight=None, random_state=42, n_jobs=-1
)
rf_model.fit(X_train_ord, y_train)
models_proba['Random Forest'] = rf_model.predict_proba(X_test_ord)[:, 1]

print("4. LightGBM 학습 중...")
lgbm_params = {
    "objective": "binary",
    "random_state": 42,
    "scale_pos_weight": 5.198795180722891,
    "n_estimators": 100,
    "min_child_samples": 10,
    "verbose": -1,
    "learning_rate": 0.15357603065597958,
    "max_depth": 7,
    "num_leaves": 9,
    "subsample": 0.8446177646084269,
    "colsample_bytree": 0.7318391795545601,
    "reg_alpha": 2.500882252228825,
    "reg_lambda": 4.121877798220284
}
lgb_model = lgb.LGBMClassifier(**lgbm_params)
lgb_model.fit(X_train_ord, y_train)
models_proba['LightGBM'] = lgb_model.predict_proba(X_test_ord)[:, 1]
print("5. TabNet 학습 중...")
class PRAUCMetric(Metric):
    def __init__(self):
        self._name = "pr_auc"
        self._maximize = True
    def __call__(self, y_true, y_score):
        preds_proba_1 = y_score[:, 1]
        return average_precision_score(y_true, preds_proba_1)

tabnet_model = TabNetClassifier(
    n_d=16, n_a=16, n_steps=2, gamma=1.5643739715662777, lambda_sparse=0.0014106348309282917,
    cat_idxs=cat_idxs, cat_dims=cat_dims, cat_emb_dim=2,
    optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=0.049689515723358134),
    scheduler_fn=torch.optim.lr_scheduler.StepLR, scheduler_params={"step_size":10, "gamma":0.9},
    mask_type='sparsemax', device_name='auto', verbose=0
)
tabnet_model.fit(
    X_train=X_train_ord, y_train=y_train.values,
    eval_set=[(X_train_ord, y_train.values), (X_valid_ord, y_valid.values)],
    eval_name=['train', 'valid'], eval_metric=[PRAUCMetric],
    max_epochs=100, patience=100, batch_size=256, virtual_batch_size=128, num_workers=0, drop_last=False
)
models_proba['TabNet'] = tabnet_model.predict_proba(X_test_ord)[:, 1]


# ==========================================
# 4. PR-AUC 단일 그래프 시각화
# ==========================================
print("\n모든 모델 학습 완료. PR-AUC 그래프를 그립니다.")
plt.figure(figsize=(11, 8))

colors = ['#8c564b', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# 모델별 PR-AUC Curve 그리기
for color, (name, proba) in zip(colors, models_proba.items()):
    precisions, recalls, _ = precision_recall_curve(y_test, proba)
    pr_auc_score = average_precision_score(y_test, proba)
    if "Baseline" in name:
        plt.plot(recalls, precisions, lw=2.5, linestyle='--', color=color, label=f'{name} (PR-AUC = {pr_auc_score:.4f})')
    else:
        plt.plot(recalls, precisions, lw=2.5, color=color, label=f'{name} (PR-AUC = {pr_auc_score:.4f})')

plt.title("Precision-Recall Curve Comparison", fontsize=16, fontweight='bold')
plt.xlabel("Recall", fontsize=13)
plt.ylabel("Precision", fontsize=13)
plt.legend(loc='upper right', fontsize=11, frameon=True, shadow=True)
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()

plt.savefig('All_Models_PR_AUC_Curve.pdf', bbox_inches='tight')
plt.show()