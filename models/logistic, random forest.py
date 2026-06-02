import pandas as pd
df = pd.read_csv("data_team7.csv")
print(df.head())
print(df.columns)
df.info()

## 0/1로 전처리
# 타겟 변수 (퇴사 여부)
df['Attrition'] = df['Attrition'].map({'Yes':1, 'No':0})
# 성별
df['Gender'] = df['Gender'].map({'Male':1, 'Female':0})
# 야근 여부
df['OverTime'] = df['OverTime'].map({'Yes':1, 'No':0})
# 해당 칼럼을 확인하여 잘 변경되었는지 확인
print(df[['Attrition', 'Gender', 'OverTime']].head())
df.info()

## one-hot encoding
# drop_first=True로 첫 번째 컬럼을 제거하여 다중공선성 방지
onehot_cols = ['MaritalStatus', 'EducationField', 'Department', 'JobRole', 'BusinessTravel']
df = pd.get_dummies(df, columns=onehot_cols, drop_first=True) 
# 모든 칼럼을 확인하여 잘 변경되었는지 확인
pd.set_option('display.max_columns', None)
print(df.head())
df.info()
# bool 타입을 int로 변환
bool_cols = df.select_dtypes('bool').columns
df[bool_cols] = df[bool_cols].astype(int)
df.info()

## 수치형 컬럼 정규화
from sklearn.preprocessing import StandardScaler
num_cols = ['Age', 'DistanceFromHome', 'MonthlyIncome', 'DailyRate', 'HourlyRate', 'MonthlyRate', 'PercentSalaryHike', 'TotalWorkingYears', 'NumCompaniesWorked', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager', 'TrainingTimesLastYear']

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
# 정규화 확인
print(df[num_cols].head())
print(df[num_cols].mean())
print(df[num_cols].std())


## 데이터 분리
from sklearn.model_selection import train_test_split
# 타겟 변수와 피처 분리
x = df.drop('Attrition', axis=1)
y = df['Attrition']

# 학습용과 테스트용 데이터로 분리(20%를 테스트용)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

## 평가지표 함수
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, log_loss)

# y_test : 실제 값, y_pred : 예측 결과 (0/1), y_proba : 예측 확률
def evaluate_model(y_test, y_pred, y_proba):
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    logloss = log_loss(y_test, y_proba)
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"Log Loss: {logloss:.4f}")
    
## 로지스틱 회귀
from sklearn.linear_model import LogisticRegression

# 모델 생성
lr = LogisticRegression(max_iter=1000, random_state=42)
# 모델 학습
lr.fit(x_train, y_train)
# 모델 예측
y_pred_lr = lr.predict(x_test)
y_proba_lr = lr.predict_proba(x_test)[:, 1] # predict_proba -> [[재직확률, 퇴사확률]] 이기에 퇴사확률만 가지고 오겠다는 것
# 평가
print('Logistic Regression Performance:')
evaluate_model(y_test, y_pred_lr, y_proba_lr)

## 랜덤 포레스트
from sklearn.ensemble import RandomForestClassifier

# 모델 생성
rf = RandomForestClassifier(random_state=42)
# 모델 학습
rf.fit(x_train, y_train)
# 모델 예측
y_pred_rf = rf.predict(x_test)
y_proba_rf = rf.predict_proba(x_test)[:, 1] 
# predict_proba -> [[재직확률, 퇴사확률]] 이기에 퇴사확률만 가지고 오겠다는 것

# 평가
print('Random Forest Performance:')
evaluate_model(y_test, y_pred_rf, y_proba_rf)