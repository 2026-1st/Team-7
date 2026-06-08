# Team 7 IBM HR Analytics 퇴사자 예측

## 1. 프로젝트 개요 (Project Overview)
**주제:** IBM HR Analytics 데이터셋을 활용한 직무 이탈(Attrition) 예측 및 핵심 유발 요인 분석
IBM HR 데이터셋 속 직원들의 다양한 정보들 가운데 퇴사 결정에 큰 영향을 미치는 원인을 찾는 것을 목표로 합니다. 이를 위해 Logistic Regression, Random Forest, LightGBM, XGBoost와 TabNet 모델들을 학습시키고 그 결과를 비교하여 의사 결정에 활용되기 가장 적합한 모델을 찾습니다. 모델 성능을 비교하기 위해 PR-AUC, ROC-AUC, Recall, f-1 score, log loss와 Confusion Matrix를 평가 지표로 결정했습니다. 

## 2. 리포지토리 구조 (Repository Structure)
```text
📦 Team-7
 ┣ 📂 Experiments (EDA)
 ┃ ┣ 📜 퇴사 요인 분석1.pdf
 ┃ ┣ 📜 퇴사 요인 분석2.pdf
 ┃ ┣ 📜 퇴사 요인 분석3.pdf
 ┃ ┣ 📜 EDA.py
 ┃ ┣ 📜 불균형 확인.pdf
 ┃ ┗ 📜 상관관계 행렬.pdf
 ┣ 📂 data
 ┃ ┣ 📜 raw_data_team7.csv (원본 데이터)
 ┃ ┗ 📜 data_team7.csv (무의미한 열 제거 완료된 데이터)
 ┣ 📂 models
 ┃ ┣ 📂 LightGBM
 ┃ ┃ ┗ 📜 LightGBM.ipynb
 ┃ ┣ 📂 Logistic
 ┃ ┃ ┗ 📜 Logistic_Regression.py
 ┃ ┣ 📂 RandomForest
 ┃ ┃ ┗ 📜 Random_Forest.py
 ┃ ┣ 📂 TabNet
 ┃ ┃ ┗ 📜 TabNet_.ipynb
 ┃ ┣ 📂 XGBoost
 ┃ ┃ ┗ 📜 XGBoost.ipynb
 ┣ 📂 results
 ┃ ┣ 📂 logistic_experiments
 ┃ ┣ 📂 randomforest_experiments
 ┃ ┣ 📂 lgbm_experiments
 ┃ ┣ 📂 xgboost_experiments
 ┃ ┗ 📂 tabnet_experiments
 ┣ 📂 src
 ┃ ┣ 📜 preprocessing_code.py 
 ┃ ┗ 📜 feature_eng.py 
 ┗ 📜 README.md
```
## 3. 실행 방법과 환경 (How to Run & Environment Notes)
**실행 방법:** models 폴더 속 LightGBM, 로지스틱 회귀와 랜덤 포레스트는 로컬 환경에서 사용되었던 코드이기에 데이터 경로를 클론한 파일 위치를 기반으로 변경하고 사용해야 합니다. 

XGBoost와 TabNet은 Colab 상에서 사용된 코드이기에 Colab에서 노트북을 여신 후 셀 단위로 하나씩 실행하시면 재현 가능합니다.

**실행 환경:** Python 3.12, scikit-learn 1.6.1, Pytorch 2.11.0, Optuna 4.9.0

## 4. 실험 재현성 및 공정성 (Reproducibility & Fairness)
랜덤 시드 관리: 모든 데이터 분할(train_test_split)과 모델 초기화 과정에서 random_state=42를 사용하여 재현성을 보장합니다.

평가 지표 통일: 하이퍼파라미터 튜닝(Optuna 등)의 목적 함수를 PR-AUC로 통일했습니다.

데이터 누수 방지: K-Fold나 Hold-out 분할 이전에 오버샘플링(SMOTE 등)을 적용하지 않고, Stratify을 엄격하게 유지했습니다.

