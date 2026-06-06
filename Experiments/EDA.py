import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# 0. 환경 세팅 (한글 깨짐 방지 및 스타일 설정)
# ==============================================================================
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

color_dict = {'No': '#4C72B0', 'Yes': '#C44E52'}
color_list = ['#4C72B0', '#C44E52']

sns.set_theme(style="whitegrid", font='Malgun Gothic', palette="pastel")

# 데이터 로드
df_eda = pd.read_csv("data/data_team7.csv")

# ==============================================================================
# 1. 타겟 변수 분포 및 불균형 확인 
# ==============================================================================
print("\n" + "="*20 + " 1. 타겟 변수 분포 및 불균형 확인 " + "="*20)

plt.figure(figsize=(6, 5))
ax = sns.countplot(x='Attrition', data=df_eda, hue='Attrition', palette=color_dict, order=['No', 'Yes'], legend=False)

# 각 바 위에 빈도수와 백분율 달아주기
total = len(df_eda)
for p in ax.patches:
    percentage = f'{100 * p.get_height() / total:.1f}%'
    x_pos = p.get_x() + p.get_width() / 2
    y_pos = p.get_height() + 15
    ax.text(x_pos, y_pos, f'{int(p.get_height())}명\n({percentage})', ha='center', va='baseline', fontsize=11, fontweight='bold')

plt.title('퇴사 여부(Attrition) 타겟 변수 분포 및 불균형 확인', fontsize=13, fontweight='bold', pad=15)
plt.xlabel('퇴사 여부 (Attrition)', fontsize=11)
plt.ylabel('임직원 수 (명)', fontsize=11)
plt.ylim(0, total * 0.95)
plt.tight_layout()
plt.show()

# ==============================================================================
# 2. 주요 퇴사 요인 분석
# ==============================================================================
print("\n" + "="*20 + " 2. 주요 퇴사 요인 분석 " + "="*20)
# ------------------------------------------------------------------------------
# [그래프 1] 야근 여부(OverTime)별 퇴사 분포
# ------------------------------------------------------------------------------
plt.figure(figsize=(7, 5))
sns.countplot(x='OverTime', hue='Attrition', data=df_eda, palette=color_dict, hue_order=['No', 'Yes'])

plt.title('야근 여부(OverTime)에 따른 퇴사 분포', fontsize=13, fontweight='bold', pad=15)
plt.xlabel('야근 여부', fontsize=11)
plt.ylabel('인원 수 (명)', fontsize=11)
plt.legend(title='Attrition', loc='upper right')
plt.subplots_adjust(top=0.85, bottom=0.15)  
plt.show()

# ------------------------------------------------------------------------------
# [그래프 2] 직무별 100% 누적 상대 비율 그래프 (가로형 barh)
# ------------------------------------------------------------------------------
plt.figure(figsize=(8.5, 6))

job_attr_ratio = pd.crosstab(df_eda['JobRole'], df_eda['Attrition'], normalize='index') * 100
job_attr_ratio = job_attr_ratio[['No', 'Yes']].sort_values(by='Yes', ascending=True)

job_attr_ratio.plot(kind='barh', stacked=True, color=color_list, width=0.6, ax=plt.gca())

for p in plt.gca().patches:
    width = p.get_width()
    if width > 4:
        x_pos = p.get_x() + width / 2
        y_pos = p.get_y() + p.get_height() / 2
        plt.gca().text(x_pos, y_pos, f'{width:.1f}%', ha='center', va='center', 
                       color='white', fontsize=9, fontweight='bold')

plt.title('직무(JobRole)별 퇴사 상대 비율 (100% Stacked)', fontsize=13, fontweight='bold', pad=15)
plt.xlabel('비율 (%)', fontsize=11)
plt.ylabel('직무 종류', fontsize=11)
plt.legend(title='Attrition', loc='lower right')
plt.subplots_adjust(top=0.88, bottom=0.15, left=0.28) 
plt.show()


# ------------------------------------------------------------------------------
# [그래프 3] 출장 빈도(BusinessTravel)에 따른 퇴사 분포
# ------------------------------------------------------------------------------
plt.figure(figsize=(7, 5))
sns.countplot(x='BusinessTravel', hue='Attrition', data=df_eda, palette=color_dict, hue_order=['No', 'Yes'],
              order=['Non-Travel', 'Travel_Rarely', 'Travel_Frequently'])

plt.title('출장 빈도(BusinessTravel)에 따른 퇴사 분포', fontsize=13, fontweight='bold', pad=15)
plt.xlabel('출장 빈도', fontsize=11)
plt.ylabel('인원 수 (명)', fontsize=11)
plt.legend(title='Attrition', loc='upper right')
plt.subplots_adjust(top=0.85, bottom=0.15)
plt.show()

# ==============================================================================
# 3. 수치형 변수 상관관계 분석 (Correlation Heatmap)
# ==============================================================================
print("\n" + "="*20 + " 3. 수치형 변수 상관관계 분석 " + "="*20)

all_num_cols = df_eda.select_dtypes(include=[np.number]).columns.tolist()

usable_num_cols = [col for col in all_num_cols if df_eda[col].nunique() > 1]

corr_matrix = df_eda[usable_num_cols].corr()

plt.figure(figsize=(14, 11))

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', 
            vmin=-1, vmax=1, center=0, square=True, linewidths=.5, 
            cbar_kws={"shrink": .8, "label": "상관계수 크기 (Color Intensity)"})

plt.title('전체 수치형 변수간 상관관계 피어슨 행렬 (Heatmap)', fontsize=15, fontweight='bold', pad=25)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.show()