import pandas as pd
import numpy as np

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