import numpy as np


def add_hr_features(df):
    """FE 아이디어 문서의 공식에 맞춰 HR 파생변수를 추가합니다."""
    data = df.copy()

    overtime = data["OverTime"].map({"Yes": 1, "No": 0}).fillna(data["OverTime"]).astype(int)

    data["Income_Per_WorkingYear"] = data["MonthlyIncome"] / (data["TotalWorkingYears"] + 1)
    data["Income_Per_YearAtCompany"] = data["MonthlyIncome"] / (data["YearsAtCompany"] + 1)
    data["Income_Per_Level"] = data["MonthlyIncome"] / data["JobLevel"]
    data["Cost_Effectiveness"] = data["PercentSalaryHike"] / data["PerformanceRating"]

    data["Overwork_Fatigue_Index"] = overtime * data["DistanceFromHome"]
    data["Burnout_Risk"] = overtime + (5 - data["WorkLifeBalance"])
    data["JobSat_WLB_Interaction"] = data["JobSatisfaction"] * data["WorkLifeBalance"]

    data["Career_Velocity_Index"] = data["JobLevel"] / (data["TotalWorkingYears"] + 1)
    data["Stagnation_Index"] = data["YearsSinceLastPromotion"] / (data["YearsAtCompany"] + 1)

    data["Total_Satisfaction_Score"] = (
        data["EnvironmentSatisfaction"]
        + data["JobSatisfaction"]
        + data["RelationshipSatisfaction"]
        + data["WorkLifeBalance"]
    )

    data["Job_Hopping_Index"] = data["NumCompaniesWorked"] / (data["TotalWorkingYears"] + 1)
    data["Organizational_Loyalty_Ratio"] = np.where(
        data["TotalWorkingYears"] > 0,
        data["YearsAtCompany"] / data["TotalWorkingYears"],
        0,
    )

    return data


def apply_advanced_feature_engineering(df):
    """기존 팀 코드와 호환되는 이전 파생변수 생성 함수입니다."""
    data = df.copy()

    data["Total_Satisfaction_Score"] = (
        data["EnvironmentSatisfaction"]
        + data["JobSatisfaction"]
        + data["RelationshipSatisfaction"]
        + data["WorkLifeBalance"]
    )

    data["Income_Per_WorkingYear"] = data["MonthlyIncome"] / (data["TotalWorkingYears"] + 1)
    data["Income_Per_YearAtCompany"] = data["MonthlyIncome"] / (data["YearsAtCompany"] + 1)
    data["Income_Per_Level"] = data["MonthlyIncome"] / data["JobLevel"]
    data["Cost_Effectiveness"] = data["PercentSalaryHike"] / data["PerformanceRating"]

    overtime = data["OverTime"].apply(lambda x: 1 if x == "Yes" or x == 1 else 0)
    data["Burnout_Risk"] = overtime + (5 - data["WorkLifeBalance"])
    data["Sat_WLB_Interaction"] = data["Total_Satisfaction_Score"] * data["WorkLifeBalance"]

    data["Promotion_Speed_Index"] = data["JobLevel"] / (data["TotalWorkingYears"] + 1)
    data["Stagnation_Index"] = data["YearsSinceLastPromotion"] / (data["YearsAtCompany"] + 1)
    data["Job_Hopping_Index"] = data["NumCompaniesWorked"] / (data["TotalWorkingYears"] + 1)
    data["Loyalty_Ratio"] = np.where(
        data["TotalWorkingYears"] > 0,
        data["YearsAtCompany"] / data["TotalWorkingYears"],
        0,
    )

    return data
