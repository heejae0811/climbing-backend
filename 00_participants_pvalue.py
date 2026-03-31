import pandas as pd
import numpy as np
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu

# ==============================
# 1. 파일 경로
# ==============================
file_path = "participants_summary.xlsx"

# ==============================
# 2. 시트 불러오기
# ==============================
total_df = pd.read_excel(file_path, sheet_name="전체")
male_df = pd.read_excel(file_path, sheet_name="남자")
female_df = pd.read_excel(file_path, sheet_name="여자")

# 컬럼명 공백 제거
total_df.columns = total_df.columns.str.strip()
male_df.columns = male_df.columns.str.strip()
female_df.columns = female_df.columns.str.strip()

# ==============================
# 3. 변수명 매핑
#    왼쪽: 실제 엑셀 컬럼명
#    오른쪽: 논문 표에 넣을 이름
# ==============================
variable_map = {
    "age": "Age (years)",
    "height": "Height (cm)",
    "weight": "Weight (kg)",
    "bmi": "BMI (kg/m²)",
    "experience": "Climbing experience (years)",
    "climbing_duration_per_week": "Climbing duration per week (days)"
}

# 숫자형 변환
for col in variable_map.keys():
    total_df[col] = pd.to_numeric(total_df[col], errors="coerce")
    male_df[col] = pd.to_numeric(male_df[col], errors="coerce")
    female_df[col] = pd.to_numeric(female_df[col], errors="coerce")

# ==============================
# 4. 평균 ± 표준편차 포맷 함수
# ==============================
def format_mean_sd(series):
    series = series.dropna()
    if len(series) == 0:
        return ""
    return f"{series.mean():.2f} ± {series.std(ddof=1):.2f}"

# ==============================
# 5. p-value 포맷 함수
# ==============================
def format_p_value(p):
    if pd.isna(p):
        return ""
    elif p < 0.001:
        return "<0.001"
    else:
        return f"{p:.3f}"

# ==============================
# 6. 두 그룹 비교 함수
#    정규성 + 등분산 확인 후 자동 검정
# ==============================
def compare_groups(male_series, female_series):
    x = male_series.dropna()
    y = female_series.dropna()

    if len(x) < 3 or len(y) < 3:
        return np.nan, "Not enough data"

    # Shapiro-Wilk 정규성 검사
    p_shapiro_x = shapiro(x)[1]
    p_shapiro_y = shapiro(y)[1]

    # Levene 등분산 검사
    p_levene = levene(x, y)[1]

    # 검정 선택
    if p_shapiro_x > 0.05 and p_shapiro_y > 0.05:
        if p_levene > 0.05:
            stat, p = ttest_ind(x, y, equal_var=True)
            test_name = "Independent t-test"
        else:
            stat, p = ttest_ind(x, y, equal_var=False)
            test_name = "Welch t-test"
    else:
        stat, p = mannwhitneyu(x, y, alternative="two-sided")
        test_name = "Mann-Whitney U"

    return p, test_name

# ==============================
# 7. 결과 테이블 생성
# ==============================
results = []

# 참가자 수 행 먼저 추가
results.append({
    "Characteristic": "Number of participants (n)",
    "Total": len(total_df),
    "Men": len(male_df),
    "Women": len(female_df),
    "p-value": "",
    "Test": ""
})

for col, label in variable_map.items():
    total_series = total_df[col]
    male_series = male_df[col]
    female_series = female_df[col]

    p, test_name = compare_groups(male_series, female_series)

    results.append({
        "Characteristic": label,
        "Total": format_mean_sd(total_series),
        "Men": format_mean_sd(male_series),
        "Women": format_mean_sd(female_series),
        "p-value": format_p_value(p),
        "Test": test_name
    })

result_df = pd.DataFrame(results)

# ==============================
# 8. 출력
# ==============================
print(result_df)

# ==============================
# 9. 엑셀 저장
# ==============================
output_file = "participants_pvalues.xlsx"
result_df.to_excel(output_file, index=False)

print(f"\n저장 완료: {output_file}")