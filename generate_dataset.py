import numpy as np
import pandas as pd


def generate(n=3000, seed=42, output="dataset.csv"):
    rng = np.random.default_rng(seed)

    age = rng.normal(34, 8, n).clip(20, 65).astype(int)
    gender = rng.choice(["М", "Ж", "Другое"], n, p=[0.58, 0.38, 0.04])
    education = rng.choice(["Среднее", "Бакалавр", "Магистр", "PhD"], n, p=[0.08, 0.42, 0.38, 0.12])
    city_size = rng.choice(["<100K", "100K-500K", "500K-1M", ">1M"], n, p=[0.10, 0.25, 0.30, 0.35])
    marital_status = rng.choice(["Не в браке", "В браке", "Разведён(а)", "Партнёрство"], n, p=[0.35, 0.40, 0.10, 0.15])
    children_count = rng.poisson(0.8, n).clip(0, 5)

    experience_years = (age - 20 - rng.exponential(2, n)).clip(0, 40).round(1)
    role = rng.choice(
        ["Junior Dev", "Middle Dev", "Senior Dev", "Lead", "Manager",
         "QA", "DevOps", "Data Scientist", "Designer", "Product Manager"],
        n, p=[0.12, 0.20, 0.18, 0.08, 0.07, 0.10, 0.08, 0.07, 0.05, 0.05])
    department = rng.choice(
        ["Backend", "Frontend", "Mobile", "Data", "Infrastructure", "Product", "QA"],
        n, p=[0.22, 0.18, 0.12, 0.15, 0.13, 0.10, 0.10])
    company_tenure = np.minimum(experience_years, rng.exponential(3, n)).clip(0).round(1)
    work_format = rng.choice(["Офис", "Удалёнка", "Гибрид"], n, p=[0.25, 0.40, 0.35])
    team_size = rng.poisson(7, n).clip(2, 25)
    projects_count = (experience_years * 0.3 + rng.poisson(2, n)).clip(0).astype(int)

    edu_map = {"Среднее": 0, "Бакалавр": 5, "Магистр": 15, "PhD": 25}
    edu_bonus = np.array([edu_map[e] for e in education], dtype=float)
    role_bonus = np.array([
        40 if r in {"Lead", "Manager"} else 25 if r in {"Senior Dev", "Data Scientist"}
        else 10 if r in {"Middle Dev", "DevOps", "Product Manager"} else 0
        for r in role], dtype=float)
    salary_k = (60 + experience_years * 5.5 + edu_bonus + role_bonus + rng.normal(0, 15, n)).clip(30, 350).round(0)
    overtime_hours_week = rng.exponential(4, n).clip(0, 30).round(1)

    sleep_hours = rng.normal(6.8, 1.2, n).clip(3, 10).round(1)
    sport_hours_week = rng.exponential(3, n).clip(0, 20).round(1)
    bmi = rng.normal(25, 4, n).clip(16, 45).round(1)
    smoking = rng.choice([0, 1], n, p=[0.78, 0.22])
    alcohol_drinks_week = rng.exponential(3, n).clip(0, 30).round(0).astype(int)
    screen_time_hours = rng.normal(9, 2.5, n).clip(2, 18).round(1)
    steps_per_day = (sport_hours_week * 800 + rng.normal(5000, 2000, n)).clip(1000, 25000).round(0).astype(int)
    meditation_min_week = rng.exponential(15, n).clip(0, 180).round(0).astype(int)
    coffee_cups_day = rng.poisson(2.5, n).clip(0, 10)
    sick_days_year = rng.poisson(5, n).clip(0, 30)
    chronic_conditions = rng.choice([0, 1, 2, 3], n, p=[0.55, 0.28, 0.12, 0.05])
    commute_minutes = np.where(work_format == "Удалёнка", 0, rng.exponential(30, n).clip(5, 120)).round(0).astype(int)

    stress_level = (5 + overtime_hours_week * 0.15 - sleep_hours * 0.3 + rng.normal(0, 1.5, n)).clip(1, 10).round(1)
    life_satisfaction = (7 - stress_level * 0.3 + salary_k * 0.005 + sleep_hours * 0.2 + rng.normal(0, 1.2, n)).clip(1, 10).round(1)

    coffee_effect = -0.15 * (coffee_cups_day - 3) ** 2 + 0.3
    sleep_overtime_mod = np.where(overtime_hours_week > 10, sleep_hours * 0.15, 0)
    productivity = (4 + sleep_hours * 0.35 + sport_hours_week * 0.08 - stress_level * 0.2
                    + coffee_effect + sleep_overtime_mod + rng.normal(0, 1.0, n)).clip(1, 10).round(1)

    burnout_score = (2 + stress_level * 0.4 + overtime_hours_week * 0.1 - sleep_hours * 0.15
                     - sport_hours_week * 0.05 + rng.normal(0, 1.2, n)).clip(1, 10).round(1)
    format_bonus = np.where(work_format == "Удалёнка", 0.8, np.where(work_format == "Гибрид", 0.4, 0))
    job_satisfaction = (5 + format_bonus - stress_level * 0.2 + salary_k * 0.005 + rng.normal(0, 1.3, n)).clip(1, 10).round(1)
    motivation = (5 + job_satisfaction * 0.2 - burnout_score * 0.15 + rng.normal(0, 1.5, n)).clip(1, 10).round(1)
    creativity_score = (5 + sleep_hours * 0.15 + meditation_min_week * 0.005 - stress_level * 0.1 + rng.normal(0, 1.5, n)).clip(1, 10).round(1)
    emotional_intelligence = rng.normal(6, 1.8, n).clip(1, 10).round(1)
    extraversion = rng.normal(5.5, 2, n).clip(1, 10).round(1)
    openness = rng.normal(6, 1.8, n).clip(1, 10).round(1)
    conscientiousness = rng.normal(6.5, 1.6, n).clip(1, 10).round(1)
    neuroticism = rng.normal(4.5, 2, n).clip(1, 10).round(1)
    agreeableness = rng.normal(6, 1.7, n).clip(1, 10).round(1)
    self_efficacy = (5 + experience_years * 0.05 + rng.normal(0, 1.5, n)).clip(1, 10).round(1)

    meetings_per_week = rng.poisson(6, n).clip(0, 25)
    focus_hours_day = (8 - meetings_per_week * 0.15 + rng.normal(0, 1, n)).clip(1, 8).round(1)
    tasks_completed_month = (productivity * 3 + rng.poisson(5, n)).clip(0).astype(int)
    code_reviews_month = rng.poisson(8, n).clip(0, 40)
    bugs_reported_month = rng.poisson(3, n).clip(0, 20)
    learning_hours_week = rng.exponential(2.5, n).clip(0, 15).round(1)

    social_events_month = rng.poisson(2, n).clip(0, 12)
    mentoring_hours_month = rng.exponential(2, n).clip(0, 20).round(1)
    vacation_days_used = rng.poisson(12, n).clip(0, 30)
    salary_satisfaction = (job_satisfaction * 0.4 + salary_k * 0.01 - 2 + rng.normal(0, 1.2, n)).clip(1, 10).round(1)
    turnover_intention = (burnout_score * 0.3 - job_satisfaction * 0.25 + 3 + rng.normal(0, 1.2, n)).clip(1, 10).round(1)

    left_company = (turnover_intention > 7).astype(int)
    flip_idx = rng.choice(n, size=int(n * 0.05), replace=False)
    left_company[flip_idx] = 1 - left_company[flip_idx]

    df = pd.DataFrame({
        "age": age, "gender": gender, "education": education, "city_size": city_size,
        "marital_status": marital_status, "children_count": children_count,
        "experience_years": experience_years, "role": role, "department": department,
        "company_tenure": company_tenure, "work_format": work_format, "team_size": team_size,
        "projects_count": projects_count, "salary_k": salary_k,
        "overtime_hours_week": overtime_hours_week, "left_company": left_company,
        "sleep_hours": sleep_hours, "sport_hours_week": sport_hours_week, "bmi": bmi,
        "smoking": smoking, "alcohol_drinks_week": alcohol_drinks_week,
        "screen_time_hours": screen_time_hours, "steps_per_day": steps_per_day,
        "meditation_min_week": meditation_min_week, "coffee_cups_day": coffee_cups_day,
        "sick_days_year": sick_days_year, "chronic_conditions": chronic_conditions,
        "commute_minutes": commute_minutes,
        "stress_level": stress_level, "life_satisfaction": life_satisfaction,
        "productivity": productivity, "burnout_score": burnout_score,
        "job_satisfaction": job_satisfaction, "motivation": motivation,
        "creativity_score": creativity_score, "emotional_intelligence": emotional_intelligence,
        "extraversion": extraversion, "openness": openness,
        "conscientiousness": conscientiousness, "neuroticism": neuroticism,
        "agreeableness": agreeableness, "self_efficacy": self_efficacy,
        "meetings_per_week": meetings_per_week, "focus_hours_day": focus_hours_day,
        "tasks_completed_month": tasks_completed_month, "code_reviews_month": code_reviews_month,
        "bugs_reported_month": bugs_reported_month, "learning_hours_week": learning_hours_week,
        "social_events_month": social_events_month, "mentoring_hours_month": mentoring_hours_month,
        "vacation_days_used": vacation_days_used, "salary_satisfaction": salary_satisfaction,
        "turnover_intention": turnover_intention,
    })

    df.to_csv(output, index=False)
    print(f"Сохранено: {output} -- {df.shape[0]} строк, {df.shape[1]} колонок")
    return df


if __name__ == "__main__":
    generate()
