# # Агентная система количественного исследования (v2)
# 
# ```
# Фаза 0: DATA QUALITY AUDIT    — пропуски, выбросы, power analysis
# Фаза 1: EDA                   — описательные, корреляции, распределения
# Фаза 2: HYPOTHESES            — генерация гипотез из находок
# Фаза 3: TESTING               — стат. тесты с автовыбором
# Фаза 4: REFLECTION            — углубление, новые гипотезы
# Фаза 5: CAUSAL REASONING      — partial correlations, медиация, DAG
# Фаза 6: ROBUSTNESS CHECKS     — bootstrap, outlier sensitivity, split-half
# Фаза 7: RECOMMENDATIONS       — сегменты, ROI, risk report
# Фаза 8: SYNTHESIS             — финальный отчёт
# ```

# ## 0. Установка

# pip install openai scipy pingouin scikit-learn matplotlib seaborn

import json, os, re, textwrap, traceback
from dataclasses import dataclass, field
from typing import Any
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import (
    pearsonr, spearmanr, ttest_ind, mannwhitneyu,
    f_oneway, kruskal, chi2_contingency, levene, normaltest,
)
import pingouin as pg
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

import warnings; warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.figsize": (10, 5), "font.size": 11})
print("Готово")

# ## 1. Конфигурация
# 
# **OPENROUTER_API_KEY** — ключ для доступа к LLM через OpenRouter. Получить: https://openrouter.ai/keys
# Бесплатного лимита обычно хватает на 2-3 полных прогона.
# 
# **MODEL** — любая модель с поддержкой function calling. Рекомендуемые:
# - `anthropic/claude-sonnet-4` — лучшее качество рассуждений
# - `openai/gpt-4o` — быстрее, дешевле
# - `google/gemini-2.5-flash` — самый дешёвый вариант
# 
# **MAX_ITERATIONS** — лимит вызовов инструментов за одну фазу. 40 хватает для большинства случаев. Если агент упирается в лимит — увеличить до 60.
# 
# **MAX_DEEPENING** — сколько раундов углубления разрешено в фазе рефлексии. 2 — баланс между глубиной и стоимостью.
# 
# **ALPHA** — порог статистической значимости. Стандарт 0.05, для строгих исследований 0.01.

OPENROUTER_API_KEY = ""
MODEL = "anthropic/claude-sonnet-4"
MAX_ITERATIONS = 40
MAX_DEEPENING = 2
ALPHA = 0.05
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

from openai import OpenAI

assert OPENROUTER_API_KEY, "Вставьте OPENROUTER_API_KEY"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

print(f"Подключено: {MODEL}")

# ## 2. Данные

DATASET_PATH = "dataset.csv"

if not os.path.exists(DATASET_PATH):
    print("Файл не найден, генерируем синтетический датасет...")
    rng = np.random.default_rng(42); N = 3000
    age = rng.normal(34,8,N).clip(20,65).astype(int)
    exp = (age-20-rng.exponential(2,N)).clip(0,40).round(1)
    education = rng.choice(["Среднее","Бакалавр","Магистр","PhD"],N,p=[.08,.42,.38,.12])
    role = rng.choice(["Junior","Middle","Senior","Lead","Manager","QA","DevOps","DS","Designer","PM"],N,p=[.12,.20,.18,.08,.07,.10,.08,.07,.05,.05])
    department = rng.choice(["Backend","Frontend","Mobile","Data","Infra","Product","QA"],N,p=[.22,.18,.12,.15,.13,.10,.10])
    work_format = rng.choice(["Офис","Удалёнка","Гибрид"],N,p=[.25,.4,.35])
    gender = rng.choice(["М","Ж","Другое"],N,p=[.58,.38,.04])
    edu_b = np.array([{"Среднее":0,"Бакалавр":5,"Магистр":15,"PhD":25}[e] for e in education],dtype=float)
    role_b = np.array([40 if r in {"Lead","Manager"} else 25 if r in {"Senior","DS"} else 10 if r in {"Middle","DevOps","PM"} else 0 for r in role],dtype=float)
    salary_k = (60+exp*5.5+edu_b+role_b+rng.normal(0,15,N)).clip(30,350).round(0)
    overtime = rng.exponential(4,N).clip(0,30).round(1)
    sleep = rng.normal(6.8,1.2,N).clip(3,10).round(1)
    sport = rng.exponential(3,N).clip(0,20).round(1)
    coffee = rng.poisson(2.5,N).clip(0,10)
    stress = (5+overtime*0.15-sleep*0.3+rng.normal(0,1.5,N)).clip(1,10).round(1)
    coffee_fx = -0.15*(coffee-3)**2+0.3
    sleep_mod = np.where(overtime>10,sleep*0.15,0)
    productivity = (4+sleep*0.35+sport*0.08-stress*0.2+coffee_fx+sleep_mod+rng.normal(0,1,N)).clip(1,10).round(1)
    burnout = (2+stress*0.4+overtime*0.1-sleep*0.15-sport*0.05+rng.normal(0,1.2,N)).clip(1,10).round(1)
    fmt_b = np.where(work_format=="Удалёнка",.8,np.where(work_format=="Гибрид",.4,0))
    job_sat = (5+fmt_b-stress*0.2+salary_k*0.005+rng.normal(0,1.3,N)).clip(1,10).round(1)
    motivation = (5+job_sat*0.2-burnout*0.15+rng.normal(0,1.5,N)).clip(1,10).round(1)
    life_sat = (7-stress*0.3+salary_k*0.005+sleep*0.2+rng.normal(0,1.2,N)).clip(1,10).round(1)
    creativity = (5+sleep*0.15-stress*0.1+rng.normal(0,1.5,N)).clip(1,10).round(1)
    turnover = (burnout*0.3-job_sat*0.25+3+rng.normal(0,1.2,N)).clip(1,10).round(1)
    left = (turnover>7).astype(int); flip=rng.choice(N,int(N*.05),replace=False); left[flip]=1-left[flip]
    tenure = np.minimum(exp,rng.exponential(3,N)).clip(0).round(1)
    team_size = rng.poisson(7,N).clip(2,25)
    meetings = rng.poisson(6,N).clip(0,25)
    focus = (8-meetings*0.15+rng.normal(0,1,N)).clip(1,8).round(1)
    tasks = (productivity*3+rng.poisson(5,N)).clip(0).astype(int)
    bmi = rng.normal(25,4,N).clip(16,45).round(1)
    screen = rng.normal(9,2.5,N).clip(2,18).round(1)
    steps = (sport*800+rng.normal(5000,2000,N)).clip(1000,25000).round(0).astype(int)
    meditation = rng.exponential(15,N).clip(0,180).round(0).astype(int)
    sick = rng.poisson(5,N).clip(0,30)
    commute = np.where(work_format=="Удалёнка",0,rng.exponential(30,N).clip(5,120)).round(0).astype(int)
    learning = rng.exponential(2.5,N).clip(0,15).round(1)
    extraversion = rng.normal(5.5,2,N).clip(1,10).round(1)
    neuroticism = rng.normal(4.5,2,N).clip(1,10).round(1)
    conscientiousness = rng.normal(6.5,1.6,N).clip(1,10).round(1)
    self_efficacy = (5+exp*0.05+rng.normal(0,1.5,N)).clip(1,10).round(1)
    sal_sat = (job_sat*0.4+salary_k*0.01-2+rng.normal(0,1.2,N)).clip(1,10).round(1)

    df = pd.DataFrame({
        "age":age,"gender":gender,"education":education,"role":role,"department":department,
        "work_format":work_format,"experience_years":exp,"company_tenure":tenure,
        "team_size":team_size,"salary_k":salary_k,"overtime_hours_week":overtime,
        "sleep_hours":sleep,"sport_hours_week":sport,"coffee_cups_day":coffee,"bmi":bmi,
        "screen_time_hours":screen,"steps_per_day":steps,"meditation_min_week":meditation,
        "sick_days_year":sick,"commute_minutes":commute,
        "stress_level":stress,"productivity":productivity,"burnout_score":burnout,
        "job_satisfaction":job_sat,"motivation":motivation,"life_satisfaction":life_sat,
        "creativity_score":creativity,"extraversion":extraversion,"neuroticism":neuroticism,
        "conscientiousness":conscientiousness,"self_efficacy":self_efficacy,
        "meetings_per_week":meetings,"focus_hours_day":focus,
        "tasks_completed_month":tasks,"learning_hours_week":learning,
        "salary_satisfaction":sal_sat,"turnover_intention":turnover,"left_company":left,
    })
    df.to_csv(DATASET_PATH, index=False)

df = pd.read_csv(DATASET_PATH)
print(f"Загружено: {df.shape[0]} строк, {df.shape[1]} колонок")
df.head(3)

# ## 3. Состояние исследования

@dataclass
class Hypothesis:
    id: str
    text: str
    variables: list
    status: str = "pending"
    parent_id: str = None
    test_results: list = field(default_factory=list)
    generation: int = 0

@dataclass
class ResearchState:
    dataset_profile: dict = field(default_factory=dict)
    quality_report: dict = field(default_factory=dict)
    eda_findings: list = field(default_factory=list)
    hypotheses: list = field(default_factory=list)
    test_log: list = field(default_factory=list)
    tested_pairs: set = field(default_factory=set)
    robustness_log: list = field(default_factory=list)
    causal_findings: list = field(default_factory=list)
    recommendations: list = field(default_factory=list)
    plots: list = field(default_factory=list)
    agent_reasoning: list = field(default_factory=list)
    deepening_round: int = 0
    _hyp_counter: int = 0

    def add_hypothesis(self, text, variables, parent_id=None, generation=0):
        self._hyp_counter += 1
        h = Hypothesis(id=f"H{self._hyp_counter}", text=text, variables=variables,
                       parent_id=parent_id, generation=generation)
        self.hypotheses.append(h)
        return h.id

    def is_tested(self, v1, v2, ttype):
        return tuple(sorted([v1, v2])) + (ttype,) in self.tested_pairs

    def mark_tested(self, v1, v2, ttype):
        self.tested_pairs.add(tuple(sorted([v1, v2])) + (ttype,))

    def get_summary(self):
        return {
            "total_hypotheses": len(self.hypotheses),
            "confirmed": sum(1 for h in self.hypotheses if h.status == "confirmed"),
            "rejected": sum(1 for h in self.hypotheses if h.status == "rejected"),
            "pending": sum(1 for h in self.hypotheses if h.status == "pending"),
            "tests_run": len(self.test_log),
            "robustness_checks": len(self.robustness_log),
            "causal_findings": len(self.causal_findings),
            "plots": len(self.plots),
            "quality_issues": self.quality_report.get("issues", []),
            "hypotheses": [{"id":h.id,"text":h.text,"status":h.status,"parent":h.parent_id,"gen":h.generation} for h in self.hypotheses],
            "tested_pairs": [list(p) for p in self.tested_pairs],
        }

state = ResearchState()
print("Состояние инициализировано")

# ## 4. Инструменты

def _save_plot(fig, label):
    fname = f"{PLOT_DIR}/{len(state.plots)+1:02d}_{label}.png"
    fig.savefig(fname, dpi=120, bbox_inches="tight"); plt.close(fig)
    state.plots.append(fname)
    return fname

def tool_describe_dataset(**kw):
    num = df.select_dtypes(include="number").columns.tolist()
    cat = df.select_dtypes(include="object").columns.tolist()
    profile = {
        "shape": list(df.shape), "numeric_columns": num, "categorical_columns": cat,
        "numeric_stats": df[num].describe().round(2).to_dict(),
        "categorical_distributions": {c: df[c].value_counts().head(8).to_dict() for c in cat},
        "missing": df.isnull().sum()[df.isnull().sum()>0].to_dict() or "none",
    }
    state.dataset_profile = profile
    return profile

def tool_audit_data_quality(**kw):
    issues = []
    num = df.select_dtypes(include="number").columns

    miss = df.isnull().mean()
    bad_cols = miss[miss > 0.05].to_dict()
    if bad_cols:
        issues.append({"type": "MISSING_DATA", "severity": "high" if max(bad_cols.values()) > 0.2 else "medium", "details": {k: f"{v*100:.1f}%" for k,v in bad_cols.items()}})

    dups = df.duplicated().sum()
    if dups > 0:
        issues.append({"type": "DUPLICATES", "severity": "medium", "count": int(dups)})

    outlier_report = {}
    for c in num:
        q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        iqr = q3 - q1
        n_out = ((df[c] < q1 - 3*iqr) | (df[c] > q3 + 3*iqr)).sum()
        if n_out > len(df)*0.01:
            outlier_report[c] = int(n_out)
    if outlier_report:
        issues.append({"type": "OUTLIERS", "severity": "medium", "columns": outlier_report})

    constants = [c for c in df.columns if df[c].nunique() <= 1]
    if constants:
        issues.append({"type": "CONSTANT_COLUMNS", "severity": "low", "columns": constants})

    low_var = [c for c in num if df[c].std() < 0.001]
    if low_var:
        issues.append({"type": "NEAR_ZERO_VARIANCE", "severity": "medium", "columns": low_var})

    min_group = 30
    cat = df.select_dtypes(include="object").columns
    small_groups = {}
    for c in cat:
        smalls = {k: int(v) for k,v in df[c].value_counts().items() if v < min_group}
        if smalls: small_groups[c] = smalls
    if small_groups:
        issues.append({"type": "SMALL_SUBGROUPS", "severity": "medium", "details": small_groups})

    from scipy.stats import norm as norm_dist
    def power_r(r, n, alpha=0.05):
        z = np.arctanh(r) * np.sqrt(n - 3)
        return float(norm_dist.cdf(abs(z) - norm_dist.ppf(1 - alpha/2)))
    n = len(df)
    power_table = {f"r={r}": f"{power_r(r,n):.3f}" for r in [0.05, 0.1, 0.2, 0.3]}

    multicollinear = []
    corr = df[num].corr()
    for i, c1 in enumerate(num):
        for j, c2 in enumerate(num):
            if i < j and abs(corr.loc[c1, c2]) > 0.85:
                multicollinear.append({"pair": [c1, c2], "r": round(corr.loc[c1, c2], 3)})
    if multicollinear:
        issues.append({"type": "HIGH_MULTICOLLINEARITY", "severity": "high", "pairs": multicollinear})

    severity_order = {"high": 0, "medium": 1, "low": 2}
    issues.sort(key=lambda x: severity_order.get(x["severity"], 3))

    overall = "STOP" if any(i["severity"]=="high" and i["type"]=="MISSING_DATA" for i in issues) else "CAUTION" if issues else "GO"
    report = {"overall": overall, "issues": issues, "power_analysis": power_table, "n": n}
    state.quality_report = report
    return report

def tool_get_correlations(top_n=20, method="pearson"):
    num = df.select_dtypes(include="number").columns
    corr = df[num].corr(method=method)
    pairs = []
    for i, c1 in enumerate(num):
        for j, c2 in enumerate(num):
            if i < j:
                r = corr.loc[c1, c2]
                pairs.append({"var_1": c1, "var_2": c2, "r": round(r, 4), "abs_r": round(abs(r), 4)})
    pairs.sort(key=lambda x: x["abs_r"], reverse=True)
    result = pairs[:int(top_n)]
    state.eda_findings.append({"type": "correlations", "top": result[:5]})
    return {"method": method, "top_pairs": result}

def tool_get_distribution(variable):
    if variable not in df.columns:
        return {"error": f"Не найдена: {variable}. Доступные: {df.columns.tolist()}"}
    col = df[variable]
    if col.dtype == "object":
        return {"variable": variable, "type": "categorical", "value_counts": col.value_counts().to_dict(), "n_unique": col.nunique()}
    result = {"variable": variable, "type": "numeric", "n": int(col.count()),
              "mean": round(col.mean(),3), "std": round(col.std(),3), "median": round(col.median(),3),
              "min": round(col.min(),3), "max": round(col.max(),3),
              "skew": round(col.skew(),3), "kurtosis": round(col.kurtosis(),3)}
    if len(col.dropna()) > 20:
        _, p = normaltest(col.dropna().sample(min(5000, len(col)), random_state=42))
        result["normality_p"] = round(p, 4)
        result["is_normal"] = p > 0.05
    return result

def tool_compare_groups(dv, groupby):
    if dv not in df.columns or groupby not in df.columns:
        return {"error": "Колонка не найдена"}
    groups = {str(k): v[dv].dropna() for k, v in df.groupby(groupby)}
    n_groups = len(groups)
    gstats = {k: {"n": len(v), "mean": round(v.mean(),3), "std": round(v.std(),3)} for k,v in groups.items()}
    if n_groups < 2: return {"error": "Нужно 2+ группы"}
    vals = list(groups.values())
    _, p_lev = levene(*vals)
    result = {"dv": dv, "groupby": groupby, "n_groups": n_groups, "group_stats": gstats, "levene_p": round(p_lev,4)}

    if n_groups == 2:
        keys = list(groups.keys()); a, b = groups[keys[0]], groups[keys[1]]
        t, pt = ttest_ind(a, b, equal_var=(p_lev>0.05))
        d = (a.mean()-b.mean()) / np.sqrt((a.std()**2+b.std()**2)/2) if (a.std()+b.std())>0 else 0
        u, pu = mannwhitneyu(a, b, alternative="two-sided")
        result.update({"test": "Welch t" if p_lev<0.05 else "Student t", "t": round(t,4), "p": round(pt,6), "cohen_d": round(d,4), "mann_whitney_p": round(pu,6), "significant": pt < ALPHA})
    else:
        F, pf = f_oneway(*vals); H, pkw = kruskal(*vals)
        gm = df[dv].mean()
        eta = sum(len(g)*(g.mean()-gm)**2 for g in vals) / max(sum((df[dv]-gm)**2), 1e-10)
        result.update({"test": "ANOVA", "F": round(F,4), "p": round(pf,6), "eta_squared": round(eta,4), "kruskal_p": round(pkw,6), "significant": pf < ALPHA})
        if pf < ALPHA:
            try:
                ph = pg.pairwise_tukey(data=df, dv=dv, between=groupby)
                result["posthoc"] = ph[["A","B","diff","p-tukey","hedges"]].round(4).to_dict("records")
            except: pass
    state.mark_tested(dv, groupby, "comparison"); state.test_log.append(result)
    return result

def tool_test_correlation(var1, var2):
    if var1 not in df.columns or var2 not in df.columns:
        return {"error": "Колонка не найдена"}
    c = df[[var1,var2]].dropna()
    rp, pp = pearsonr(c[var1], c[var2]); rs, ps = spearmanr(c[var1], c[var2])
    result = {"var1": var1, "var2": var2, "n": len(c),
              "pearson_r": round(rp,4), "pearson_p": round(pp,6),
              "spearman_rho": round(rs,4), "spearman_p": round(ps,6),
              "r_squared": round(rp**2,4), "significant": pp < ALPHA,
              "effect": "strong" if abs(rp)>0.5 else "moderate" if abs(rp)>0.3 else "weak" if abs(rp)>0.1 else "negligible"}
    state.mark_tested(var1, var2, "correlation"); state.test_log.append(result)
    return result

def tool_test_association(var1, var2):
    if var1 not in df.columns or var2 not in df.columns:
        return {"error": "Колонка не найдена"}
    ct = pd.crosstab(df[var1], df[var2])
    chi2, p, dof, exp = chi2_contingency(ct)
    v = np.sqrt(chi2/(ct.sum().sum()*(min(ct.shape)-1))) if min(ct.shape)>1 else 0
    result = {"var1": var1, "var2": var2, "chi2": round(chi2,4), "p": round(p,6), "dof": int(dof), "cramers_v": round(v,4), "significant": p < ALPHA}
    state.mark_tested(var1, var2, "association"); state.test_log.append(result)
    return result

def tool_run_regression(dv, predictors):
    missing = [p for p in predictors if p not in df.columns]
    if missing: return {"error": f"Не найдены: {missing}"}
    if dv not in df.columns: return {"error": f"Не найдена: {dv}"}
    data = df[predictors+[dv]].dropna(); X = data[predictors].values; y = data[dv].values
    Xs = StandardScaler().fit_transform(X)
    m = LinearRegression().fit(Xs, y)
    r2 = m.score(Xs, y); r2a = 1-(1-r2)*(len(y)-1)/(len(y)-len(predictors)-1)
    cv = cross_val_score(LinearRegression(), Xs, y, cv=5, scoring="r2")
    vifs = []
    for i in range(len(predictors)):
        others = [j for j in range(len(predictors)) if j != i]
        if others:
            r2_i = LinearRegression().fit(Xs[:, others], Xs[:, i]).score(Xs[:, others], Xs[:, i])
            vifs.append({"predictor": predictors[i], "VIF": round(1/(1-r2_i+1e-10), 2)})
    try:
        lm = pg.linear_regression(data[predictors], data[dv], as_dataframe=True)
        coefs = lm.to_dict("records")
    except:
        coefs = [{"names": p, "coef": round(b,4)} for p,b in zip(["Intercept"]+predictors, [m.intercept_]+list(m.coef_))]
    result = {"dv": dv, "predictors": predictors, "n": len(data), "R2": round(r2,4), "R2_adj": round(r2a,4),
              "cv_R2": round(cv.mean(),4), "cv_std": round(cv.std(),4), "coefficients": coefs, "VIF": vifs}
    for p in predictors: state.mark_tested(p, dv, "regression")
    state.test_log.append(result)
    return result

def tool_create_plot(plot_type, variables, title="", groupby=None):
    fig, ax = plt.subplots(figsize=(10,6))
    try:
        if plot_type == "scatter" and len(variables)>=2:
            sns.regplot(data=df, x=variables[0], y=variables[1], scatter_kws={"alpha":.15,"s":10}, line_kws={"color":"red"}, ci=95, ax=ax)
        elif plot_type == "box" and groupby:
            sns.boxplot(data=df, x=groupby, y=variables[0], palette="Set2", ax=ax)
        elif plot_type == "violin" and groupby:
            sns.violinplot(data=df, x=groupby, y=variables[0], palette="Set2", inner="box", ax=ax)
        elif plot_type == "hist":
            for v in variables: sns.histplot(df[v], kde=True, ax=ax, label=v, alpha=.6)
            ax.legend()
        elif plot_type == "heatmap":
            cols = variables if variables else df.select_dtypes(include="number").columns[:15].tolist()
            sns.heatmap(df[cols].corr(), annot=len(cols)<=12, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
        elif plot_type == "bar" and groupby:
            df.groupby(groupby)[variables[0]].mean().plot(kind="bar", ax=ax, color=sns.color_palette("muted"))
        else:
            plt.close(fig); return {"error": f"Неверные параметры: {plot_type}"}
        ax.set_title(title or f"{plot_type}: {','.join(variables)}", fontweight="bold")
        plt.tight_layout()
        return {"status": "ok", "file": _save_plot(fig, plot_type)}
    except Exception as e:
        plt.close(fig); return {"error": str(e)}

def tool_partial_correlation(var1, var2, covariates):
    if any(v not in df.columns for v in [var1, var2] + covariates):
        return {"error": "Колонка не найдена"}
    data = df[[var1, var2] + covariates].dropna()
    r_full, p_full = pearsonr(data[var1], data[var2])
    try:
        pc = pg.partial_corr(data=data, x=var1, y=var2, covar=covariates, method="pearson")
        r_partial = round(pc["r"].values[0], 4)
        p_partial = round(pc["p-val"].values[0], 6)
    except:
        from numpy.linalg import lstsq
        Xc = data[covariates].values
        res1 = data[var1].values - Xc @ lstsq(Xc, data[var1].values, rcond=None)[0]
        res2 = data[var2].values - Xc @ lstsq(Xc, data[var2].values, rcond=None)[0]
        r_partial, p_partial = pearsonr(res1, res2)
        r_partial, p_partial = round(r_partial, 4), round(p_partial, 6)

    change = round(abs(r_full) - abs(r_partial), 4)
    interpretation = "confound_likely" if change > 0.1 else "partial_confound" if change > 0.05 else "robust"
    result = {"var1": var1, "var2": var2, "covariates": covariates,
              "r_original": round(r_full, 4), "r_partial": r_partial, "p_partial": p_partial,
              "r_change": change, "interpretation": interpretation}
    state.causal_findings.append(result)
    return result

def tool_mediation_analysis(x, mediator, y):
    if any(v not in df.columns for v in [x, mediator, y]):
        return {"error": "Колонка не найдена"}
    data = df[[x, mediator, y]].dropna()
    try:
        med = pg.mediation_analysis(data=data, x=x, m=mediator, y=y, seed=42)
        result = {"x": x, "mediator": mediator, "y": y, "n": len(data), "paths": med.to_dict("records")}
    except:
        a_model = LinearRegression().fit(data[[x]], data[mediator])
        b_model = LinearRegression().fit(data[[x, mediator]], data[y])
        c_model = LinearRegression().fit(data[[x]], data[y])
        a = a_model.coef_[0]; b = b_model.coef_[1]; c = c_model.coef_[0]; c_prime = b_model.coef_[0]
        indirect = a * b
        result = {"x": x, "mediator": mediator, "y": y, "n": len(data),
                  "a_path": round(a,4), "b_path": round(b,4), "c_total": round(c,4),
                  "c_prime_direct": round(c_prime,4), "indirect_ab": round(indirect,4),
                  "mediation_pct": round(abs(indirect/c)*100,1) if abs(c)>0.001 else 0,
                  "type": "full" if abs(c_prime)<0.05 else "partial" if abs(indirect)>0.01 else "none"}
    state.causal_findings.append(result)
    return result

def tool_robustness_check(var1, var2, test_type="correlation"):
    if var1 not in df.columns or var2 not in df.columns:
        return {"error": "Колонка не найдена"}
    results = {"var1": var1, "var2": var2, "checks": {}}

    if test_type == "correlation":
        data = df[[var1, var2]].dropna()
        x, y = data[var1].values, data[var2].values
        r_orig, _ = pearsonr(x, y)

        boot_rs = []
        rng = np.random.default_rng(42)
        for _ in range(1000):
            idx = rng.integers(0, len(x), len(x))
            r_b, _ = pearsonr(x[idx], y[idx])
            boot_rs.append(r_b)
        boot_rs = np.array(boot_rs)
        results["checks"]["bootstrap"] = {"r_mean": round(np.mean(boot_rs),4), "ci_95": [round(np.percentile(boot_rs,2.5),4), round(np.percentile(boot_rs,97.5),4)]}

        q_lo = np.percentile(x, 2); q_hi = np.percentile(x, 98)
        mask = (x >= q_lo) & (x <= q_hi)
        q_lo2 = np.percentile(y, 2); q_hi2 = np.percentile(y, 98)
        mask &= (y >= q_lo2) & (y <= q_hi2)
        r_trim, _ = pearsonr(x[mask], y[mask])
        results["checks"]["outlier_sensitivity"] = {"r_original": round(r_orig,4), "r_trimmed": round(r_trim,4), "delta": round(abs(r_orig-r_trim),4), "fragile": abs(r_orig-r_trim)>0.1}

        mid = len(x)//2; idx = rng.permutation(len(x))
        r1, _ = pearsonr(x[idx[:mid]], y[idx[:mid]]); r2, _ = pearsonr(x[idx[mid:]], y[idx[mid:]])
        results["checks"]["split_half"] = {"r_half1": round(r1,4), "r_half2": round(r2,4), "delta": round(abs(r1-r2),4), "stable": abs(r1-r2)<0.1}

        r_sp, _ = spearmanr(x, y)
        results["checks"]["method_sensitivity"] = {"pearson": round(r_orig,4), "spearman": round(r_sp,4), "delta": round(abs(r_orig-r_sp),4)}

    confidence = "high"
    for ch in results["checks"].values():
        if isinstance(ch, dict) and (ch.get("fragile") or ch.get("delta",0) > 0.15):
            confidence = "low"; break
        if isinstance(ch, dict) and ch.get("delta",0) > 0.08:
            confidence = "medium"

    results["overall_confidence"] = confidence
    state.robustness_log.append(results)
    return results

def tool_segment_analysis(target, risk_vars, threshold_percentile=75):
    if target not in df.columns: return {"error": f"Не найдена: {target}"}
    bad = [v for v in risk_vars if v not in df.columns]
    if bad: return {"error": f"Не найдены: {bad}"}

    segments = []
    for var in risk_vars:
        if df[var].dtype == "object":
            for val in df[var].unique():
                sub = df[df[var]==val]
                if len(sub) >= 30:
                    segments.append({"condition": f"{var}={val}", "n": len(sub),
                                   "target_mean": round(sub[target].mean(),3),
                                   "target_std": round(sub[target].std(),3)})
        else:
            thresh = df[var].quantile(threshold_percentile/100)
            high = df[df[var]>=thresh]; low = df[df[var]<thresh]
            segments.append({"condition": f"{var}>={thresh:.1f} (top {100-threshold_percentile}%)", "n": len(high),
                           "target_mean": round(high[target].mean(),3), "baseline_mean": round(low[target].mean(),3),
                           "diff": round(high[target].mean()-low[target].mean(),3)})

    segments.sort(key=lambda x: abs(x.get("diff", x.get("target_mean",0))), reverse=True)
    overall_mean = round(df[target].mean(), 3)
    risk_groups = [s for s in segments if s.get("diff",0) > 0.5 or s.get("target_mean",0) > overall_mean + df[target].std()*0.5]
    result = {"target": target, "overall_mean": overall_mean, "segments": segments[:15], "high_risk_groups": risk_groups[:5]}
    state.recommendations.append(result)
    return result

def tool_check_subgroups(var1, var2, split_by, test_type="correlation"):
    if any(v not in df.columns for v in [var1, var2, split_by]):
        return {"error": "Колонка не найдена"}
    results = {}
    for g, sub in df.groupby(split_by):
        if len(sub) < 30: continue
        if test_type == "correlation":
            c = sub[[var1,var2]].dropna()
            if len(c)<10: continue
            r, p = pearsonr(c[var1], c[var2])
            results[str(g)] = {"n": len(c), "r": round(r,4), "p": round(p,4)}
        elif test_type == "comparison":
            grps = [v[var1].dropna().values for _,v in sub.groupby(var2) if len(v)>5]
            if len(grps)>=2:
                F, p = f_oneway(*grps)
                results[str(g)] = {"n": len(sub), "F": round(F,4), "p": round(p,4)}
    return {"var1": var1, "var2": var2, "split_by": split_by, "subgroups": results}

def tool_get_research_state(**kw):
    return state.get_summary()

TOOLS_REGISTRY = {
    "describe_dataset": tool_describe_dataset,
    "audit_data_quality": tool_audit_data_quality,
    "get_correlations": tool_get_correlations,
    "get_distribution": tool_get_distribution,
    "compare_groups": tool_compare_groups,
    "test_correlation": tool_test_correlation,
    "test_association": tool_test_association,
    "run_regression": tool_run_regression,
    "create_plot": tool_create_plot,
    "partial_correlation": tool_partial_correlation,
    "mediation_analysis": tool_mediation_analysis,
    "robustness_check": tool_robustness_check,
    "segment_analysis": tool_segment_analysis,
    "check_subgroups": tool_check_subgroups,
    "get_research_state": tool_get_research_state,
}
print(f"{len(TOOLS_REGISTRY)} инструментов зарегистрировано")

# ## 5. Схемы инструментов

TOOL_SCHEMAS = [
  {"type":"function","function":{"name":"describe_dataset","description":"Обзор датасета: shape, типы колонок, описательные статистики, пропуски.","parameters":{"type":"object","properties":{},"required":[]}}},
  {"type":"function","function":{"name":"audit_data_quality","description":"Аудит качества данных: пропуски, выбросы, дубликаты, мультиколлинеарность, power analysis. Возвращает GO/CAUTION/STOP.","parameters":{"type":"object","properties":{},"required":[]}}},
  {"type":"function","function":{"name":"get_correlations","description":"Топ-N корреляций среди числовых переменных.","parameters":{"type":"object","properties":{"top_n":{"type":"integer"},"method":{"type":"string","enum":["pearson","spearman"]}},"required":[]}}},
  {"type":"function","function":{"name":"get_distribution","description":"Статистики распределения и тест нормальности для одной переменной.","parameters":{"type":"object","properties":{"variable":{"type":"string"}},"required":["variable"]}}},
  {"type":"function","function":{"name":"compare_groups","description":"Сравнение групп. Автовыбор: t-test / ANOVA / Mann-Whitney / Kruskal-Wallis. Проверяет допущения, считает размер эффекта, post-hoc.","parameters":{"type":"object","properties":{"dv":{"type":"string","description":"Зависимая переменная (числовая)"},"groupby":{"type":"string","description":"Группирующая переменная"}},"required":["dv","groupby"]}}},
  {"type":"function","function":{"name":"test_correlation","description":"Pearson и Spearman корреляция с интерпретацией размера эффекта.","parameters":{"type":"object","properties":{"var1":{"type":"string"},"var2":{"type":"string"}},"required":["var1","var2"]}}},
  {"type":"function","function":{"name":"test_association","description":"Хи-квадрат и V Крамера для категориальных переменных.","parameters":{"type":"object","properties":{"var1":{"type":"string"},"var2":{"type":"string"}},"required":["var1","var2"]}}},
  {"type":"function","function":{"name":"run_regression","description":"Множественная регрессия: стандартизированные коэффициенты, R-квадрат, кросс-валидация, VIF.","parameters":{"type":"object","properties":{"dv":{"type":"string"},"predictors":{"type":"array","items":{"type":"string"}}},"required":["dv","predictors"]}}},
  {"type":"function","function":{"name":"create_plot","description":"Графики: scatter, box, hist, heatmap, violin, bar.","parameters":{"type":"object","properties":{"plot_type":{"type":"string","enum":["scatter","box","hist","heatmap","violin","bar"]},"variables":{"type":"array","items":{"type":"string"}},"title":{"type":"string"},"groupby":{"type":"string"}},"required":["plot_type","variables"]}}},
  {"type":"function","function":{"name":"partial_correlation","description":"Частная корреляция var1-var2 при контроле за ковариаты. Выявляет конфаунды.","parameters":{"type":"object","properties":{"var1":{"type":"string"},"var2":{"type":"string"},"covariates":{"type":"array","items":{"type":"string"}}},"required":["var1","var2","covariates"]}}},
  {"type":"function","function":{"name":"mediation_analysis","description":"Анализ медиации: идёт ли эффект X на Y напрямую или через медиатор M. Возвращает прямой, косвенный эффект и процент медиации.","parameters":{"type":"object","properties":{"x":{"type":"string"},"mediator":{"type":"string"},"y":{"type":"string"}},"required":["x","mediator","y"]}}},
  {"type":"function","function":{"name":"robustness_check","description":"Проверка устойчивости: bootstrap CI, чувствительность к выбросам, split-half, сравнение методов. Возвращает уровень уверенности.","parameters":{"type":"object","properties":{"var1":{"type":"string"},"var2":{"type":"string"},"test_type":{"type":"string","enum":["correlation"]}},"required":["var1","var2"]}}},
  {"type":"function","function":{"name":"segment_analysis","description":"Сегментация по группам риска. Находит подгруппы с экстремальными значениями целевой переменной.","parameters":{"type":"object","properties":{"target":{"type":"string","description":"Целевая переменная"},"risk_vars":{"type":"array","items":{"type":"string"},"description":"Переменные для сегментации"},"threshold_percentile":{"type":"integer"}},"required":["target","risk_vars"]}}},
  {"type":"function","function":{"name":"check_subgroups","description":"Связь var1-var2 отдельно в подгруппах split_by. Обнаруживает модерацию.","parameters":{"type":"object","properties":{"var1":{"type":"string"},"var2":{"type":"string"},"split_by":{"type":"string"},"test_type":{"type":"string","enum":["correlation","comparison"]}},"required":["var1","var2","split_by"]}}},
  {"type":"function","function":{"name":"get_research_state","description":"Текущее состояние исследования: гипотезы, тесты, что уже проверено.","parameters":{"type":"object","properties":{},"required":[]}}},
]
print(f"{len(TOOL_SCHEMAS)} схем")

# ## 6. Промпты фаз

PHASE_PROMPTS = {

"data_quality": """Фаза: АУДИТ КАЧЕСТВА ДАННЫХ.
1. Вызови describe_dataset
2. Вызови audit_data_quality
3. Оцени результат
Ответь текстом:
=== QUALITY AUDIT COMPLETE ===
Статус: GO/CAUTION/STOP
Перечисли проблемы и рекомендации.""",

"eda": """Фаза: РАЗВЕДОЧНЫЙ АНАЛИЗ.
1. get_correlations
2. get_distribution для 3-5 ключевых переменных
3. Построй 2-3 графика
Ответь:
=== EDA COMPLETE ===
5-7 находок.""",

"hypotheses": """Фаза: ГЕНЕРАЦИЯ ГИПОТЕЗ.
1. Вызови get_research_state чтобы не дублировать
2. Сформулируй 6-8 гипотез на основе EDA
3. Разнообразие тестов
Ответь:
=== HYPOTHESES ===
H1: формулировка | Переменные: список | Тест: тип
H2: ...""",

"testing": """Фаза: ПРОВЕРКА ГИПОТЕЗ.
Для каждой гипотезы:
1. Выбери инструмент
2. Построй визуализацию
Ответь:
=== TESTING COMPLETE ===
Сводка по каждой.""",

"reflection": """Фаза: РЕФЛЕКСИЯ.
Вызови get_research_state. Спроси себя:
- Неожиданные результаты? Конфаунды? Модерация?
- Нужны ли новые гипотезы?
Если да, сформулируй и проверь.
Если нет:
=== REFLECTION COMPLETE ===
Почему углубление не нужно.""",

"causal": """Фаза: КАУЗАЛЬНЫЙ АНАЛИЗ.
Для 3-5 самых важных связей:
1. partial_correlation с контролем за конфаунды
2. mediation_analysis для разложения эффектов
Ответь:
=== CAUSAL COMPLETE ===
Что прямой эффект, что через медиатор, что конфаунд.""",

"robustness": """Фаза: ПРОВЕРКА УСТОЙЧИВОСТИ.
Для 3-5 ключевых результатов:
1. robustness_check
Ответь:
=== ROBUSTNESS COMPLETE ===
Таблица: результат, уверенность high/medium/low, обоснование.""",

"recommendations": """Фаза: ПРАКТИЧЕСКИЕ РЕКОМЕНДАЦИИ.
1. segment_analysis по turnover_intention и burnout_score
2. Конкретные рекомендации
Ответь:
=== RECOMMENDATIONS COMPLETE ===
5-7 рекомендаций с приоритетами и цифрами.""",

"synthesis": """Фаза: ФИНАЛЬНЫЙ ОТЧЁТ в Markdown.
Структура:
## 1. Данные и качество
## 2. Методология
## 3. Разведочный анализ
## 4. Гипотезы и результаты (таблица)
## 5. Каузальный анализ
## 6. Устойчивость результатов (таблица)
## 7. Группы риска
## 8. Рекомендации
## 9. Ограничения
## 10. Дальнейшие исследования
Цитируй конкретные числа. APA-стиль.""",
}
print(f"{len(PHASE_PROMPTS)} фаз")

# ## 7. Цикл агента

def execute_tool(name, arguments):
    func = TOOLS_REGISTRY.get(name)
    if not func: return json.dumps({"error": f"Неизвестный инструмент: {name}"})
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else arguments
        return json.dumps(func(**args), ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})

def run_phase(phase, extra=""):
    prompt = PHASE_PROMPTS[phase]
    if extra: prompt += f"\n\nКонтекст:\n{extra}"
    messages = [{"role": "user", "content": f"Начни фазу: {phase}"}]

    print(f"\n{'='*60}\n ФАЗА: {phase.upper()}\n{'='*60}")

    for i in range(MAX_ITERATIONS):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role":"system","content":prompt}] + messages,
                tools=TOOL_SCHEMAS, tool_choice="auto", max_tokens=4096)
        except Exception as e:
            print(f"   Ошибка API: {e}"); break

        msg = resp.choices[0].message

        if msg.content and not msg.tool_calls:
            print(f"\n Завершено ({i+1} итераций)")
            print(textwrap.indent(msg.content[:2000], "   "))
            state.agent_reasoning.append({"phase": phase, "text": msg.content})
            return msg.content

        if msg.tool_calls:
            messages.append({"role":"assistant","content":msg.content or "",
                "tool_calls":[{"id":tc.id,"type":"function","function":{"name":tc.function.name,"arguments":tc.function.arguments}} for tc in msg.tool_calls]})
            for tc in msg.tool_calls:
                print(f"   [{i+1}] {tc.function.name}({tc.function.arguments[:80]})")
                result = execute_tool(tc.function.name, tc.function.arguments)
                print(f"      -> {result[:150]}...")
                messages.append({"role":"tool","tool_call_id":tc.id,"content":result})

        if resp.choices[0].finish_reason == "stop" and not msg.tool_calls:
            state.agent_reasoning.append({"phase": phase, "text": msg.content or ""})
            return msg.content or ""

    return "MAX_ITERATIONS"

print("Цикл готов")

# ## 8. Парсер гипотез

def parse_hypotheses(text):
    count = 0
    for line in text.split("\n"):
        line = line.strip()
        m = re.match(r"H\d+:\s*(.+?)(?:\s*\|\s*Переменные:\s*(.+?))?(?:\s*\|\s*Тест:\s*(.+?))?$", line)
        if m:
            t = m.group(1).strip()
            vs = [v.strip() for v in (m.group(2) or "").split(",") if v.strip()]
            hid = state.add_hypothesis(t, vs, generation=state.deepening_round)
            print(f"   {hid}: {t}"); count += 1
    if count == 0:
        for line in text.split("\n"):
            line = line.strip()
            if re.match(r"^(H\d+|[0-9]+[\.\)])", line) and len(line) > 15:
                t = re.sub(r"^(H\d+[:.]?\s*|[0-9]+[\.\)]\s*)", "", line).strip()
                if t:
                    hid = state.add_hypothesis(t, [], generation=state.deepening_round)
                    print(f"   {hid}: {t}"); count += 1
    print(f"   Итого: {count}")
    return count

print("Парсер готов")

# ## 9. Запуск

quality_result = run_phase("data_quality")

eda_result = run_phase("eda")

hyp_result = run_phase("hypotheses", extra=f"Находки EDA:\n{eda_result}")
parse_hypotheses(hyp_result)

hyp_list = "\n".join([f"{h.id}: {h.text} ({h.variables})" for h in state.hypotheses])
test_result = run_phase("testing", extra=f"Гипотезы:\n{hyp_list}")

for rd in range(MAX_DEEPENING):
    state.deepening_round = rd + 1
    summary = json.dumps(state.get_summary(), ensure_ascii=False, default=str)
    reflection = run_phase("reflection", extra=summary)
    if "REFLECTION COMPLETE" in reflection and any(w in reflection.lower() for w in ["не нужно","достаточно","завершаем"]):
        print(f"   Рефлексия завершена (раунд {rd+1})"); break
    new = parse_hypotheses(reflection)
    if new > 0:
        new_hyps = "\n".join([f"{h.id}: {h.text}" for h in state.hypotheses if h.status=="pending" and h.generation==state.deepening_round])
        run_phase("testing", extra=f"Новые гипотезы:\n{new_hyps}")
    else:
        break

causal_ctx = json.dumps(state.test_log[:10], ensure_ascii=False, default=str)[:4000]
run_phase("causal", extra=f"Результаты тестов:\n{causal_ctx}")

run_phase("robustness", extra=f"Тесты:\n{causal_ctx}")

run_phase("recommendations", extra=json.dumps(state.get_summary(), ensure_ascii=False, default=str)[:4000])

print(f"\n{'='*60}")
print(f"Итого: {len(state.hypotheses)} гипотез, {len(state.test_log)} тестов, {len(state.causal_findings)} каузальных, {len(state.robustness_log)} робастности, {len(state.plots)} графиков")
print(f"{'='*60}")

# ## 10. Отчёт

full = json.dumps(state.get_summary(), ensure_ascii=False, default=str)
tests = json.dumps(state.test_log, ensure_ascii=False, default=str)[:6000]
causal = json.dumps(state.causal_findings, ensure_ascii=False, default=str)[:3000]
robust = json.dumps(state.robustness_log, ensure_ascii=False, default=str)[:3000]
recs = json.dumps(state.recommendations, ensure_ascii=False, default=str)[:3000]

report = run_phase("synthesis", extra=f"Состояние:\n{full}\n\nТесты:\n{tests}\n\nКаузальный:\n{causal}\n\nРобастность:\n{robust}\n\nРекомендации:\n{recs}")

with open("research_report.md", "w", encoding="utf-8") as f:
    f.write(report)
print("Отчёт сохранён: research_report.md")

# from IPython.display import display, Image, Markdown

for p in state.plots:
    print(p)
    try: # display(Image(filename=p))
    except: pass

# display(Markdown(report))

# ## 11. Отладка

print("ЛОГ РАССУЖДЕНИЙ:")
for e in state.agent_reasoning:
    print(f"\n--- {e['phase'].upper()} ---")
    print(e['text'][:1500])

print("ВСЕ ТЕСТЫ:")
for i, t in enumerate(state.test_log, 1):
    short = {k:v for k,v in t.items() if k not in ("contingency_table","coefficients","posthoc","VIF")}
    print(f"[{i}] {json.dumps(short, ensure_ascii=False, default=str)[:250]}")

print("КАУЗАЛЬНЫЕ НАХОДКИ:")
for f_item in state.causal_findings:
    print(json.dumps(f_item, ensure_ascii=False, default=str)[:300])

print("РОБАСТНОСТЬ:")
for r in state.robustness_log:
    print(f"{r['var1']}<->{r['var2']}: {r['overall_confidence']} | {json.dumps(r['checks'], default=str)[:200]}")

print("ДЕРЕВО ГИПОТЕЗ:")
for h in state.hypotheses:
    indent = "  " * h.generation
    icon = "[+]" if h.status=="confirmed" else "[-]" if h.status=="rejected" else "[?]"
    parent = f" <- {h.parent_id}" if h.parent_id else ""
    print(f"{indent}{icon} {h.id}: {h.text}{parent}")
