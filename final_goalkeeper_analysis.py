# HIT140 Foundations of Data Science - Assessment 2
# Analytic Task: Goalkeeper Saves by Tournament Progress
# Prakash Subedi - Student ID 407192

import pandas as pd
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from statsmodels.stats.power import TTestIndPower


# STEP 1: Read the dataset
df = pd.read_csv("data/wc2026_player_goalkeeping.csv")

print("Number of player records:", df.shape[0])
print("Number of teams:", df["team"].nunique())


# STEP 2: Data wrangling - group the players by team
# Some teams used two goalkeepers, so we add their saves,
# minutes and starts together to get one record per team.
team = df.groupby("team", as_index=False).agg({
    "saves": "sum",
    "minutes": "sum",
    "starts": "sum"
})

print("Number of teams after grouping:", team.shape[0])


# STEP 3: Create the grouping variable
# Each match has one starting goalkeeper, so the total number of
# starts equals the number of matches the team played.
# The group stage has 3 matches, so more than 3 means the team advanced.
team["matches_played"] = team["starts"]
team["advanced"] = team["matches_played"] > 3

print("\nMatches played by each team:")
print(team["matches_played"].value_counts().sort_index())


# STEP 4: Create the variable we are analysing
team["saves_per_90"] = team["saves"] / (team["minutes"] / 90)

print("\nMissing values in the dataset:", team.isnull().sum().sum())


# STEP 5: Split into the two groups
eliminated = team[team["advanced"] == False]["saves_per_90"]
advanced = team[team["advanced"] == True]["saves_per_90"]

print("\nEliminated teams:", len(eliminated))
print("Advanced teams:", len(advanced))


# STEP 6: Descriptive statistics
print("\n----- DESCRIPTIVE STATISTICS -----")
print("\nEliminated group:")
print(f"Mean   = {eliminated.mean():.3f}")
print(f"Median = {eliminated.median():.3f}")
print(f"SD     = {eliminated.std():.3f}")
print(f"Min    = {eliminated.min():.3f}")
print(f"Max    = {eliminated.max():.3f}")

print("\nAdvanced group:")
print(f"Mean   = {advanced.mean():.3f}")
print(f"Median = {advanced.median():.3f}")
print(f"SD     = {advanced.std():.3f}")
print(f"Min    = {advanced.min():.3f}")
print(f"Max    = {advanced.max():.3f}")


# STEP 7: Boxplot
plt.figure(figsize=(7, 5))
plt.boxplot([eliminated, advanced],
            tick_labels=["Eliminated\n(group stage)", "Advanced\n(knockout)"])
plt.ylabel("Saves per 90 minutes")
plt.title("Goalkeeper Saves per 90 by Tournament Progress")
plt.savefig("goalkeeper_saves_boxplot.png", dpi=150, bbox_inches="tight")
print("\nBoxplot saved as goalkeeper_saves_boxplot.png")


# STEP 8: Confidence intervals
# We use the t distribution because the population SD is unknown.
print("\n----- 95% CONFIDENCE INTERVALS -----")

n1 = len(eliminated)
mean1 = eliminated.mean()
se1 = st.sem(eliminated)
ci1 = st.t.interval(0.95, n1 - 1, loc=mean1, scale=se1)
print(f"Eliminated: mean = {mean1:.3f}, 95% CI = [{ci1[0]:.3f}, {ci1[1]:.3f}]")

n2 = len(advanced)
mean2 = advanced.mean()
se2 = st.sem(advanced)
ci2 = st.t.interval(0.95, n2 - 1, loc=mean2, scale=se2)
print(f"Advanced:   mean = {mean2:.3f}, 95% CI = [{ci2[0]:.3f}, {ci2[1]:.3f}]")

# Confidence interval for the difference between the two means
diff = mean1 - mean2
se_diff = np.sqrt(eliminated.var() / n1 + advanced.var() / n2)
df_diff = n1 + n2 - 2
ci_diff = st.t.interval(0.95, df_diff, loc=diff, scale=se_diff)
print(f"Difference: {diff:.3f}, 95% CI = [{ci_diff[0]:.3f}, {ci_diff[1]:.3f}]")


# STEP 9: Two-sample t-test
# H0: the two population means are equal
# H1: the two population means are not equal
# Significance level alpha = 0.05
print("\n----- TWO-SAMPLE t-TEST -----")

levene_stat, levene_p = st.levene(eliminated, advanced)
print(f"Levene's test p-value = {levene_p:.4f}")

t_stat, p_value = st.ttest_ind(eliminated, advanced, equal_var=False)
print(f"t statistic = {t_stat:.3f}")
print(f"p-value     = {p_value:.4f}")

if p_value < 0.05:
    print("Result: reject the null hypothesis")
else:
    print("Result: fail to reject the null hypothesis")

# Effect size
pooled_sd = np.sqrt((eliminated.var() + advanced.var()) / 2)
cohens_d = diff / pooled_sd
print(f"Cohen's d   = {cohens_d:.3f}")


# STEP 10: Normality check
# Levene tested equal variance. Shapiro-Wilk tests the other
# assumption of the t-test, that each group is roughly normal.
print("\n----- ASSUMPTION CHECKS -----")
print(f"Shapiro-Wilk eliminated: p = {st.shapiro(eliminated).pvalue:.4f}")
print(f"Shapiro-Wilk advanced:   p = {st.shapiro(advanced).pvalue:.4f}")


# STEP 11: Non-parametric robustness check
# Mann-Whitney U makes no assumption about the shape of the
# distribution, so agreement with the t-test strengthens the result.
print("\n----- ROBUSTNESS: MANN-WHITNEY U -----")
u_stat, u_p = st.mannwhitneyu(eliminated, advanced, alternative="two-sided")
print(f"U statistic = {u_stat:.1f}")
print(f"p-value     = {u_p:.4f}")


# STEP 12: Post-hoc power analysis
# Quantifies how likely this design was to detect an effect of the
# size observed, rather than simply asserting the sample was small.
print("\n----- POWER ANALYSIS -----")
analysis = TTestIndPower()
ratio = len(advanced) / len(eliminated)

power = analysis.power(effect_size=cohens_d, nobs1=len(eliminated),
                       ratio=ratio, alpha=0.05)
needed = analysis.solve_power(effect_size=cohens_d, power=0.8,
                              ratio=ratio, alpha=0.05)

print(f"Achieved power = {power:.3f}")
print(f"Teams needed for 80% power = {needed * (1 + ratio):.0f} total")


# STEP 13: Sensitivity analysis
# Repeats the comparison using a sharper contrast (quarter-final or
# better). Exploratory only: this split was chosen after seeing the
# main result, so it is reported as a robustness check and not as a
# second hypothesis test.
print("\n----- SENSITIVITY: QUARTER-FINAL SPLIT -----")
team["reached_qf"] = team["matches_played"] >= 6
early = team[team["reached_qf"] == False]["saves_per_90"]
late = team[team["reached_qf"] == True]["saves_per_90"]

t2, p2 = st.ttest_ind(early, late, equal_var=False)
print(f"Groups: n = {len(early)} vs {len(late)}")
print(f"Means:  {early.mean():.3f} vs {late.mean():.3f}")
print(f"t = {t2:.3f}, p = {p2:.4f}")
print("Note: exploratory split, reported as a robustness check only.")