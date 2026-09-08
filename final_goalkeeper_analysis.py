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


# STEP 2: Group the players by team
# Some teams used two goalkeepers, so their saves, minutes
# and starts are added together to give one record per team.
team = df.groupby("team", as_index=False).agg({
    "saves": "sum",
    "minutes": "sum",
    "starts": "sum"
})

print("Number of teams after grouping:", team.shape[0])


# STEP 3: Create the grouping variable
# One goalkeeper starts each match, so starts sum to matches played.
# The group stage is 3 matches, so more than 3 means the team advanced.
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

above = (eliminated > advanced.median()).sum()
print(f"\nEliminated teams above the advanced median: {above} of {len(eliminated)}")


# STEP 7: Boxplot
plt.figure(figsize=(7, 5))
plt.boxplot([eliminated, advanced],
            tick_labels=["Eliminated\n(group stage)", "Advanced\n(knockout)"])
plt.ylabel("Saves per 90 minutes")
plt.title("Goalkeeper Saves per 90 by Tournament Progress")
plt.savefig("goalkeeper_saves_boxplot.png", dpi=150, bbox_inches="tight")
print("\nBoxplot saved as goalkeeper_saves_boxplot.png")


# STEP 8: Confidence intervals
# The t distribution is used because the population SD is unknown.
print("\n----- 95% CONFIDENCE INTERVALS -----")

n1 = len(eliminated)
mean1 = eliminated.mean()
ci1 = st.t.interval(0.95, n1 - 1, loc=mean1, scale=st.sem(eliminated))
print(f"Eliminated: mean = {mean1:.3f}, 95% CI = [{ci1[0]:.3f}, {ci1[1]:.3f}]")

n2 = len(advanced)
mean2 = advanced.mean()
ci2 = st.t.interval(0.95, n2 - 1, loc=mean2, scale=st.sem(advanced))
print(f"Advanced:   mean = {mean2:.3f}, 95% CI = [{ci2[0]:.3f}, {ci2[1]:.3f}]")

# Difference between the two means.
# Welch-Satterthwaite degrees of freedom, so the interval matches
# the unpooled standard error and agrees with the t-test below.
diff = mean1 - mean2
var1 = eliminated.var()
var2 = advanced.var()

se_diff = np.sqrt(var1 / n1 + var2 / n2)
df_diff = ((var1 / n1 + var2 / n2) ** 2 /
           ((var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1)))

ci_diff = st.t.interval(0.95, df_diff, loc=diff, scale=se_diff)
print(f"Difference: {diff:.3f}, 95% CI = [{ci_diff[0]:.3f}, {ci_diff[1]:.3f}]")
print(f"Welch degrees of freedom = {df_diff:.2f}")


# STEP 9: Two-sample t-test
# H0: the two population means are equal
# H1: the two population means are not equal
# Significance level alpha = 0.05
print("\n----- TWO-SAMPLE t-TEST -----")

levene_stat, levene_p = st.levene(eliminated, advanced)
print(f"Levene's test p-value = {levene_p:.4f}")

# Welch's version, because the groups are unequal in size.
t_stat, p_value = st.ttest_ind(eliminated, advanced, equal_var=False)
print(f"t statistic = {t_stat:.3f}")
print(f"p-value     = {p_value:.4f}")

if p_value < 0.05:
    print("Result: reject the null hypothesis")
else:
    print("Result: fail to reject the null hypothesis")

pooled_sd = np.sqrt((var1 + var2) / 2)
cohens_d = diff / pooled_sd
print(f"Cohen's d   = {cohens_d:.3f}")


# STEP 10: Normality check
# Levene tested equal variance; Shapiro-Wilk tests normality.
print("\n----- ASSUMPTION CHECKS -----")
print(f"Shapiro-Wilk eliminated: p = {st.shapiro(eliminated).pvalue:.4f}")
print(f"Shapiro-Wilk advanced:   p = {st.shapiro(advanced).pvalue:.4f}")


# STEP 11: Non-parametric robustness check
# Mann-Whitney U assumes no distribution, so agreement with
# the t-test strengthens the result.
print("\n----- ROBUSTNESS: MANN-WHITNEY U -----")
u_stat, u_p = st.mannwhitneyu(eliminated, advanced, alternative="two-sided")
print(f"U statistic = {u_stat:.1f}")
print(f"p-value     = {u_p:.4f}")


# STEP 12: Post-hoc power analysis
# How likely this design was to detect an effect of the size found.
print("\n----- POWER ANALYSIS -----")
analysis = TTestIndPower()
ratio = n2 / n1

power = analysis.power(effect_size=cohens_d, nobs1=n1, ratio=ratio, alpha=0.05)
needed = analysis.solve_power(effect_size=cohens_d, power=0.8,
                              ratio=ratio, alpha=0.05)

print(f"Achieved power = {power:.3f}")
print(f"Teams needed for 80% power = {needed * (1 + ratio):.0f} total")


# STEP 13: Sensitivity analysis
# A sharper contrast: quarter-finalists or better against the rest.
# Exploratory only, because this split was chosen after seeing the
# main result, so it is a robustness check and not a second test.
print("\n----- SENSITIVITY: QUARTER-FINAL SPLIT -----")
team["reached_qf"] = team["matches_played"] >= 6
early = team[team["reached_qf"] == False]["saves_per_90"]
late = team[team["reached_qf"] == True]["saves_per_90"]

t2, p2 = st.ttest_ind(early, late, equal_var=False)
print(f"Groups: n = {len(early)} vs {len(late)}")
print(f"Means:  {early.mean():.3f} vs {late.mean():.3f}")
print(f"t = {t2:.3f}, p = {p2:.4f}")
print("Note: exploratory split, reported as a robustness check only.")