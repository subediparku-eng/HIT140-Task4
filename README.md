# HIT140 Assessment 2 — Analytic Task 4
Goalkeeper Saves by Tournament Progress
Prakash Subedi (407192)

## Question
Do goalkeepers of eliminated teams face a heavier save workload than
those whose teams advanced?

## Data
FBref Player Goalkeeping table, FIFA World Cup 2026 — 62 goalkeepers, 48 teams.

## Method
62 player records aggregated to 48 team records to preserve independence.
Stage derived from goalkeeper starts. Welch's two-sample t-test on saves per 90.

## Results
Eliminated 3.354 vs advanced 2.571.
Difference 0.783, 95% CI [-0.013, 1.578].
t = 2.040, p = 0.0535, Cohen's d = 0.662.

Levene 0.244 · Shapiro-Wilk 0.258 / 0.353 · Mann-Whitney U 0.088.
Power 0.56; about 82 teams needed for 0.80.

## Conclusion
Not significant, but a medium-to-large effect on an underpowered design.

## Run
python3 final_goalkeeper_analysis.py
