# HIT140 Assessment 2 — Analytic Task 4
Goalkeeper Saves by Tournament Progress
Prakash Subedi (407192)

## Question
Do goalkeepers of teams eliminated in the group stage face a heavier
Save workload than those whose teams advanced to the knockout rounds?

## Data
FBref Player Goalkeeping table, FIFA World Cup 2026 — 62 goalkeepers, 48 teams.

## Method
Player records aggregated to 48 team records to preserve independence.
The tournament stage is derived from the goalkeeper's start. Welch's two-sample
t-test on saves per 90 minutes.

## Results
Eliminated 3.354 vs advanced 2.571.
Difference 0.783, 95% CI [0.010, 1.555].
t = 2.040, p = 0.0535, Cohen's d = 0.662.

## Run
python3 final_goalkeeper_analysis.py
