# LEDGER — Results

All numbers are on the **official RelBench test split**, scored by RelBench's
own evaluator (`python -m relbench.submit`, relbench 3.0.1): **31/31 tasks
valid**, all three boards complete.

Comparisons are against the **official RelBench leaderboard**
(<https://star-project.stanford.edu/relbench/leaderboard/>), read from its
source of truth, `leaderboard/leaderboard.json` in `stanford-star/relbench`.
Only entries that cover a full board are listed, which is how the board itself
ranks. Regenerate with `python scripts/leaderboard.py --markdown --refresh`.

**In-context.** The board's rule is that an in-context submission may not do
gradient-based training on the target database. LEDGER uses **no task labels**,
but it does pretrain on the target database, so it is **not in-context** and is
listed as such.

---

## 1. Classification — test AUROC (%), higher is better

### 1.1 Per task

| Database | Task | LEDGER | Best on board |
|---|---|---|---|
| rel-amazon | user-churn | 66.62 | 71.55 |
| rel-amazon | item-churn | 79.71 | 83.18 |
| rel-avito | user-visits | 64.91 | 83.00 |
| rel-avito | user-clicks | 65.62 | 79.70 |
| rel-event | user-repeat | 76.17 | 81.16 |
| rel-event | user-ignore | 71.19 | 88.89 |
| rel-f1 | driver-dnf | 67.87 | 83.27 |
| rel-f1 | driver-top3 | 73.86 | 93.57 |
| rel-hm | user-churn | 66.99 | 71.62 |
| rel-stack | user-engagement | 89.02 | 91.51 |
| rel-stack | user-badge | 85.23 | 89.49 |
| rel-trial | study-outcome | 67.44 | 82.08 |

**Mean: 72.89** over 12/12 tasks.

### 1.2 Leaderboard

| # | Method | In-context | Mean |
|---|---|---|---|
| 1 | Kapso (Leeroo Team) | no | 81.19 |
| 2 | Kurve-RSC + CatBoost | no | 79.85 |
| 3 | RT-PluRel (fine-tuned) | no | 78.96 |
| 4 | RT | no | 77.76 |
| 5 | RT-PluRel (in-context) | yes | 74.11 |
| 6 | GNN + LightGBM | no | 73.89 |
| **7** | **LEDGER (ours)** | **no** | **72.89** |
| 8 | GNN | no | 72.83 |
| 9 | Entity Mean | yes | 65.79 |
| 10 | LightGBM | no | 63.06 |
| 11 | Entity Median | yes | 59.44 |
| 12 | Majority | yes | 50.00 |
| 13 | Random | yes | 49.80 |

**Position: 7 of 13.**

## 2. Regression — test NMAE, lower is better

### 2.1 Per task

| Database | Task | LEDGER | Best on board |
|---|---|---|---|
| rel-amazon | user-ltv | 0.2635 | 0.2377 |
| rel-amazon | item-ltv | 0.0929 | 0.0655 |
| rel-avito | ad-ctr | 0.3878 | 0.3190 |
| rel-event | user-attendance | 0.3444 | 0.3157 |
| rel-f1 | driver-position | 0.6557 | 0.3440 |
| rel-hm | item-sales | 0.1536 | 0.0634 |
| rel-stack | post-votes | 0.1243 | 0.1221 |
| rel-trial | study-adverse | 0.1572 | 0.0872 |
| rel-trial | site-success | 0.8722 | 0.6591 |

**Mean: 0.3391** over 9/9 tasks.

### 2.2 Leaderboard

| # | Method | In-context | Mean |
|---|---|---|---|
| 1 | Kapso (Leeroo Team) | no | 0.2476 |
| 2 | Kurve-RSC + CatBoost | no | 0.2777 |
| 3 | RT-PluRel (fine-tuned) | no | 0.2814 |
| 4 | RT | no | 0.2926 |
| 5 | GNN | no | 0.3130 |
| 6 | RT-PluRel (in-context) | yes | 0.3299 |
| 7 | GNN + LightGBM | no | 0.3320 |
| **8** | **LEDGER (ours)** | **no** | **0.3391** |
| 9 | LightGBM | no | 0.3403 |
| 10 | Global Median | yes | 0.3610 |
| 11 | Entity Median | yes | 0.4283 |
| 12 | Global Mean | yes | 0.4530 |
| 13 | Entity Mean | yes | 0.4553 |
| 14 | Zero | yes | 0.4932 |

**Position: 8 of 14.**

## 3. Recommendation — test MAP (%), higher is better

### 3.1 Per task

| Database | Task | LEDGER | Best on board |
|---|---|---|---|
| rel-amazon | user-item-purchase | 1.13 | 2.54 |
| rel-amazon | user-item-rate | 1.17 | 2.31 |
| rel-amazon | user-item-review | 0.79 | 2.95 |
| rel-avito | user-ad-visit | 0.68 | 4.20 |
| rel-f1 | driver-circuit-compete | 60.53 | 87.95 |
| rel-hm | user-item-purchase | 2.41 | 3.26 |
| rel-stack | user-post-comment | 2.34 | 13.13 |
| rel-stack | post-post-related | 1.11 | 21.78 |
| rel-trial | condition-sponsor-run | 5.98 | 12.28 |
| rel-trial | site-sponsor-run | 11.33 | 33.33 |

**Mean: 8.75** over 10/10 tasks.

### 3.2 Leaderboard

| # | Method | In-context | Mean |
|---|---|---|---|
| 1 | Kapso (Leeroo Team) | no | 18.37 |
| 2 | ID-GNN | no | 9.33 |
| **3** | **LEDGER (ours)** | **no** | **8.75** |
| 4 | LightGBM | no | 7.59 |
| 5 | Global Popularity | yes | 5.87 |
| 6 | Past Visit | yes | 5.01 |
| 7 | GNN | no | 4.14 |

**Position: 3 of 7.**
