# BO Studio User Guide

## 1. What BO Studio is for

BO Studio helps you design, execute, and analyze Bayesian Optimization campaigns through a graphical interface.

Typical use:
- start with BO Classroom to understand strategy choices,
- run optimization campaigns in Single Objective or Multi Objective pages,
- compare and store results in Data Analysis and Experiment Database.

## 2. Navigation overview

Main sections:
- Home
- Single Objective Optimization
- Multi Objective Optimization
- Data Analysis
- Bayesian Optimization Classroom
- Experiment Database

## 3. Single Objective Optimization workflow

Recommended sequence:

1. Define variables and bounds.
2. Set response/objective function and direction (maximize or minimize).
3. Configure BO settings:
- initial experiments (`n_init`),
- total iterations,
- initialization strategy,
- acquisition function and exploration controls.
4. Initialize campaign.
5. Enter measured results.
6. Request next BO suggestion and continue iteratively.
7. Save campaign when satisfied.

Available controls include save/resume/reuse workflows so campaigns can be continued across sessions.

## 4. Multi Objective Optimization workflow

Recommended sequence:

1. Define variables and bounds.
2. Select multiple objectives and objective directions.
3. Configure initialization and acquisition settings.
4. Initialize and record results.
5. Review Pareto behavior and tradeoffs.
6. Save campaign for comparison or future continuation.

## 5. BO Classroom workflow

The classroom is designed as a learning path:

1. BO Intuition
2. BO Mechanics
3. Chemist Workflow
4. Multiobjective Decisions

Use Beginner mode for low-math conceptual learning and Advanced mode for deeper theoretical detail.

## 6. Core parameter glossary

- `n_init`: number of initial experiments before BO suggestions dominate.
- `total iterations`: total campaign budget.
- `init method`: how initial points are distributed (for example Random, LHS, Halton).
- `acquisition function (AF)`: policy for selecting the next experiment (for example EI, PI, LCB).
- `xi`: exploration pressure for EI/PI.
- `kappa`: exploration pressure for LCB.
- `seed`: reproducibility control for stochastic steps.
- `measurement noise`: simulated observation noise (if enabled).
- `failure probability`: chance of a failed run in relevant simulations.

## 7. Save, resume, and reuse behavior

- Save writes campaign state and results to local storage.
- Resume loads a prior campaign and continues from that exact state.
- Reuse imports previous experiments as seeds for a new run.

These workflows are important for comparing strategy changes without losing campaign history.

## 8. Local desktop executable usage

If distributed as Windows executable:

- Launch `BOStudio.exe`.
- End users do not need Python.
- Campaign data is stored locally per user context.

For build details, see `PACKAGING.md`.

## 9. Troubleshooting

If something looks stale or inconsistent:

1. Close all running BO Studio windows/processes.
2. Relaunch from `python run_bo_studio.py` (source) or `BOStudio.exe` (packaged).
3. Re-test.

If a page crashes, capture:
- section name,
- exact button/selection sequence,
- full traceback text.

That information is enough to reproduce and patch quickly.

