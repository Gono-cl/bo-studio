# BenchBO User Guide

## 1. What BenchBO is for

BenchBO helps you design, execute, and analyze Bayesian Optimization campaigns through a graphical interface.

Typical use:

- start with the classroom to understand BO strategy choices,
- run optimization campaigns in the single-objective or multiobjective pages,
- compare, save, and revisit results through the analysis and database pages.

## 2. Recommended first run

If you are new to the software, start with:

1. `Bayesian Optimization Classroom`
2. `4) Chemist Workflow`
3. `5) Multiobjective Decisions`

That path gives a complete first demonstration without requiring external data entry.

For a reviewer-oriented reproducible walkthrough, see `docs/EXAMPLE_WORKFLOW.md`.

## 3. Navigation overview

Main sections:

- `Home`
- `Single Objective Optimization`
- `Multi Objective Optimization`
- `Data Analysis`
- `Bayesian Optimization Classroom`
- `Experiment Database`

## 4. Single Objective Optimization workflow

Recommended sequence:

1. Define variables and bounds.
2. Set the response name and optimization direction.
3. Configure BO settings such as `n_init`, total budget, initialization strategy, and acquisition function.
4. Initialize the campaign.
5. Enter measured results for the suggested experiments.
6. Request the next BO suggestion and continue iteratively.
7. Save the campaign when satisfied.

Available controls include save, resume, and reuse workflows so campaigns can be continued across sessions.

## 5. Multi Objective Optimization workflow

Recommended sequence:

1. Define variables and bounds.
2. Select multiple objectives and their directions.
3. Configure initialization and acquisition settings.
4. Initialize the campaign and record measured results.
5. Review Pareto behavior and tradeoffs.
6. Save the campaign for later comparison or continuation.

## 6. Bayesian Optimization Classroom workflow

The classroom is organized as a learning path:

1. `BO Intuition`
2. `BO Mechanics`
3. `Chemist Workflow`
4. `Multiobjective Decisions`

Use `Beginner` mode for a low-math conceptual path and `Advanced` mode for additional theory.

## 7. Core parameter glossary

- `n_init`: number of initial experiments before BO suggestions dominate.
- `total iterations` or `total experiments`: total campaign budget.
- `init method`: how initial points are distributed, for example `Random`, `LHS`, or `Halton`.
- `acquisition function (AF)`: rule used to choose the next experiment, for example `EI`, `PI`, or `LCB`.
- `xi`: exploration pressure for `EI` and `PI`.
- `kappa`: exploration pressure for `LCB`.
- `seed`: reproducibility control for stochastic steps.
- `measurement noise`: simulated observation noise where relevant.
- `failure probability`: chance of a failed run in the relevant classroom workflow.

## 8. Save, resume, and reuse behavior

- `Save` writes campaign state and results to local storage.
- `Resume` loads a prior campaign and continues from that exact state.
- `Reuse` imports previous experiments as seeds for a new run.

These workflows are useful when comparing strategy changes without losing campaign history.

## 9. Local desktop executable usage

If distributed as a Windows executable:

- launch `BenchBO.exe`,
- no Python installation is required for the end user,
- campaign data is stored locally in the application storage context.

For build details, see `PACKAGING.md`.

## 10. Troubleshooting

If something looks stale or inconsistent:

1. Close all running BenchBO windows or processes.
2. Relaunch from `python run_bo_studio.py` (source) or `BenchBO.exe` (packaged).
3. Re-test the same sequence.

If a page crashes, capture:

- the section name,
- the exact button or selection sequence,
- the full traceback text.

That is usually enough to reproduce and patch the issue quickly.

