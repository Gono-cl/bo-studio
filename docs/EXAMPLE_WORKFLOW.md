# BenchBO Example Workflow

This document gives a reviewer-friendly first run for BenchBO. It is designed to verify the core classroom functionality without external credentials, lab hardware, or manually prepared input files.

## Goal

Use the built-in classroom simulations to confirm that:

- the app launches correctly,
- interactive controls respond,
- simulated chemistry campaigns run end-to-end,
- tables, metrics, and plots are rendered,
- no hidden setup beyond installation is required.

## Launch

From a source checkout:

```bash
python run_bo_studio.py
```

Or, if using the packaged Windows app:

`BenchBO.exe`

## Workflow A: Chemist Workflow classroom page

1. Open `Bayesian Optimization Classroom` from the main navigation.
2. In the sidebar, keep `Teaching mode` set to `Beginner`.
3. In `Learning path`, choose `4) Chemist Workflow`.
4. Leave the default settings unchanged for the first run.
5. Click `Run simulated campaign`.

Expected outcomes:

- The page should display metrics for successful runs, failed runs, and best measured yield.
- A full experiment table should appear below the controls.
- A trend plot should appear showing campaign progress across experiments.
- A parallel-coordinates plot should appear for the successful 4D conditions.

Useful follow-up checks:

- Change `Initial experiments` or `Acquisition function`, rerun, and confirm the campaign behavior changes.
- Increase `Measurement noise (sigma)` and confirm that the campaign becomes less stable or less efficient.

## Workflow B: Multiobjective Decisions classroom page

1. Stay in `Bayesian Optimization Classroom`.
2. In `Learning path`, choose `5) Multiobjective Decisions`.
3. Set `How to generate the classroom campaign` to `Scalarized SO BO on Yield + Purity`.
4. Set `Scalarization method` to `Weighted sum`.
5. Set `Weight policy` to `Fixed weights`.
6. Keep `Yield weight in BO` at `0.50`.
7. Leave the default experiment budget unchanged for the first run.
8. Click `Run simulated scalarized MO classroom campaign`.

Expected outcomes:

- The page should display metrics for completed runs, BO suggestions, best yield, and best purity.
- A results table should appear with `Yield`, `Purity`, and selection-stage information.
- A Pareto-front summary should appear, including the front size.
- A scatter plot should appear with dominated sampled points and a highlighted Pareto frontier.

Useful follow-up checks:

- Change the yield weight from `0.50` to `0.80`, rerun, and confirm that the sampled tradeoff pattern changes.
- Switch to `Yield-driven BO + Pareto analysis (contrast case)` and confirm that the explanatory caption changes accordingly.

## What this verifies

Completing the two workflows above verifies the main publication-facing claims of the classroom module:

- BenchBO can be installed and launched locally.
- The software contains reproducible chemistry-oriented BO teaching workflows.
- Single-objective and multiobjective classroom simulations both execute successfully.
- The GUI provides interpretable outputs without requiring code-level interaction from the end user.

## Related documentation

- `README.md`
- `docs/USER_GUIDE.md`
- `PACKAGING.md`
