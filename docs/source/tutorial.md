# Tutorial: Your First BO Studio Campaign (15 minutes)

This tutorial gives you one full end-to-end workflow in BO Studio.

Goal:
- run one single-objective campaign,
- understand why BO proposes each next experiment,
- save results for later comparison.

## Before you start

You should have:
- BO Studio running,
- a basic idea of your variables and target metric,
- 10-15 minutes.

If you are completely new to BO, first review **BO Classroom -> BO Intuition**.

## Step 1: Define your objective

Open **Single Objective Optimization**.

Set:
- objective function: `Yield` (or your metric),
- direction: `Maximize`.

Practical note:
- keep the objective measurable and comparable across runs.

## Step 2: Define variables and bounds

Add your process variables (example):
- Temperature: 20 to 120 C
- Catalyst loading: 0.00 to 1.00
- Pressure: 1 to 20 bar
- Residence time: 30 to 300 s

Rule of thumb:
- choose realistic bounds you can actually execute in lab.

## Step 3: Configure BO settings

Use this starter setup:
- `n_init`: 6
- `total iterations`: 20
- `init method`: LHS
- `acquisition`: EI
- `seed`: 42

Why this setup:
- LHS improves initial space coverage,
- EI is a balanced default for most campaigns,
- fixed seed improves reproducibility while learning.

## Step 4: Initialize and enter results

Click the button that initializes/suggests initial experiments.

For each suggested experiment:
1. run it (or simulate it),
2. enter measured objective value,
3. submit.

After initial points are recorded, BO starts proposing new experiments sequentially.

## Step 5: Continue BO suggestions

For each new suggestion:
1. run experiment,
2. submit measured result,
3. inspect progress plot and candidate behavior.

What to observe:
- early stage: broader exploration,
- later stage: stronger exploitation near high-performing regions.

## Step 6: Interpret model behavior

Use the charts to answer:
- Are suggestions clustering too early?
- Is uncertainty still high in unsampled regions?
- Is improvement plateauing?

If campaign stalls, test one change at a time:
- increase total iterations,
- increase exploration pressure (`xi` or `kappa` depending on AF),
- increase `n_init` for larger search spaces.

## Step 7: Save campaign

When you have a meaningful trajectory:
- save campaign,
- add notes about settings and best run,
- keep this as baseline for comparison.

## Step 8: Compare with an alternative strategy

Repeat quickly with one controlled change, for example:
- EI -> LCB, or
- `n_init` from 6 -> 10.

Then compare:
- best objective value reached,
- number of iterations to reach near-best,
- stability/consistency of suggestions.

This is where BO Studio becomes most useful: strategy comparison, not just one run.

## Common beginner mistakes

- Too few initial points for high-dimensional spaces.
- Changing many settings at once and losing interpretability.
- Treating noisy measurements as exact truth.
- Forgetting to save campaign metadata.

## What to do next

After this tutorial:
- Study **classroom_guide** to understand behavior in more depth.
- Use **multi** if you need yield-productivity tradeoff optimization.
- Use **experiment_database** to build a reusable campaign library.
