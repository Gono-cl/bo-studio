# BO Classroom Guide

This page explains how to use the BO Classroom in BO Studio as a structured learning path.

## Goal

The classroom is designed to help users understand how Bayesian Optimization works before running real campaigns.

You can use it in two modes:

- Beginner: low-math explanations, chemistry-first interpretation.
- Advanced: deeper theory, equations, and parameter effects.

## Recommended learning order

1. BO Intuition
2. BO Mechanics
3. Chemist Workflow
4. Multiobjective Decisions

Each step builds on the previous one.

## How to study each section

1. Read the context card first.
2. Change only one parameter at a time.
3. Run the simulation and observe how suggestions change.
4. Compare behavior across acquisition functions (EI, PI, LCB).
5. Record what reduced the number of iterations needed to reach high objective values.

## Key parameters to focus on

- Initial experiments (`n_init`)
- Initialization strategy (Random, LHS, Halton, Maximin LHS)
- Acquisition function (`EI`, `PI`, `LCB`)
- Exploration parameters (`xi`, `kappa`)
- Iteration budget (`total iterations`)
- Measurement noise and failure probability (when enabled)

## Chemistry interpretation

In reaction optimization, the objective function is often yield, selectivity, productivity, or a weighted tradeoff.

The classroom helps interpret BO suggestions as practical decisions:

- exploit near known good conditions,
- or explore uncertain regions that may hide better performance.

## Transition to campaign pages

After finishing the classroom:

- Use Single Objective Optimization for one target metric.
- Use Multi Objective Optimization for tradeoff problems.
- Save campaigns and compare settings in Data Analysis.

## Common mistakes

- Using too few initial points in larger search spaces.
- Changing many parameters simultaneously and losing interpretability.
- Comparing campaigns with different seeds without noting the differences.

## Practical checklist

- Keep notes for every run.
- Repeat high-performing points to confirm robustness.
- Use the minimum iterations needed to reach stable top performance.
