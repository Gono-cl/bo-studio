# BO Classroom Guide

This guide explains how to use BO Classroom as a structured learning path.

## Learning path

1. BO Intuition
2. BO Mechanics
3. Chemist Workflow
4. Multiobjective Decisions

## Recommended usage

1. Start in Beginner mode.
2. Change one parameter at a time.
3. Observe how GP and AF behavior changes.
4. Compare EI, PI, and LCB strategies.
5. Repeat with Advanced mode for deeper theory.

## Parameters that matter most

- Initial experiments (`n_init`)
- Initialization method
- Acquisition function
- Exploration controls (`xi`, `kappa`)
- Total iterations
- Measurement noise and failure probability

## Chemistry perspective

In reaction optimization, objective functions are often yield, selectivity, productivity, or a weighted combination.

BO helps you balance exploitation of known good regions with exploration of uncertain regions.

## After classroom

- Use Single Objective Optimization for one target metric.
- Use Multi Objective Optimization for tradeoff analysis.
- Use Data Analysis and Experiment Database to compare campaigns.
