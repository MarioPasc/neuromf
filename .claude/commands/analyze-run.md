---
allowed-tools:
  - Read
  - Glob
  - Grep
  - Bash
  - Write
  - Edit
  - Agent
description: "Analyze a completed training run against MOTFM baseline and propose next experiments"
argument-hint: "<run_dir, e.g. /media/mpascual/Sandisk2TB/research/neuromf/results/runs/run_20260306_041930>"
---

# Analyze Training Run

Launch a results-analyst agent to perform comprehensive analysis of a completed training run.

## Steps

1. The `$ARGUMENTS` should be the path to the run directory (containing `checkpoints/`, `diagnostics/`, `samples/`).
2. Launch the `results-analyst` agent with the run directory.
3. The agent will:
   - Load training telemetry from `diagnostics/aggregate_results/training_summary.json`
   - Compare against MOTFM baseline at NFE=1, 10, 50
   - Identify failure modes with literature citations
   - Propose prioritized next experiments
   - Write analysis to `docs/analysis/`
