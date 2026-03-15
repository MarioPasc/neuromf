---
allowed-tools:
  - Read
  - Glob
  - Grep
  - Bash
  - Agent
description: "Validate code and config changes before submitting a multi-day training job to Picasso"
argument-hint: "<config_path, e.g. configs/picasso/train_meanflow.yaml>"
---

# Pre-Flight Validation

Launch a pre-flight-validator agent to check that the current code and config are ready for a training run.

## Steps

1. If `$ARGUMENTS` is provided, use it as the config path. Otherwise, use `configs/picasso/train_meanflow.yaml`.
2. Launch the `pre-flight-validator` agent with the config path.
3. The agent will run all validation checks and report READY or BLOCKED.

Training runs on Picasso cost 3-8 A100 GPUs for 2-5 days. This validation catches config errors, forbidden JVP combos, shape mismatches, and test regressions BEFORE submitting the SLURM job.
