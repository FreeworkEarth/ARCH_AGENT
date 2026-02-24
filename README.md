# ARCH_AGENT — Software Architecture Analysis Pipeline

A two-stage pipeline for automated software architecture analysis using DV8, NeoDepends, and local LLMs.

## What it does

1. **Stage 1 — Analyze**: Clone a GitHub repo, run DV8/NeoDepends dependency analysis across multiple commits (temporal), compute M-score, propagation cost, decoupling level, independence level, and generate time-series plots.
2. **Stage 2 — Interpret**: Feed the analysis results to a local reasoning LLM (DeepSeek-R1) to generate per-transition DRH diff reports, a combined temporal interpretation report, and answer specific architectural questions.

All outputs go into `REPOS_ANALYZED/<repo-name>/` (auto-created on first run).

## Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| Python | 3.10+ | |
| Java | 11+ | Required by DV8 and Depends |
| DV8 CLI | latest | Download from [dv8.io](https://www.archdia.com/dv8/) — add `dv8-console` to PATH |
| NeoDepends | v0.2.7 | Download binary for your platform from the [NeoDepends releases page](https://github.com/gdrosos/neodepends/releases) |
| Ollama | latest | Installed by `setup.sh` |
| Git | 2.x+ | |

## Install

```bash
git clone <this-repo-url>
cd ARCH_AGENT
chmod +x setup.sh
./setup.sh
```

This installs Python dependencies and pulls `deepseek-r1:14b` (~9 GB).

## Quick Start

All commands run from `01_stage_analyze/`:

```bash
cd 01_stage_analyze
```

### Analyze + interpret a GitHub repo (all-in-one)

```bash
python3 LLM_frontend_upgraded.py \
  "analyze and interpret https://github.com/apache/commons-io.git all-time 5 timesteps with deepseek-r1:14b and answer: how did the architecture evolve?"
```

### Interpret an existing analysis folder (fast — skips re-analysis)

```bash
python3 LLM_frontend_upgraded.py \
  "interpret this temporal analysis folder '/path/to/REPOS_ANALYZED/commons-io/temporal_analysis_alltime_...' with deepseek-r1:14b and answer: what caused the m-score drop?"
```

If a prior interpretation run exists, you will be prompted to reuse it (fast, ~30s) or re-run the LLM (slow, ~10min).

### Toy example (TrainTicket — god-class → refactored)

The toy example repo is at: [ARCH_ANALYSIS_TRAINTICKET_TOY_EXAMPLES_MULTILANG](https://github.com/FreeworkEarth/ARCH_ANALYSIS_TRAINTICKET_TOY_EXAMPLES_MULTILANG/tree/temporal)

```bash
python3 LLM_frontend_upgraded.py \
  "analyze and interpret ARCH_ANALYSIS_TRAINTICKET_TOY_EXAMPLES_MULTILANG all-time in 2 timesteps on branch temporal with deepseek-r1:14b and answer: how did the architecture change from the god-class version to the refactored version?"
```

## Pipeline stages

```
GitHub URL / local repo
        ↓
01_stage_analyze/
  dv8_agent.py           ← clones repo, runs DV8 + NeoDepends per commit
  temporal_analyzer.py   ← selects N commits, iterates stage 1
  metric_plotter.py      ← generates time-series plots (PNG)
        ↓
  REPOS_ANALYZED/<repo>/temporal_analysis_*/
    timeseries.json       ← metrics per revision
    plots/                ← PNG charts
    INPUT_INTERPRETATION/ ← DRH diff payloads
        ↓
02_stage_interpret/
  interpret_temporal_bundle.py  ← orchestrates per-transition LLM calls
  interpret_drh_diff.py         ← per-transition DRH diff report (LLM)
  interpret_metrics.py          ← metric explanation
        ↓
  INPUT_INTERPRETATION/<timestamp>_<model>/
    temporal_interpretation_report_*.md  ← combined report
    drh_diff_report_*_new{N}_old{N}.md  ← per-transition reports
    USER_ANSWER_*.md                     ← Q&A answers
```

## Models

| Model | Use | Speed |
|-------|-----|-------|
| `deepseek-r1:14b` | Interpretation + Q&A (default) | ~3-5 min/transition |
| `deepseek-r1:32b` | Higher quality interpretation | ~8-10 min/transition |
| `deepseek-r1:70b` | Best quality (requires ~140 GB RAM) | Cluster recommended |

Switch model by adding `with deepseek-r1:32b` to your prompt.
