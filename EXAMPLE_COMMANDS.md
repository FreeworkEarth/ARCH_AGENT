# ARCH_AGENT — Example Commands

All analysis is launched through one entry point:

```bash
cd ARCH_AGENT/01_stage_analyze
python3 LLM_frontend_upgraded.py "<your prompt here>"
```

The LLM parses your prompt, selects the right tool, and runs the full pipeline automatically.

---

## The single prompt that does everything

**Question: "What are the 5 most dangerous files in commons-io over the last 3 years?"**

```bash
python3 LLM_frontend_upgraded.py \
  "analyze and interpret https://github.com/apache/commons-io.git \
   last 3 years 36 commits 1 per month with deepseek-r1:32b \
   and answer: What are the 5 most dangerous files in the repository right now, \
   and why? Base the answer on anti-pattern involvement, structural coupling (fan-in), \
   SCC membership, bug-linked churn, and co-change coupling signals."
```

**What this single prompt triggers automatically:**

| Step | Script | What it does |
|------|--------|--------------|
| 1 | `temporal_analyzer.py` | Clones repo (if needed), picks 36 commits 1/month over 3 years, runs DV8 on each |
| 2 | `backfill_temporal_payloads.py` | Builds interpretation payloads + evidence graph diffs |
| 3 | `fetch_github_issues.py` | Auto-detects JIRA (IO project) or GitHub, fetches issue→type map for bug churn |
| 4 | `export_dv8_binary_files.py` | Converts all `.dv8-clsx`/`.dv8-dsm` binary files → JSON + CSV |
| 5 | `compute_file_risk_scores.py` | Multi-signal composite risk score for every file (anti-patterns + fan-in + SCC + bug churn + co-change) |
| 6 | `plot_risk_score_stats.py` | 6 statistical plots + `risk_score_stats.json` |
| 7 | `interpret_temporal_bundle.py` | LLM interprets all pairwise DRH transitions + writes overall summary |
| 8 | Q&A | LLM answers your question using the interpretation report |

**Output locations (all auto-created):**

```
REPOS_ANALYZED/commons-io/temporal_analysis_recent_<timestamp>/
├── timeseries.json                          ← M-score, PC, DL, IL over time
├── issue_map.json                           ← JIRA IO issue→type map
├── INPUT_INTERPRETATION/
│   ├── timeseries.json
│   ├── file_risk_scores.json                ← ranked per-file risk scores ★
│   ├── file_risk_scores.csv                 ← spreadsheet version ★
│   ├── plots/risk_stats/
│   │   ├── risk_score_distribution.png      ★
│   │   ├── signal_distributions.png         ★
│   │   ├── signal_correlation_heatmap.png   ★
│   │   ├── top_files_bar.png                ★
│   │   ├── anti_pattern_risk_boxplot.png    ★
│   │   ├── signals_scatter_matrix.png       ★
│   │   └── risk_score_stats.json            ★
│   ├── EVIDENCE_GRAPH_DIFF/
│   │   └── evidence_graph_diff_new*_old*.json
│   └── SINGLE_REVISION_ANALYSIS_DATA/
│       └── <rev>/OutputData/
│           ├── interpretation_payload.json
│           └── **/anti-pattern-instances/<type>/<id>/
│               ├── *-clsx_files.json        ← readable anti-pattern members ★
│               ├── *-clsx_files.csv         ★
│               ├── *-sdsm_deps.json         ← readable DSM dependencies ★
│               └── *-sdsm_deps.csv          ★
└── OUTPUT_INTERPRETATION/
    └── <run>/
        ├── temporal_interpretation_report_deepseek-r1_32b_*.md
        └── USER_ANSWER_*.md                 ← answer to your question ★
```

---

## More prompt examples

### Use an already-cloned local repo
```bash
python3 LLM_frontend_upgraded.py \
  "analyze and interpret commons-io last 3 years 36 commits 1 per month \
   with deepseek-r1:32b and answer: what are the 5 most dangerous files?"
```

### Different repo — pdfbox (note: branch is trunk)
```bash
python3 LLM_frontend_upgraded.py \
  "analyze and interpret https://github.com/apache/pdfbox.git \
   last 3 years 36 commits 1 per month on branch trunk \
   with deepseek-r1:32b and answer: what are the 5 most dangerous files?"
```

### Just analyze + risk scores, no LLM interpretation
```bash
python3 LLM_frontend_upgraded.py \
  "only analyze commons-io last 3 years 36 commits 1 per month"
```
*(omit "interpret"/"answer" → skips LLM interpretation, still runs risk pipeline)*

### All-time analysis (first commit to now, evenly spaced)
```bash
python3 LLM_frontend_upgraded.py \
  "analyze and interpret commons-io all time 10 timesteps \
   with deepseek-r1:32b and answer: how has the architecture quality changed over the full history?"
```

### Fast query from existing results (no re-run)
```bash
python3 LLM_frontend_upgraded.py \
  "query commons-io: what are the 5 most dangerous files?"
```

### Interactive Q&A session on existing results
```bash
python3 LLM_frontend_upgraded.py "query commons-io"
```

---

## Environment variables

| Variable | Purpose | Required? |
|----------|---------|-----------|
| `GH_TOKEN` or `GITHUB_TOKEN` | GitHub personal access token — improves bug churn accuracy (typed issue→commit linking) | Optional but recommended |
| `OLLAMA_MODEL` | Override default LLM model (default: `llama3.1` for dispatch, `deepseek-r1:32b` for interpretation) | Optional |
| `OLLAMA_ENDPOINT` | Ollama server URL (default: `http://127.0.0.1:11434`) | Optional |

```bash
# Recommended: set GitHub token before running
export GH_TOKEN="ghp_your_token_here"
python3 LLM_frontend_upgraded.py "analyze and interpret commons-io ..."
```

---

## Manual step-by-step (if you need to re-run individual steps)

```bash
BASE="REPOS_ANALYZED/commons-io/temporal_analysis_recent_<TIMESTAMP>"
INTERP="$BASE/INPUT_INTERPRETATION"
REPO="TEST_AUTO/REPOS/commons-io"

# Step 1: DV8 temporal analysis (already done if folder exists)
python3 temporal_analyzer.py \
  --repo "$REPO" --mode recent --months 1 --count 36 --branch master

# Step 2: Backfill payloads
python3 backfill_temporal_payloads.py "$BASE" --meta-repo commons-io \
  --issue-map "$BASE/issue_map.json"

# Step 3: Fetch issues (JIRA auto-detected)
python3 fetch_github_issues.py \
  --git-root "$REPO" --out "$BASE/issue_map.json" --verbose

# Step 4: Export DV8 binary files → JSON/CSV
python3 export_dv8_binary_files.py --all "$INTERP"

# Step 5: Compute file risk scores
python3 compute_file_risk_scores.py "$INTERP" --git-root "$REPO" --verbose

# Step 6: Generate statistical plots
python3 plot_risk_score_stats.py "$INTERP/file_risk_scores.json" --top-n 30

# Step 7: LLM interpretation + answer question
python3 ../02_stage_interpret/interpret_temporal_bundle.py \
  --temporal-root "$BASE" \
  --model deepseek-r1:32b \
  --user-question "What are the 5 most dangerous files in commons-io right now, and why?"
```

---

## Risk score formula (for reference)

```
risk_score(file) =
    0.30 × norm(bug_churn_total)        ← lines changed in bug-fix commits
  + 0.25 × norm(anti_pattern_count)    ← revisions where file is in a DV8 anti-pattern
  + 0.20 × norm(hotspot_fanin_score)   ← sum of fan-in across revisions (blast radius)
  + 0.15 × norm(scc_membership_count)  ← revisions where file is in a cyclic SCC
  + 0.10 × norm(co_change_without_dep) ← co-change partners with no declared dependency
```

All signals are min-max normalised to [0, 1] across all files.
Weights are configurable: `--weights '{"bug_churn":0.40,"anti_pattern":0.20,...}'`
