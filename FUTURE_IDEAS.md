# ARCH_AGENT — Future Ideas & Research Directions

**Last updated**: 2026-03-29

---

## 1. Empirical Weight Learning for File Risk Scoring

**Status**: Future research — ~25 Java project study

### Motivation
The current `risk_score` in ARCH_AGENT is a heuristic formula with manually tuned weights. It is NOT empirically validated. For the study with ~25 Java projects, the professor wants to:

1. **Label each timestep**: For each 3-month snapshot, record which files had the highest *future* bug_churn in the next timestep. This is the prediction target.
2. **Features** (signals already computed in our pipeline):
   - AP membership count (how many anti-pattern instances does this file belong to?)
   - AP type membership: in a CRITICAL group (appears in both scope+pain ranked lists)?
   - fan-in and fan-out absolute values
   - fan-in/fan-out *delta* from last timestep (+102 fan-in growth = strong signal)
   - bug_churn in last window (historical predictor of future bug_churn)
   - total_churn (bug + nonbug)
   - DRH layer (files in higher layers may be more exposed)
   - M-score of the module the file belongs to
3. **Prediction target**: next-window bug_churn per file, or binary "top-10% by churn"

### Two paths for the scoring model

**Option A — Linear scoring formula (interpretable, no training needed)**
- Fit weights via regression: `score = w1*ap_count + w2*is_critical + w3*fan_in_delta + ...`
- Can report coefficients as "which signals matter most"
- Very fast, explainable to professor/reviewers

**Option B — Small neural network (learnable, per-repo fine-tunable)**
- Input: feature vector per file per timestep (as above)
- Hidden layers: 2-3 layers, ReLU, small (32–64 units)
- Output: predicted bug_churn score (regression) or risk bucket (classification)
- Key design question: **train globally across 25 repos**, or **fine-tune per repo after global pre-training**?
- The fine-tune path is interesting because repo-specific co-change thresholds (see §3 below) are learnable per-repo parameters — the network could adapt to each project's "style"
- This naturally extends to: after temporal analysis of a new repo, run a quick few-shot fine-tune step as automated post-processing

### What we need to build first
- Ground truth labels: for each file+timestep, record actual next-window bug_churn → store in temporal payload JSONs
- Feature extraction script: pull the signals above from existing DV8 JSON outputs + git log
- Possibly: extend `backfill_temporal_payloads.py` to also write a `file_features.jsonl` per revision for ML training

### Economic interpretation
- If we have developer cost statistics (e.g. avg $X per line changed), we can convert bug_churn to **estimated cost in $**
- `bug_cost = bug_churn × lines_per_commit × $/line` (rough) or more precisely per actual commit
- Makes the analysis actionable for managers: "IOUtils.java is predicted to cost ~$12k in bug fixes next quarter"

---

## 2. Replacing `risk_score` With Observable Signals (FOR NOW)

**Status**: Active — implement before empirical study

### Motivation
The professor wants to **exclude `risk_score`** from arguments until it is empirically validated. However, the same underlying signals are valid observational data. Replace "risk_score" in reports with:

### Observable replacement signals
| Old: `risk_score` | New: observable count |
|---|---|
| High risk_score → file is dangerous | **AP instance count**: "IOUtils.java appears in 7 anti-pattern instances" |
| Very high risk | **CRITICAL flag**: "appears in groups that rank in both Top-5 by scope AND Top-5 by pain" |
| Worst files ranking | **Highest bug_churn raw count**: "FileUtils.java: 416 lines changed in bug-fix commits" |
| Dependency centrality | **fan-in absolute value + delta**: "+102 fan-in in last 3 months" |

### Reporting change needed in `interpret_temporal_bundle.py`
- In the "Worst files overall" section of the Q&A prompt: instead of ranking by `risk_score`, rank by: `bug_churn DESC`, then `ap_count DESC` as tiebreaker
- In the hotspot data passed to the LLM: include `ap_count` (count of AP instances the file belongs to) alongside or instead of `risk_score`
- Label: say "most-changed files in bug-fix commits" not "highest risk files"

---

## 3. DV8 Modularity Violation Threshold — Learnable Parameter

**Status**: Research question raised by professor (2026-03-29)

### What the threshold does
DV8 computes modularity violations by finding file pairs that **co-change more than X times** without a structural dependency. The threshold X is a hardcoded parameter in the DV8 command:

```
dv8-console arch-issue --issue modularity-violation --threshold <X>
```

(The exact flag name may vary — check DV8 docs/CLI help for the current flag name.)

Default X is typically low (e.g. 2–3 co-changes). For a large project (1000 files), this catches many incidental co-changes (e.g. license header reformatting, indentation linting, variable naming conventions changing across the whole repo). These are **not semantically meaningful co-changes**.

### Types of co-changes — what we want to filter
| Co-change type | Meaningful? | Notes |
|---|---|---|
| Bug fix propagation | YES | The coupling we want to detect |
| Feature implementation | YES | Intentional co-evolution |
| License header update | NO | Affects hundreds of files at once |
| Indentation / formatting | NO | CI linting or new style guide applied globally |
| Variable naming convention | NO | "add underscore prefix" applied repo-wide |
| Dependency version bump | SOMETIMES | Large refactor = meaningful, minor bump = noise |
| Privacy/legal compliance | MAYBE | One-time batch change, not recurring coupling |
| Encryption key length / data format | YES | Hidden semantic dependency — exactly what MV captures |

### Professor's comment (paraphrased from meeting notes)
> "Look at a set of 1000 files — you try to understand what subset of files that co-change more than X represents real changes (bug fix or feature) vs process-based changes (licensing, indentation, all variables need underscore). Modularity violation shall capture semantically significant co-changes."

### Why threshold is per-repo and learnable
- A **small library** (50 files): threshold=2 may be correct — even 2 co-changes is a signal
- A **large monorepo** (5000 files): threshold=2 is too low — noise dominates, MV instances balloon to 200+ files (we see this: "106.7% of system")
- The right threshold likely correlates with: repo size, commit frequency, team size, release cadence
- **LLM-based solution** (professor suggestion): use an LLM to classify co-change commits as process-noise vs semantically meaningful, then back-compute the optimal threshold that keeps only the meaningful ones

### SNR framing — the threshold IS a signal-to-noise ratio problem

Every co-change between two files is either:
- **Signal**: the two files are genuinely coupled (bug propagates, data format shared, timing dependency)
- **Noise**: they just happened to be in the same commit for unrelated reasons (license sweep, formatter run, rename convention applied across all files)

The threshold X controls where you draw the line. Too low → noise dominates (MV balloons to 106% of system). Too high → real couplings get filtered out.

**Formal SNR per file pair (A, B)**:
```
co_changes_total(A,B)      = semantic_co_changes + process_co_changes
SNR(A,B)                   = semantic_co_changes / process_co_changes
```
A pair is a true MV only if `semantic_co_changes >= threshold` — meaning it co-changed enough times in *real* work, not just batch commits.

**What makes a commit "process noise"**:
- It touches many files at once (license header → touches every .java file)
- The commit message contains keywords: "license", "formatting", "rename", "style", "checkstyle", "copyright", "version bump"
- The changed lines are uniform across all files (same header added to all = 100% identical diff structure)

**What makes a commit "semantic signal"**:
- It touches a small focused set of files (2–10 files)
- Commit message: "fix", "bug", "NPE", "regression", "feature", "refactor [specific class]"
- The co-changing files have no declared structural dependency (import/call) — that's the hidden coupling DV8 is hunting

### Concrete approaches to threshold tuning

1. **Heuristic scaling**: `threshold = max(2, round(log10(num_files) × k))` — scale with project size. For commons-io (~268 files): `log10(268) ≈ 2.4 × k`. With k=2 → threshold=5. This matches our empirical observation that threshold=5 reduces noise on medium repos.

2. **Commit-level noise classification (no % filter needed)**:
   - For each commit that caused a co-change between files A and B, check the commit message
   - Classify as `process` if message matches noise keywords OR if the commit touches >20 files AND the changed line count per file is small (<5 lines avg)
   - Only count co-changes from non-process commits
   - This avoids the blunt "% of files" heuristic — instead classifies per-commit

3. **LLM commit classifier** (professor suggestion):
   - Sample 50–100 commits that caused co-changes
   - Ask LLM: "Is this commit a semantically meaningful change (bug fix, feature, hidden coupling) or a process/tooling change (formatting, license, rename convention)?"
   - Compute `p_semantic` = fraction classified as semantic
   - Optimal threshold = the X where `semantic_co_changes(A,B) >= X` keeps ~90% of truly coupled pairs
   - This makes the threshold data-driven and per-repo

4. **Mutual information approach** (information-theoretic):
   - Compute MI(A, B) = mutual information between the commit-change vectors of files A and B
   - High MI = genuinely coupled (they change together more than chance)
   - Low MI = coincidental co-change (big batch commits inflate the count)
   - Threshold becomes: only flag pairs where MI(A,B) > some percentile cutoff
   - This is the most principled approach — it directly measures how much knowing "A changed" tells you about "B changed"

5. **Empirical from 25-repo study**: find threshold T* that maximizes correlation between MV membership and future bug_churn. This IS the learnable parameter — the ground truth answer.

### Which approach to implement first
- **Short term**: approach 2 (commit message classifier, no LLM needed) — fast, explainable, avoids blunt % filter
- **Medium term**: approach 3 (LLM classifier) — more accurate, fits naturally into our pipeline
- **Long term**: approach 5 (empirical from study) — the academically sound answer

### Integration point in our pipeline
- `backfill_temporal_payloads.py` calls DV8 analysis — threshold passed as CLI arg (already implemented as `--mv-cochange`)
- Threshold calibration script could run once per repo before temporal analysis, store result in temporal metadata JSON
- The calibrated threshold would then be passed automatically to every `dv8_agent.py` run for that repo

---

## 3b. Semantic Co-change Pre-filter — Optional Toggle

**Status**: PLANNED — implement as `--mv-filter-semantic` toggle in `dv8_agent.py` + `LLM_frontend_upgraded.py`

### Implementation order
1. **NOW**: use `--mv-cochange 5` as manual workaround for large repos ✓ (already done)
2. **SOON**: implement `--mv-filter-semantic` toggle using existing `issue_map.json` — pre-filter git history before DV8 sees it, no new infrastructure needed
3. **25-repo study**: validate whether the filter actually improves MV quality vs just raising the threshold — this is a separate empirical experiment

### Does it work on git-only repos (no JIRA)?
Yes — `issue_map.json` uses a two-layer approach:
- **Layer 1 (JIRA)**: if the repo has JIRA issue IDs in commit messages (e.g. `IO-123` for commons-io), look up the issue type (Bug/Improvement/Task) from the JIRA API
- **Layer 2 (keyword fallback)**: for any commit without a JIRA ID, classify by commit message keywords: `fix/bug/npe/regression/error/exception` → semantic; `license/copyright/format/indent/style/checkstyle/rename/version bump` → process noise
- This means it works on **any git repo** — JIRA just gives higher confidence classification when available

### The idea
Before DV8 counts co-changes for MV detection, filter the git history to only include **semantically meaningful commits**. DV8 then sees a clean signal — process noise commits (license sweeps, formatting, mass renames) never contribute to co-change counts. The MV threshold can stay low (2–3) because the noise is gone before counting starts.

### Why it's its own empirical study
To know whether this filter actually improves MV quality, you need ground truth: do the MVs detected on filtered history better predict future bug_churn than MVs on unfiltered history? That requires the 25-repo study (§6) to compare:
- Baseline: unfiltered history + threshold=5
- Treatment: filtered history + threshold=2
- Outcome: which produces MVs that better correlate with future bugs?

This is non-trivial — the filter itself might introduce bias (e.g. refactoring commits are semantic but produce large co-change batches that look like noise).

### How to implement it (when ready)

**Step 1 — Reuse existing issue_map.json classification**

We already classify commits as `bug / feature / refactoring / test` via JIRA + keyword matching (see `bug_churn_commits.json`). This is the semantic/process signal we need — no new LLM call required for a first pass.

**Step 2 — Pre-filter git-history.txt**

`dv8_agent.py` generates `git-history.txt` (full repo history in DV8 gittxt format) before calling `scm:history:gittxt:convert-matrix`. Insert a filtering step here:

```python
# NEW optional step — filter git-history.txt to semantic commits only
if mv_filter_semantic:
    filter_git_history_to_semantic(
        history_file=git_history_txt,
        issue_map=issue_map_json,       # already computed
        output_file=git_history_filtered_txt,
    )
    history_input = git_history_filtered_txt
else:
    history_input = git_history_txt
```

`filter_git_history_to_semantic()` reads the gittxt format, looks up each commit hash in `issue_map.json`, and writes only commits classified as `bug` or `feature`. Commits not in the issue map default to: keep if they touch ≤ 15 files (heuristic for focused work), drop if they touch > 15 files with short diffs (likely process batch).

**Step 3 — Toggle via CLI flag**

Add `--mv-filter-semantic` flag to `dv8_agent.py` and `LLM_frontend_upgraded.py`. Default: off (current behavior preserved). When on: uses filtered history for MV computation only (structural DSM, metrics, DRH are unaffected).

**Step 4 — Validate**

Compare MV instances between filtered and unfiltered runs on commons-io:
- Do the filtered MVs have higher average bug_churn per member file?
- Does the filtered run produce fewer but more actionable instances?
- Does the 106% → reasonable % shrinkage happen at threshold=2 instead of needing threshold=5?

### LLM-based classifier (future upgrade)
The keyword/JIRA approach misclassifies edge cases (e.g. a commit titled "cleanup" that's actually a bug fix). An LLM classifier would:
- Sample the diff + commit message for ambiguous commits
- Ask: "Is this a semantically meaningful change between these specific files, or a process/tooling change?"
- This is more accurate but requires LLM calls per commit — expensive at scale
- Practical path: run keyword classifier first, then LLM only on commits the keyword classifier is uncertain about (those touching 5–20 files with mixed message signals)

### Note on IB clustering
Information Bottleneck clustering of file pairs by co-change pattern is theoretically the most principled approach (it finds which file pairs share genuine mutual information vs coincidental co-occurrence). However it requires building the full co-change probability matrix first and is computationally expensive. Best suited for the 25-repo study as a comparison baseline, not for per-repo runtime use.

---

## 4. M-score Layer Breakdown in Reports

**Status**: Future — computation exists, not yet surfaced

- `mscore_dv8_exact.py` computes per-layer/module M-score components
- Not currently included in the LLM interpretation payload
- When surfaced: report should identify *which specific layers/modules* are worst and give concrete refactoring targets

---

## 5. Cluster / Cloud Deployment

**Status**: Future — wait for professor feedback before starting

- Replace Ollama with **vLLM** (Linux + NVIDIA GPU — MLX is Mac-only)
- DV8 license: must solve headless activation for cluster use
- Multi-user: job queue (Celery/Redis) so multiple queries don't conflict
- Model upgrade path: DeepSeek-R1:32b/70b viable on A100/H100
- RAG/GraphRAG knowledge base makes more sense at multi-user scale

---

## 7. NeoDepends Stage 3 — Interprocedural Return-Type Tracking (close 3 remaining FNs)

**Status**: Future — not yet implemented. Stage 2 (return-type annotation extraction) was implemented 2026-05-12.

### The 3 remaining false-negative file-pairs (Toy Python FIRST)

After Stage 2, NeoDepends achieves **91.7% file-pair recall** on the Python toy. The 3 remaining FNs all share the same root cause: **2-hop interprocedural reasoning** across field accesses and method return values, which is hard without either type annotations or a runtime trace.

| FN pair | Access pattern | Why it fails |
|---------|---------------|-------------|
| `ticket.py → train_station.py` | `self.route.get_origin().get_name()` | `Route.get_origin()` returns a `TrainStation`, but `Route.origin` field's type is unknown — param is named `origin`, which doesn't map to `TrainStation` by name convention |
| `ticket_booking_system.py → person.py` | `passenger.name` where `.name` is on parent `Person` | Requires: (1) infer `passenger` var type = `Passenger`, (2) look up `name` field on `Passenger`, (3) resolve inherited field upward to `Person` class in a different file |
| `train_station.py → route.py` | `train.route.destination == destination` | `train` is a local var of type `Train`, `.route` is a field of type `Route`, `.destination` is a field of `Route`. Requires tracking `var.field.field` chains — 2 hops through the type graph |

### What Stage 2 already does (implemented)

In `tools/enhance_python_deps.py`, Stage 2 handles:
1. Methods with explicit `-> ClassName` return annotations → `method_return_types[mid] = {ClassName}`
2. Methods that `return self.field` where `field` has an inferred type → propagate to `method_return_types`
3. In C2 (`self.field.method()` calls): after resolving the called method, emit a `Method → ReturnClass Use` edge if the callee has a known return type

This works for annotated real-world code but **not** for the toy because:
- `Route.get_origin()` has no annotation and `self.origin` field type is unknown (param name `origin` ≠ class `TrainStation`)
- `train.route.destination` is a `var.field.field` chain, not a method call

### What Stage 3 needs to implement

**Option A — `var.field.field` chain tracking** (closes `train_station.py → route.py`):
- In `_MethodBodyFacts`, detect `var.attr1.attr2` attribute chains where `var` has a known type `T`
- Look up `attr1` field type on `T` → yields type `U`
- If `U` is a known internal class, emit a `Method → U Use` edge
- Implementation: extend `visit_Attribute` in `_MethodBodyFacts` to walk chained attribute accesses

**Option B — Inherited field resolution for var types** (closes `ticket_booking_system.py → person.py`):
- Currently `var.field` is not tracked when `var` type is inferred in `env`
- Extend section D (var_calls) to also detect `var.attr` attribute reads (not just `var.method()` calls)
- When `env[var] = "Passenger"`, `passenger.name` → look up `name` in `Passenger` fields, traverse inheritance to `Person` → emit Use edge to `Person`
- Implementation: add `var_attr_reads: List[Tuple[str, str]]` to `_MethodBodyFacts` via new `visit_Attribute` detection

**Option C — Untyped param name → class matching** (closes `ticket.py → train_station.py` fully):
- `Route.__init__(self, route_id, origin, destination, ...)`: params `origin`/`destination` don't map to `TrainStation` by camelCase
- Would need a "usage-based type inference": if `self.origin.get_name()` is called later and `get_name()` is only defined on `TrainStation`, infer `self.origin: TrainStation`
- Or: if caller always passes a `TrainStation` instance to `origin` (interprocedural flow)
- This is the hardest case and requires true dataflow analysis

### Recommended implementation order

1. **Stage 3A** — `var.field` attribute read tracking in section D (~50 lines) — closes the `person.py` FN
2. **Stage 3B** — `var.field.field` chain tracking in `_MethodBodyFacts` (~80 lines) — closes the `route.py` FN
3. **Stage 3C** — usage-based method dispatch → type inference (most complex) — closes `train_station.py` FN

### Expected impact on benchmark

After Stage 3A+B: **~97%+ file-pair recall** on toy Python FIRST (36 → 35 GT pairs detected, or possibly 36/36).
The toy's 3 FNs are intentionally hard coupling patterns — real-world codebases with type annotations will benefit from Stage 2 already.

---

## 6. 25-Project Empirical Study — Data Plan

**Status**: Planning

- **Repos**: ~25 Java projects (open source, varied sizes)
- **Timestep**: 3-month windows (already supported by pipeline)
- **Labels**: actual bug_churn per file in next window (ground truth)
- **Prediction features**: AP membership, fan-in/fan-out delta, DRH layer, M-score module
- **Deliverable**: validated weight coefficients for risk formula, or trained small network
- **Optional**: link churn to $-cost using developer cost statistics for economic impact framing
