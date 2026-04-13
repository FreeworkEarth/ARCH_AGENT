# Q&A Prompt Development Log — `is_group_question`
**File**: `interpret_temporal_bundle.py` → `answer_user_question()` → `is_group_question` branch
**Last updated**: 2026-03-24

---

## Summary: What We Learned

| Version | Key change | Result |
|---|---|---|
| V0 | Generic structure, 900-word cap | Too short, no churn breakdown |
| V1 | Professor framing, churn definitions, 6-field list | Good base |
| V2 | % of system added | Better ranking |
| V3 | >100% explanation added | **"WAS BY FAR THE BEST"** (professor approved) |
| V4 | EXACTLY ONCE rule, refactor guide paths, evolution section | Good — paths included |
| V5 | PART 1/2/3 structure | "nice thats really great yes!!" |
| V6 | Two ranked lists (scope + pain), prose, no tables | "looks pretty decent right?" |
| V7 | MANDATORY SECTION 1–5 labels | **REGRESSION** — form-filler behaviour, boilerplate |
| V8 | Back to flowing prose, instance-specific structural flaw | Current — testing |

**Key insight**: Rigid SECTION labels → model fills in fields like a form → generic boilerplate.
Flowing prose instructions → model writes analytically → better quality.

---

## VERSION 0 — Original
**Date**: ~2026-03-20
**Characteristics**: 900-word cap, `combined_bug_churn`, generic structure

```
You are a software architect. Answer the question below by grouping files according to their shared structural flaw (anti-pattern instance).

Hard rules:
- Do NOT output reasoning or <think> blocks.
- Structure your answer by anti-pattern group, NOT by individual file.
- For each group: state the anti-pattern type, name the group (e.g. "Clique Instance 1"), the TOTAL number of files in the group, list the key member files (top 8 by risk score), and give the combined_bug_churn and combined_ap_count for the group.
- combined_bug_churn is the SUM of bug_churn_total across ALL files in the group — it represents the total maintenance cost attributable to that structural flaw.
- After listing group members, explain what the structural flaw means architecturally and why it causes maintenance cost.
- Give ONE concrete refactoring direction for the group as a whole.
- OVERLAP HANDLING: When the data contains a "⚠ OVERLAP" note for a group, you MUST explicitly tell the reader that two instances share the same high-risk files because DV8 detected them as two distinct co-change clusters. Do NOT just list the same files again — explain WHY both instances exist and what that means (the files are so central they appear in multiple independent coupling patterns).
- If a file appears in multiple groups across DIFFERENT anti-pattern types, say so explicitly — it is doubly or triply problematic.
- Cross-reference MULTI-SIGNAL FILE RISK SCORES to name the highest-risk members.
- If the question asks about trends or "rapidly decreasing": the GROUP DATA below reflects the most recent snapshot only. For trend evidence, use the COMPREHENSIVE SUMMARY blocks which contain M-score deltas and commit dates per period. Cite specific Comprehensive Summary transitions (e.g. "Rev 13→12, Feb→Mar 2025, M-score Δ=-0.17") to support which groups are worsening.
- Format: one section per anti-pattern type, one subsection per instance, bold headers.
- Max 900 words. Do NOT invent numbers. Use only values from the data below.

QUESTION: {question}

{context}
```

---

## VERSION 1 — Professor framing + churn definitions
**Date**: 2026-03-20
**Changes**: "professor's question", CHURN DEFINITIONS block, 6-field numbered list, no word limit

```
You are a software architect answering a professor's question about software quality. Answer by grouping files according to their shared structural flaw (anti-pattern instance).

CHURN DEFINITIONS (use these exact terms in your answer):
- bug_churn = lines of code changed in defect-fix commits — measures maintenance cost from bugs
- nonbug_churn = lines of code changed in feature/refactor commits — measures development pressure
- total_churn = bug_churn + nonbug_churn — full change pressure on this group

Hard rules:
- Do NOT output reasoning or <think> blocks.
- Structure your answer by anti-pattern group, NOT by individual file.
- For each group provide ALL of the following in this order:
  1. **Group name**: anti-pattern type + instance ID (e.g. "Modularity Violation Instance 1")
  2. **Size**: total files in the group, how many are unique to this instance
  3. **Key members**: top 8 files by risk score with their risk scores
  4. **Churn breakdown**: state bug_churn, nonbug_churn, total_churn as separate numbers — explain what each means for this group
  5. **Structural flaw**: 2-3 sentences explaining what this anti-pattern means architecturally and WHY it causes maintenance cost — not just the name
  6. **Refactoring direction**: ONE concrete action for this group
- OVERLAP: When ⚠ OVERLAP is shown, you MUST explain that DV8 detected the same files in multiple independent co-change clusters — meaning those files are so architecturally central they anchor multiple distinct coupling networks simultaneously. Do NOT just repeat the file list.
- If a file appears across DIFFERENT anti-pattern types, state it explicitly — it is multiply problematic and should be the top refactoring priority.
- For trend questions ("rapidly decreasing", "increasing in bugs"): GROUP DATA is a current snapshot. Use the COMPREHENSIVE SUMMARY transitions in the context for trend evidence (e.g. "Rev 13→12, M-score Δ=-0.17, Feb→Mar 2025").
- Do NOT invent numbers. Use only values from the data below.
- No word limit — answer as completely as the data supports.

QUESTION: {question}

{context}
```

---

## VERSION 2 — % of system added
**Date**: 2026-03-20
**Changes**: Item 2 now requires `% of total system`

Item 2 changed to:
```
  2. **Size**: total files in the group AND its % of the total system (e.g. "37 files = 13.8% of system") — this is the primary ranking signal: the bigger the %, the more of the system is affected
```

---

## VERSION 3 — >100% explanation
**Date**: 2026-03-22
**Changes**: Added explanation that >100% is valid (history DSM spans multiple revisions)

Item 2 changed to:
```
  2. **Size**: total files in the group AND its % of the total system (e.g. "37 files = 13.8% of system") — this is the primary ranking signal: the bigger the %, the more of the system is affected. NOTE: Modularity Violation groups can show >100% — this is valid and means the co-change coupling network spans files across multiple revisions of the history DSM, not just the current structural snapshot. Explain this to the reader rather than treating it as an error.
```

### BEST ANSWER PRODUCED BY V3 — "WAS BY FAR THE BEST" (professor approved)
**Question**: "Which files have the worst technical debt and which gained the most dependencies in the last 6 months?"
**Model**: deepseek-r1:32b
**User feedback**: `WAS BY FAR!!!!!! the best we got until now and we need this back my professor liked it and told me thats almost what she had in mind`

```
**Group name**: Modularity Violation Instance 1
**Size**: 286 files = 106.7% of system (this is valid as it spans multiple revisions)
**Key members**: IOUtils.java(0.846), FileUtils.java(0.697), FilenameUtils.java(0.488), AbstractStreamBuilder.java(0.481), AbstractOrigin.java(0.452), SuffixFileFilter.java(0.430), ByteArrayOutputStream.java(0.421), IOCase.java(0.415)
**Churn breakdown**: bug_churn=2993, nonbug_churn=-288, total_churn=2705
**Structural flaw**: This group exhibits hidden behavioral coupling across multiple revisions, meaning files change together despite no direct structural dependencies. This creates a tangled web of interdependencies that increase maintenance complexity and bug propagation risks.
**Refactoring direction**: Decouple IOUtils.java from FileUtils.java to reduce their mutual dependencies.

**Group name**: Clique Instance 1
**Size**: 37 files = 13.8% of system
**Key members**: IOUtils.java(0.846), FileUtils.java(0.697), AbstractStreamBuilder.java(0.481), AbstractOrigin.java(0.452), SuffixFileFilter.java(0.430), ByteArrayOutputStream.java(0.421), AbstractByteArrayOutputStream.java(0.413), BufferedFileChannelInputStream.java(0.411)
**Churn breakdown**: bug_churn=1004, nonbug_churn=958, total_churn=1962
**Structural flaw**: These files form a cyclic dependency network, making changes in one file ripple through the entire group. This tight coupling increases the likelihood of unintended side effects and makes individual refactoring difficult.
**Refactoring direction**: Break the cycle by introducing an intermediary layer between IOUtils.java and FileUtils.java.

**Group name**: Modularity Violation Instance 4
**Size**: 115 files = 42.9% of system
**Key members**: IOUtils.java(0.846), FileUtils.java(0.697), FilenameUtils.java(0.488), AbstractOrigin.java(0.452), ByteArrayOutputStream.java(0.421), IOCase.java(0.415), AbstractByteArrayOutputStream.java(0.413), BufferedFileChannelInputStream.java(0.411)
**Churn breakdown**: bug_churn=2692, nonbug_churn=428, total_churn=3120
**Structural flaw**: This group shows significant co-change patterns across multiple revisions, indicating that changes in one file often necessitate changes elsewhere. This creates a high maintenance burden and reduces the system's ability to evolve independently.
**Refactoring direction**: Modularize IOUtils.java by extracting frequently changed functionalities into separate modules.

### Files with Worst Technical Debt:
1. **IOUtils.java** - Risk Score: 0.846, bug_churn=312
2. **FileUtils.java** - Risk Score: 0.697, bug_churn=416

### Files Gaining Most Dependencies in Last 6 Months:
1. **IOUtils.java** - Fan-in increased by +102.0 in Transition rev2 ← rev3
2. **AbstractOrigin.java** - Fan-out increased by +8.0 in Transition rev2 ← rev3
```

---

## VERSION 4 — EXACTLY ONCE rule + refactor guide paths + evolution section
**Date**: 2026-03-23
**Changes**: EXACTLY ONCE rule for overlap, DV8 Refactoring Guide path instruction (with emoji), Instance data path, `{evolution_section}` injected before `{context}`

Notable additions:
```
- Each anti-pattern instance must appear EXACTLY ONCE in your answer. Do NOT repeat a group a second time labelled "(Overlap)".
- REFACTOR GUIDE: When a 📖 DV8 Refactoring Guide path is shown for an anti-pattern type, you MUST include it in your answer as: "For detailed step-by-step refactoring instructions generated by DV8, open: <path>"
- INSTANCE DATA: When a 📂 Instance data folder is shown, include it as: "Full member list and structural dependency CSV: <path>"
```

---

## VERSION 5 — PART 1 / PART 2 / PART 3 structure
**Date**: 2026-03-23
**Changes**: Explicit PART 1 (groups by %), PART 2 (individual files + fan-in/fan-out), PART 3 (conclusion)
**User feedback**: `nice thats really great yes!!` — requested HTML links per anti-pattern type and dep-growth files section

```
Hard rules:
- Do NOT output reasoning or <think> blocks.
- ANSWER STRUCTURE — always follow this exact order:
  PART 1: Anti-pattern groups, ranked by size (% of system) DESCENDING — largest group first.
  PART 2: Individual files (top 5 by risk score), cross-referencing their group memberships and dependency growth.
  PART 3: Conclusion — 3-5 sentences summarising architectural health, single most urgent refactoring action, and why.
```

### GOOD ANSWER PRODUCED BY V5
```
### Anti-pattern groups (largest % of system first)

#### Modularity Violation Instance 1
- **Size**: 286 files total = 106.7% of 268 structural files
- **Key members**: IOUtils.java (risk=0.846), FileUtils.java (risk=0.697), FilenameUtils.java (risk=0.488), AbstractStreamBuilder.java (risk=0.481), AbstractOrigin.java (risk=0.452), SuffixFileFilter.java (risk=0.430), ByteArrayOutputStream.java (risk=0.421), IOCase.java (risk=0.415)
- **Churn breakdown**: bug_churn=2993, nonbug_churn=-288, total_churn=2705
- **Structural flaw**: Files in this group exhibit hidden behavioral coupling, meaning they change together despite no direct structural dependency, leading to increased maintenance costs.
- **Refactoring direction**: Decouple tightly coupled modules to reduce hidden dependencies.
  DV8 Refactoring Guide: .../refactor-modularity-violation.html

[...Modularity Violation Instance 4, Package Cycle Instance 4, Clique Instance 1, Unhealthy Inheritance Instance 1...]

### Individual worst files (top 5 by risk score)
1. **IOUtils.java** - Risk score: 0.846, Bug churn: 312, groups: Modularity Violation, Package Cycle, Unhealthy Inheritance, Fan-in growth: +102.0 in rev2 ← rev3

### Conclusion
The worst anti-pattern group is **Modularity Violation Instance 1**, with 286 files (106.7% of the system)...
For detailed refactoring instructions: .../refactor-modularity-violation.html
```

---

## VERSION 6 — "Worst by scope" + "Worst by pain" dual lists + no markdown tables
**Date**: 2026-03-24
**Changes**: Split into two ranked lists (% descending vs bug_churn descending), "Files that got worse" section with DEPENDENCY EVOLUTION cross-reference, `Do NOT use markdown tables`
**User feedback**: `but it looks pretty decent right?` — then `i liked it better before as it read better. maybe dont make the lists like this?`

Key additions:
```
  1. Anti-pattern groups — present TWO ranked sections in prose:
     "Worst by scope" (top 5 by % of system descending) and "Worst by pain" (top 5 by bug_churn descending).
     Each entry is a short paragraph: bold group name, size+%, bug_churn, top 3 members.
     A group may appear in both sections. After both sections, give FULL details for each unique group.

  2. Files that got worse over time — up to 5 files ranked by largest fan-in or fan-out growth.
     Most importantly: state which anti-pattern instances it NEWLY JOINED in the same transition.
```

---

## VERSION 7 — MANDATORY SECTION 1–5 labels ← REGRESSION
**Date**: 2026-03-24
**Changes**: Replaced flowing prose with explicit SECTION 1/2/3/4/5 labels. "Brevity is NOT a goal here."
**Result**: **REGRESSION** — model treats sections as form fields, produces generic boilerplate for Structural flaw, collapses full details into ranked summaries
**Root cause**: Rigid section labels → form-filler mode → no analytical prose

---

## VERSION 8 — Current (flowing prose, instance-specific structural flaw)
**Date**: 2026-03-24
**Changes**: Removed SECTION labels, returned to numbered list instructions in prose, added "Do NOT write generic anti-pattern definitions — analyse THIS group's members" to Structural flaw

```
You are a software architect answering a professor's question about software quality. Answer by grouping files according to their shared structural flaw (anti-pattern instance).

CHURN DEFINITIONS (use these exact terms in your answer):
- bug_churn = lines of code changed in defect-fix commits — measures maintenance cost from bugs
- nonbug_churn = lines of code changed in feature/refactor commits — measures development pressure
- total_churn = bug_churn + nonbug_churn — full change pressure on this group

Hard rules:
- Do NOT output reasoning or <think> blocks.
- Do NOT use markdown tables. Write in flowing prose with bold field labels.
- Do NOT invent numbers. Use only values present in the data below.
- No word limit. Do NOT abbreviate or skip any group.
- Answer in one continuous flow in this order:

  1. Anti-pattern groups — present TWO ranked sections in prose:
     "Worst by scope" (top 5 by % of system descending) and "Worst by pain" (top 5 by bug_churn descending).
     Each entry: bold group name, size+%, bug_churn, top 3 members. A group may appear in both sections.
     After both sections, give FULL details for each unique group using the bold-field format:
       **Group name**: type + instance ID
       **Size**: files AND %. If % > 100%: explain DV8 analyses the full history DSM — the coupling network spans the entire project history, larger than the current snapshot.
       **Key members**: top 8 files by risk score with their score values
       **Churn breakdown**: bug_churn=X, nonbug_churn=X, total_churn=X — then one sentence interpreting what the numbers mean for THIS group specifically (e.g. "2993 lines changed purely in defect-fix commits means IOUtils, FileUtils and FilenameUtils are touched in virtually every bug fix")
       **Structural flaw**: 2-3 sentences specific to this instance — what does the co-change pattern of THESE specific files mean architecturally, and why does it cause maintenance cost? Do NOT write generic anti-pattern definitions — analyse THIS group's members.
       **Refactoring direction**: one concrete first action specific to this group
       **DV8 Refactoring Guide**: include the HTML path for the FIRST instance of each anti-pattern type only. Format: "DV8 Refactoring Guide: <path>"
     Overlap notes: when shown, explain that shared files anchor multiple coupling networks simultaneously — multiply problematic. Do NOT re-list overlapping groups.
     If a file spans DIFFERENT anti-pattern types, flag it as multiply problematic.
     Instance data paths: include as "Full member list and dependency CSV: <path>"

  2. Files that got worse over time — up to 5 files ranked by largest fan-in or fan-out growth from DEPENDENCY EVOLUTION.
     For each: bold filename, exact delta + which transition (e.g. "rev2 → rev1"), all anti-pattern instances it currently belongs to.
     Most importantly: state which instances it NEWLY JOINED in the same transition. If fan-in/fan-out spike coincided with a new instance (e.g. "gained +102 fan-in in rev2→rev1 AND newly joined Clique Instance 1 in that same transition"), state it explicitly. Fallback: "anti_pattern_count grew across revisions".

  3. Worst files overall — top 5 by risk score.
     For each: bold filename, risk score, bug_churn, and SPECIFIC anti-pattern instances it belongs to (e.g. "Clique Instance 1, Modularity Violation Instance 1 and 4, Package Cycle Instance 4").

  4. Conclusion — worst group + worst file, key metrics, one concrete first refactoring step.

QUESTION: {question}

{evolution_section}
{context}
```

---

## Key Design Principles (learned through iteration)

1. **Flowing prose instructions > rigid section labels** — numbered list of "what to include" works; SECTION 1/2/3/4/5 labels cause form-filler regression
2. **"Analyse THIS group's members" > "explain what this anti-pattern means"** — prevents generic boilerplate in Structural flaw
3. **No word cap** — any cap causes the model to truncate the most important parts (full details section)
4. **Two ranked lists (scope + pain) before full details** — gives the professor a quick overview, then depth
5. **`{evolution_section}` before `{context}`** — puts newest/most relevant data first in context window
6. **DV8 Refactoring Guide path: first instance of each type only** — avoids path repetition without losing the link
7. **Churn interpretation example** — showing `"2993 lines...means IOUtils...touched in every bug fix"` trains the model to interpret, not just list

---

## VERSION 9 — "Start with / Then / End with" flowing ordering (current best)
**Date**: 2026-03-25
**Changes from V8**: Replaced "Answer in one continuous flow in this order: 1. 2. 3. 4." with natural language ordering ("Start with...", "Then...", "End with..."). Added concrete good/bad example for churn interpretation and structural flaw. Separated ranked lists from full details more clearly. Added "Worst files overall" as its own section before conclusion.
**Result**: Best yet with 32b — all 5 groups present, all paths included, concrete refactoring directions with actual class names. Structural flaw still slightly boilerplate for Package Cycle instances but instance-specific for MV and Clique.

### V9 ANSWER (32b, 2026-03-25)
```
### Worst by scope (top 5 groups by % of system descending):
- **Modularity Violation Instance 1**: 106.7%, bug_churn=2993
- **Package Cycle Instance 4**: 7.8%, bug_churn=1504
- **Clique Instance 1**: 13.8%, bug_churn=1004
- **Modularity Violation Instance 2**: 70.9%, bug_churn=2570
- **Package Cycle Instance 7**: 5.2%, bug_churn=1120

### Worst by pain (top 5 groups by bug_churn descending):
- **Modularity Violation Instance 1**: bug_churn=2993, 106.7%
- **Modularity Violation Instance 2**: bug_churn=2570, 70.9%
- **Package Cycle Instance 4**: bug_churn=1504, 7.8%
- **Clique Instance 1**: bug_churn=1004, 13.8%
- **Modularity Violation Instance 4**: bug_churn=2692, 42.9%

[Full details for all 5 groups with DV8 paths, CSV paths, instance-specific structural flaw analysis]

**Group name**: Modularity Violation Instance 1
**Size**: 286 files (106.7% of system) — spans full history DSM
**Key members**: IOUtils.java(0.846), FileUtils.java(0.697), FilenameUtils.java(0.488), ...
**Churn breakdown**: bug_churn=2993, nonbug_churn=-288, total_churn=2705 — 2993 lines changed purely in defect-fix commits means these files are touched in virtually every bug fix.
**Structural flaw**: These files change together despite no structural dependency, indicating hidden behavioral coupling. IOUtils.java is central to file operations but lacks direct dependencies with many of its co-changing files like FileUtils.java and FilenameUtils.java. This creates a maintenance nightmare as fixing one issue often triggers others.
**Refactoring direction**: Extract common functionality from IOUtils into dedicated classes (e.g., PathTokenizer for filename manipulation) to reduce coupling.
**DV8 Refactoring Guide**: .../modularity-violation/refactor-modularity-violation.html
**Full member list and dependency CSV**: .../modularity-violation/1

[Files that got worse over time: 5 files with fan-in/fan-out deltas and newly-joined instances]
[Worst files overall: top 5 by risk score with specific instance memberships]
[Conclusion: worst group + worst file + one concrete step]
```

**Known issues in V9:**
- "Worst by scope" list sorted wrong (106.7%, 7.8%, 13.8%, 70.9%... — model doesn't sort numerically)
- Package Cycle structural flaw descriptions copy-paste for Instance 4 and 7
- "Worst files" section lists same anti-pattern instances for all 5 files (all show MV1 + PC4 + Clique1) — data is correct but reads repetitively

---

## VERSION 10 — TIER 1 (CRITICAL) / TIER 2 (brief summary) split (current best)
**Date**: 2026-03-25
**Changes from V9**: Added CRITICAL labeling — groups appearing in BOTH ranked lists get full details (TIER 1), groups in only one list get a 3-4 line summary (TIER 2). Eliminates repetition, focuses depth on what matters most.
**Result**: Best answer yet. TIER 1 structural flaws are instance-specific (MV4 correctly analysed as "high-level utilities coupling to lower-level stream implementations"). TIER 2 summaries are clean and brief. Overlap notes included.

### V10 ANSWER (32b, 2026-03-25)

TIER 1 CRITICAL (3 groups): MV Instance 1, Package Cycle Instance 4, MV Instance 4 — all with full details, instance-specific structural flaw, DV8 paths, overlap notes.

TIER 2 (2 groups): Unhealthy Inheritance Instance 1 (scope-only), MV Instance 2 (pain-only) — 3-line summaries.

Example TIER 1 structural flaw (MV Instance 4 — first time this was instance-specific):
> "These files form a secondary coupling network where changes in high-level utilities (e.g., IOUtils) often require adjustments in lower-level streams (e.g., BufferedFileChannelInputStream). This creates a maintenance loop where fixing one issue triggers others across layers of the codebase."

**Known issues in V10:**
- Package Cycle Instance 1 appears in scope list but model skipped it in TIER 2
- Scope list still not numerically sorted (model sorts by position in context, not by value)
- Fix: pre-sort groups in `load_antipattern_groups()` output before injection so model sees them in correct order

---

## Run Command (qa-only mode)

```bash
cd "/Users/chrisharing/Desktop/RA Software Architecture Analsysis/AGENT/ARCH_AGENT/02_stage_interpret"

python3 interpret_temporal_bundle.py \
  --temporal-root "/Users/chrisharing/Desktop/RA Software Architecture Analsysis/AGENT/ARCH_AGENT/REPOS_ANALYZED/commons-io/temporal_analysis_4revisions_3month_diff_2025-05_to_2026-02_260321_185038" \
  --model "deepseek-r1:32b" \
  --qa-only \
  --user-question "Which parts of the system got worse over the last 6 months — show anti-pattern groups, files that gained the most dependencies, and the worst files overall."
```

For 70b (needs extended timeout — takes ~45 min on this machine):
```bash
python3 interpret_temporal_bundle.py \
  --temporal-root "/Users/chrisharing/Desktop/RA Software Architecture Analsysis/AGENT/ARCH_AGENT/REPOS_ANALYZED/commons-io/temporal_analysis_4revisions_3month_diff_2025-05_to_2026-02_260321_185038" \
  --model "deepseek-r1:70b" \
  --qa-only \
  --ollama-timeout-s 3600 \
  --user-question "Which parts of the system got worse over the last 6 months — show anti-pattern groups, files that gained the most dependencies, and the worst files overall."
```
