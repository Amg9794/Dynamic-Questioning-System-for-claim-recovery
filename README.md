# Dynamic Insurance Claim Intake System

**Meeami Technologies — ML Engineering Assessment**

A state-driven, streaming question engine for vehicle insurance claim intake. The system conducts a structured interview with a claimant, selecting the next best question from a 1,200-question bank based on the evolving claim state — with no ML in the retrieval path.

---

## Table of Contents

1. [What This System Does](#what-this-system-does)
2. [Project Structure](#project-structure)
3. [Quick Start](#quick-start)
4. [Task-by-Task Explanation](#task-by-task-explanation)
   - [Task 0 — Foundation](#task-0--foundation)
   - [Task 1 — Question Bank](#task-1--question-bank)
   - [Task 2 — Validator Pipeline](#task-2--validator-pipeline)
   - [Task 3 — Retriever Engine](#task-3--retriever-engine)
   - [Task 4 — CLI Demo Loop](#task-4--cli-demo-loop)
   - [Task 5 — Termination Policy](#task-5--termination-policy)
   - [Task 6 — Test Suite](#task-6--test-suite)
5. [Architecture and Design Decisions](#architecture-and-design-decisions)
6. [How to Run](#how-to-run)
7. [File Reference](#file-reference)

---

## What This System Does

Traditional insurance claim forms ask every question regardless of context — repeating fields that are already answered, ignoring the incident type, and providing no intelligence about what matters next.

This system replaces that with a **dynamic interview** that:

- Selects the next best question based on the current claim state
- Suppresses questions whose answers are already known
- Asks collision-specific questions only for collision claims, theft-specific only for theft claims, and so on
- Automatically promotes fraud-consistency checks when red flags appear in the session
- Parses free-text user responses into structured JSON using the Claude API
- Terminates the session when sufficient information has been collected
- Produces a structured claim summary and a full audit log

**Per-turn flow:**

```
Retriever selects next question
          |
          v
Question shown to user
          |
          v
User types free-text response
          |
          v
Claude API parses response --> JSON patch
          |
          v
StateManager merges patch into claim state
          |
          v
already_extracted_categories updated
          |
          v
Termination check --> if done: save outputs and exit
          |
          v
Next turn
```

---

## Project Structure


```
├── ML Assign/
│   ├── claim_state_schema.json        ← all claim fields, types, required flags
│   ├── domain_vocabulary.json         ← controlled enums (incident types, damage areas, etc.)
│   └── priority_policy.md             ← rules for question ordering and termination
│   ├── build_question_bank.py         ← generates question_bank_raw.jsonl
│   └── question_bank_raw.jsonl        ← 2400 questions with full metadata
│   ├── validator.py                   ← 4-stage validation pipeline
│   ├── question_bank_validated.jsonl  ← validated output (input to retriever)
│   └── validation_report.json         ← full structured validation report
│   └── retriever.py                   ← Stage 1 filter + Stage 2 ranking engine
│   └── demo_loop.py                   ← end-to-end CLI session
│   └── termination.py                 ← should_terminate() function
│   └── test_cases.py                  ← 15 test cases (15/15 passing)
---

---

## Quick Start

**Step 1 — Place these files in one folder:**

```
demo_loop.py
retriever.py
question_bank_validated.jsonl
claim_state_schema.json
domain_vocabulary.json
```

**Step 2 — Set your API key (optional but recommended for full NLU):**

```bash
# Windows
set ANTHROPIC_API_KEY=sk-ant-...

# Mac / Linux
export ANTHROPIC_API_KEY=sk-ant-...
```

**Step 3 — Run:**

```bash
# Built-in demo (no typing required)
python demo_loop.py --demo

# Interactive session (you type the answers)
python demo_loop.py

# Run the full test suite
python test_cases.py
```

> **No API key?** The system runs in mock mode using basic keyword extraction. Full multi-field NLU extraction requires the Claude API.

> **Dependencies:** Only `requests` is required — included in most Python environments. No other installs needed.

---

## Task-by-Task Explanation

---

### Task 0 — Foundation

**What it was:** Before any other task could begin,three files that define the complete contract for the entire system need to be created 

**Three deliverables:**

| File | Purpose |
|---|---|
| `claim_state_schema.json` | Defines every claim field — type, required flag, default value |
| `domain_vocabulary.json` | All allowed enum values in one place |
| `priority_policy.md` | Rules for question ordering and when to end the session |

**Key design decision — three distinct null values:**

| Value | Meaning | Retriever behaviour |
|---|---|---|
| `null` | Field not yet asked | May ask a question targeting this field |
| `"__NA__"` | Not applicable for this incident type | Must skip questions targeting this field |
| `false` | User explicitly confirmed negative | Suppress the question but react to the value downstream |

This distinction matters because conflating `null` and `"__NA__"` causes the retriever to incorrectly suppress or trigger questions. A flood-specific question on a theft claim should be `"__NA__"` — not `null` — so the retriever knows it is permanently inapplicable, not simply unanswered.

**Why this approach:**

The schema was designed not as a database schema, but as the **retriever's runtime input object**. Every turn, the retriever receives one JSON object and must evaluate trigger conditions in O(1). This required flat typed fields, explicit null conventions, and session control fields all living inside the same object — no synchronisation, no multi-object lookups.

The vocabulary was kept in a separate file so that Task 1 (question generation), Task 2 (validation), Task 3 (retrieval), and Task 4 (NLU extraction) all share the same enum values. One change propagates everywhere.

Explicit integer priority rules were chosen over ML ranking because every question selection in an insurance system must be **auditable**. A rule-based system can always answer "why was this question asked" — a learned ranker cannot.

---

### Task 1 — Question Bank

**What it was:** Generate 1,200 questions — 12 categories × 100 each — where every question carries not just text but full retrieval metadata.

**Question format:**

```json
{
  "id": "QTHI0001",
  "text": "Was any other vehicle or person involved in the incident?",
  "question_field": "third_party_involved",
  "category": "third_party_details",
  "priority": 1,
  "triggers": {},
  "targets": { "fill_fields": ["third_party_involved"] }
}
```

**12 categories and their coverage:**

| Category | Focus |
|---|---|
| `incident_core` | Universal — applies to every claim type |
| `collision_dynamics` | Impact direction, speed, fault, road conditions |
| `damage_assessment` | Damage zones, severity, drivability, repair estimate |
| `third_party_details` | Other vehicle, driver, insurer, hit-and-run |
| `legal_reporting` | FIR, police, witnesses, legal notices |
| `policy_eligibility` | Coverage type, add-ons, NCB, IDV |
| `fraud_consistency` | Timeline, damage consistency, document checks |
| `repair_settlement` | Workshop, surveyor, cashless vs reimbursement |
| `theft_specific` | GPS tracker, keys, forced entry — theft only |
| `fire_specific` | Fire source, fire brigade, arson — fire only |
| `flood_specific` | Water level, engine submersion — flood only |
| `vandalism_specific` | Vandalism type, motive, suspects — vandalism only |

**Three trigger types:**

```python
# Incident gate — question appears only for a specific incident type
{"incident_type": ["theft"]}

# Prerequisite gate — a prior field must be answered first
{"required_fields_present": ["police_report_filed"]}

# Direct value gate — question appears only when a specific value is confirmed
{"third_party_involved": True}
```

**Why this approach:**

Hardcoded questions were chosen over API generation for **determinism** — the same bank every run, stable test cases, and no hallucinated enum values.

The `fill_fields` in each question were assigned deliberately because they drive the **gap score calculation** in the retriever. The question for `third_party_involved` fills a field that 200+ other questions require as a prerequisite — giving it a gap score of 200 and placing it first in every session. This was intentional design, not coincidence.

The fraud category was kept separate because its effective priority changes dynamically at runtime (Rule E escalation) and must remain auditable as a distinct concern from damage and legal questions.

---

### Task 2 — Validator Pipeline

**What it was:** Provide a guarantee that the question bank is correct — structurally, logically, uniquely, and comprehensively — before the retriever ever sees it.

**Data flow:**

```
question_bank_raw.jsonl        (Task 1 output)
          |
          v
      validator.py
          |
          v
question_bank_validated.jsonl  (Task 3 input)
validation_report.json
```

**4-stage pipeline:**

| Stage | What it checks |
|---|---|
| 1 — Schema | Required fields present, ID format, category enum, priority 1–5, text contains "?", fill_fields non-empty |
| 2 — Logic | Trigger key validity, priority consistency rules, fraud P1 guards |
| 3 — Deduplication | Exact text match + Jaccard similarity ≥ 0.85 within the same category |
| 4 — Coverage | All 11 required schema fields covered; all 12 categories have ≥ 100 questions |

**Final result:**

```
Input:    1,200 questions
Stage 1:  0 errors removed
Stage 2:  0 errors removed  (1,056 warnings kept — runtime fields, not malformed)
Stage 3:  0 duplicates removed
Stage 4:  11/11 required fields covered  ·  12/12 categories at ≥ 100
Output:   1,200 validated questions — 100% pass rate
```

**Key decision — Errors vs Warnings:**

Some trigger keys did not appear in the formal schema document (e.g. `challan_issued`, `total_loss_assessed`) but are valid runtime claim-state fields that exist during a live session. Treating these as errors would have removed 22 legitimate questions.

Rule applied: **clearly malformed → error and remove | valid runtime field → warning and keep**

**Why this approach:**

Pipeline stages enforce separation of concerns. A single validation function would hide Stage 2 errors whenever Stage 1 fails. Each stage has one clear responsibility, making failures straightforward to diagnose.

The coverage check sits in Stage 4 — not Stage 1 — because coverage is a property of the **entire bank**, not of individual questions. Individual validation must run first; collective completeness is verified last.

---

### Task 3 — Retriever Engine

**What it was:** Given the current claim state at any turn, select the single best next question from 1,200 candidates in O(n).

**Two-stage architecture:**

```
Stage 1: Hard Filter  -->  eliminate ineligible questions completely
Stage 2: Ranking      -->  score and order what survived
```

**Stage 1 — Hard Filter:**

Every question's trigger conditions are evaluated against the flattened claim state. All conditions must pass. Rule C is also enforced here — if the target field already has a value in state, or appears in `already_extracted_categories`, the question is suppressed.

**Stage 2 — Composite Score:**

```python
score = (effective_priority × 1000) - (gap_score × 10) - (incident_specific × 5)
# Lower score = better rank
```

**Five ordering rules:**

| Rule | What it does | How it is implemented |
|---|---|---|
| A — Gap Fill First | Prefer questions that unlock the most downstream questions | `gap_score` computed once at load time — counts how many other questions list this question's fill_fields as a prerequisite |
| B — Incident-Specific First | At the same priority, incident-specific beats generic | `incident_specific = 1` if the question has an `incident_type` trigger |
| C — Never Re-Ask | Suppress questions targeting already-answered fields | Stage 1 removes them before ranking begins |
| D — Correction Handling | When the user corrects a prior answer, re-enable the field | StateManager removes the field from `already_extracted_categories` on receiving `_correction` |
| E — Fraud Escalation | When fraud flags are set, promote fraud questions to P1 | `effective_priority = 1` for all `fraud_consistency` questions when any `fraud_flags` value is `True` |

**Public API:**

```python
retriever = Retriever("question_bank_validated.jsonl")

retriever.next_question(state)           # single best question
retriever.next_questions(state, n=5)     # top N ranked questions
retriever.explain(state, "QTHI0001")     # full scoring breakdown for one question
retriever.filter_stats(state)            # Stage 1 diagnostics
```

**Gap score examples:**

```
third_party_involved  →  200+ questions require it as a prerequisite  →  gap_score = 200
police_report_filed   →  60+ questions require it as a prerequisite   →  gap_score = 64
```

This means `third_party_involved` is the first question in every fresh session by design.

**Why this approach:**

The two-stage design ensures Stage 1 completely eliminates ineligible questions before any ranking occurs. Without this, a high-gap-score P2 question could incorrectly outrank an unanswered P1 question.

The ×1000 multiplier on priority guarantees that priority always dominates. No gap score value can make a P2 question beat a P1 question.

The `explain()` method makes every selection fully auditable — a requirement in insurance systems where decisions must be traceable to specific rules.

---

### Task 4 — CLI Demo Loop

**What it was:** Connect Tasks 0–3 into a working end-to-end conversation.

**Five components:**

| Component | Responsibility |
|---|---|
| `NLUExtractor` | Call Claude API to parse free text into a structured JSON patch |
| `StateManager` | Merge the JSON patch into the nested claim state |
| `AuditLog` | Record every turn — question asked, user response, extracted fields, NLU latency |
| `check_termination` | Decide whether to end the session |
| `run_session` | Coordinate all components in a loop |

**NLU system prompt design:**

Three elements were injected into every Claude API call:
1. Schema field types and enums — so the model knows `"collision"` is valid and `"car accident"` is not
2. List of boolean fields — so `police_report_filed` is returned as `true`/`false`, not as a string
3. Worked examples — showing the exact expected output format

Output is always sparse — only explicitly mentioned fields are returned:

```
User: "Yes, a police report was filed at MG Road station"
→ {"police_report_filed": true, "police_station_name": "MG Road"}

User: "What do you mean?"
→ {}
```

**Correction detection:**

```
User: "Actually it was not a collision — it was theft"
→ {
    "category": "theft",
    "_correction": {
      "field": "category",
      "old_value": "collision",
      "new_value": "theft"
    }
  }
```

The StateManager removes `"category"` from `already_extracted_categories` — the retriever treats it as unanswered and will ask again.

**Three output files per session:**

```
session_{id}/
  claim_summary.json   →  filled fields only — for the claims adjuster
  audit_log.json       →  every turn recorded — for compliance
  final_state.json     →  complete raw state including nulls — for debugging
```

**Flat file layout supported:**

Place all files in the same folder and the system auto-detects the layout. This resolves the Windows path issue where a nested `parent/task0/` directory does not exist.

**Why this approach:**

The Claude API is used only for NLU — not for retrieval or question selection. Free-text parsing is the one task where ML provides unique value that rules cannot replace. Everything else — retrieval, ranking, termination — is rule-based, deterministic, and auditable.

A mock extractor was included for offline testing. Production systems should always have a fallback when an external dependency is unavailable.

Three separate output files were created because each has a different consumer: the claim summary for the adjuster, the audit log for compliance, and the full state for developers.

---

### Task 5 — Termination Policy

**What it was:** A standalone, independently testable function that decides when a session should end.

**Function signature:**

```python
def should_terminate(state, history, retriever) -> TerminationDecision
```

**Return type:**

```python
@dataclass
class TerminationDecision:
    should_stop:     bool   # True = end the session
    reason:          str    # human-readable — can be shown to the user
    condition_code:  str    # C1 / C2 / C3 / C4 / C5 / CONTINUE — for the audit log
    diagnostics:     dict   # details for test assertions and logging
```

**Five conditions evaluated in this exact order:**

| Code | Condition | Trigger logic |
|---|---|---|
| C4 | Safety Cap | `turn_count ≥ 25` — enforced regardless of state |
| C5 | Policy Inactive | `policy_active = False` — claim cannot proceed without an active policy |
| C2 | Required Complete | All 10 `required: true` schema fields are filled |
| C3 | No Questions Left | Stage 1 filter returns 0 matching questions |
| C1 | P1/P2 Exhausted | No Priority-1 or Priority-2 questions remain after turn 6 |

**Minimum turn guard:** The session never terminates before turn 3. This prevents premature exit when a user provides rich context in their first message that happens to fill several required fields at once.

**CONTINUE also returns diagnostics:**

```python
{
    "p1p2_remaining": 45,
    "required_fields_missing": ["vehicle_make", "loss_datetime"],
    "stage1_pass": 423,
    "fraud_flags_active": False
}
```

**Why this approach:**

Termination was extracted into a separate function — not embedded in the demo loop — because it needs to be independently testable, reusable across different hosts, and auditable via the `condition_code` field.

The order of conditions matters: C4 first because the safety cap must be enforced regardless of state. C5 second because an inactive policy makes every subsequent question pointless. C2, C3, and C1 follow in order of completion signal strength.

C5 was added beyond the four conditions in the policy document as a domain-specific optimisation. The `history` parameter is included for future use without requiring a signature change.

---

### Task 6 — Test Suite

**What it was:** Prove that the system works correctly — every retrieval rule, every termination condition, every branching scenario.

**15 test cases:**

| Test | Scenario | Group |
|---|---|---|
| TC01 | Empty state → first question must be Priority 1 | Retriever |
| TC02 | Rule A: `QTHI0001` (gap=200) ranks highest among P1 questions | Retriever |
| TC03 | Rule B: incident-specific question beats generic at same priority | Retriever |
| TC04 | Rule C: question suppressed when field is in `already_extracted_categories` | Retriever |
| TC05 | Rule C: question suppressed when target field already has a value in state | Retriever |
| TC06 | Rule E: fraud flag active → all fraud questions get `effective_priority = 1` | Retriever |
| TC07 | Collision branch: collision questions activate only after `category = collision` | Branching |
| TC08 | Theft gating: theft questions must not appear for a collision claim | Branching |
| TC09 | Hit-and-run: follow-up questions activate when `hit_and_run = True` | Branching |
| TC10 | Third-party: TP questions suppressed when `third_party_involved = False` | Branching |
| TC11 | C4: session terminates at turn 26 | Termination |
| TC12 | C5: session terminates when `policy_active = False` | Termination |
| TC13 | C2: session terminates when all required fields are filled | Termination |
| TC14 | CONTINUE: mid-session state with P1/P2 remaining does not terminate | Edge case |
| TC15 | Minimum turn guard: no termination at turn 0, 1, or 2 | Edge case |

**Result: 15/15 passing**

No external test framework is required. A pure Python decorator pattern handles pass/fail tracking, assertion messages, and result reporting. `python test_cases.py` is all that is needed.

**Why this approach:**

Tests were selected using risk-based prioritisation — scenarios where a bug causes the most damage (wrong question ordering every session, infinite loops, incorrect branch activation) were prioritised over exhaustive coverage.

Specific ID assertions such as asserting `QTHI0001` in TC02 are intentional regression tests. If the question bank is ever modified in a way that breaks a critical invariant, the test fails immediately and signals that something important has changed.

---

## Architecture and Design Decisions

### Why no ML in the retrieval path?

| Concern | ML Ranking | Rule-based (this system) |
|---|---|---|
| Auditable — can explain every decision | No | Yes |
| Deterministic — same state gives same question | No | Yes |
| Requires labelled training data | Yes | No |
| Behaviour can change unexpectedly | Yes | No |

Insurance systems require every decision to be explainable. "The model selected this question" is not an acceptable answer. "This question has Priority 1, gap score 200, and all its trigger conditions are satisfied" is.

### Why three null values?

Without the three-way distinction the retriever makes silent errors:

- `null` — the retriever can ask; it does not know the answer yet
- `"__NA__"` — the retriever must skip; this field is permanently irrelevant for this claim type
- `false` — suppress the question, but downstream logic can react to the confirmed negative

A flood-specific question on a theft claim must be `"__NA__"`, not `null`, so the retriever does not treat it as simply unanswered.

### Why validate before retrieval?

```
Raw questions  →  Validated questions  →  Retriever
```

The retriever trusts that trigger keys map to real schema fields, enum values are valid, and fill_fields are correct. Without the validation layer, a malformed trigger silently fails — causing a question to never appear or always appear incorrectly.

### Why compute gap scores at load time?

Gap score computation scans every question against every other question — O(n²). This runs once when the `Retriever` is initialised. Each session turn is then O(n), fast enough for real-time use.

### Why separate termination from the demo loop?

Embedding `should_terminate()` inside the loop would make it untestable without running a full session, non-reusable in other contexts, and non-auditable. A standalone function with a structured return type solves all three problems.

---

## How to Run

### Interactive session with Claude API

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python demo_loop.py
```

### Built-in demo replay

```bash
python demo_loop.py --demo
```

### Custom replay file

```bash
python demo_loop.py --replay answers.txt
```

### Test suite

```bash
python test_cases.py
```

### Regenerate the question bank from scratch

```bash
python build_question_bank.py    # produces question_bank_raw.jsonl
python validator.py              # produces question_bank_validated.jsonl
```

### All command-line options

```
python demo_loop.py --help

options:
  --bank PATH        Path to validated question bank JSONL
  --schema PATH      Path to claim_state_schema.json
  --vocab PATH       Path to domain_vocabulary.json
  --session-id ID    Custom session ID (auto-generated if not provided)
  --replay PATH      Path to replay answers file (one answer per line)
  --demo             Run the built-in demo replay
```

---

## File Reference

| File | Task | Description |
|---|---|---|
| `claim_state_schema.json` | 0 | 91 fields across 12 groups; 11 marked required |
| `domain_vocabulary.json` | 0 | Controlled enums — incident types, damage areas, coverage types, fraud indicators |
| `priority_policy.md` | 0 | 5 priority levels, 5 ordering rules, termination conditions |
| `build_question_bank.py` | 1 | Deterministic question generator |
| `question_bank_raw.jsonl` | 1 | 1,200 questions with full retrieval metadata |
| `validator.py` | 2 | 4-stage pipeline — schema, logic, deduplication, coverage |
| `question_bank_validated.jsonl` | 2 | 1,200 validated questions — input to the retriever |
| `validation_report.json` | 2 | Full structured report — errors, warnings, coverage statistics |
| `retriever.py` | 3 | `Retriever` class — Stage 1 filter and Stage 2 ranking |
| `demo_loop.py` | 4 | End-to-end CLI — NLU extraction, state manager, audit log |
| `termination.py` | 5 | `should_terminate()` — 5 conditions, returns `TerminationDecision` |
| `test_cases.py` | 6 | 15 test cases — 15/15 passing |
| `technical_report.docx` | — | 10-section technical report |
| `presentation.pptx` | — | 8-slide presentation deck |

---

## Summary

The system demonstrates that a robust, auditable, and state-aware insurance claim intake engine can be built without ML in the core retrieval path. Explicit priority rules combined with a rich trigger-based question bank and a Claude-powered NLU layer produce a system that is accurate, deterministic, and fully traceable. Every question selection, every state update, and every termination decision can be explained from first principles.
