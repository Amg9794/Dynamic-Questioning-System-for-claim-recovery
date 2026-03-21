from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Question:
    id: str
    text: str
    question_field: str
    category: str
    priority: int                 # base priority 1-5 from question bank
    triggers: dict[str, Any]
    fill_fields: list[str]

    # Derived at load time
    gap_score: int = 0            # how many other questions' triggers this question unlocks
    incident_specific: bool = False   # True if has incident_type trigger

    def __repr__(self):
        return f"<Q {self.id} P{self.priority} [{self.category}] '{self.text[:55]}...'>"


@dataclass
class ScoredQuestion:
    question: Question
    effective_priority: int       # may differ from base due to Rule E (fraud escalation)
    gap_score: int
    incident_specificity: int     # 1 if incident-specific, 0 if generic
    score: float                  # final composite score (lower = higher rank)
    reason: str                   # human-readable explanation


def flatten_state(state: dict) -> dict:
    """
    Flatten a nested claim state dict into a single-level dict.
    E.g. state["incident_core"]["category"] -> flat["category"]
         state["third_party"]["third_party_involved"] -> flat["third_party_involved"]

    Also surfaces session_control fields at top level.
    Handles both flat (already-flat) and nested claim states.
    """
    flat = {}

    def _recurse(obj, _prefix=""):
        if not isinstance(obj, dict):
            return
        for k, v in obj.items():
            if k.startswith("_"):
                continue
            # Store both prefixed and unprefixed versions for maximum match coverage
            if _prefix:
                flat[f"{_prefix}_{k}"] = v
            flat[k] = v
            if isinstance(v, dict):
                _recurse(v, k)

    _recurse(state)
    return flat


def evaluate_trigger(triggers: dict, flat_state: dict) -> bool:
    """
    Returns True if ALL trigger conditions are satisfied.

    Trigger key semantics:
      incident_type             → list; flat_state["category"] must be in list
      required_fields_present   → list; all named fields must be non-null/non-NA in state
      <any other key>           → direct equality check against flat_state[key]
                                  If value is True → field must be truthy
                                  If value is False → field must be falsy
                                  If value is a string → field must equal that string
    """
    for key, expected in triggers.items():

        # ── Special: incident_type ──────────────────────────────────────────
        if key == "incident_type":
            actual_category = flat_state.get("category")
            if actual_category is None:
                # Category not yet known → suppress incident-specific questions
                return False
            if not isinstance(expected, list):
                return False
            if actual_category not in expected:
                return False

        # ── Special: required_fields_present ────────────────────────────────
        elif key == "required_fields_present":
            if not isinstance(expected, list):
                return False
            for req_field in expected:
                val = flat_state.get(req_field)
                if val is None or val == "__NA__":
                    return False   # prerequisite field not yet filled

        # ── Direct field value check ─────────────────────────────────────────
        else:
            actual = flat_state.get(key)
            if isinstance(expected, bool):
                # True → field must be truthy (not None, not False, not "__NA__")
                if expected:
                    if not actual or actual == "__NA__":
                        return False
                else:
                    # False → field must be explicitly False
                    if actual is not False:
                        return False
            elif isinstance(expected, str):
                if actual != expected:
                    return False
            elif isinstance(expected, list):
                # value must be in the list
                if actual not in expected:
                    return False
            else:
                if actual != expected:
                    return False

    return True


def is_field_answered(question: Question, flat_state: dict) -> bool:
    """
    Rule C: suppress if question_field or any fill_field appears in
    already_extracted_categories or answered_question_ids, OR if the
    target field already has a non-null value in the claim state.
    """
    extracted = set(flat_state.get("already_extracted_categories") or [])
    answered_ids = set(flat_state.get("answered_question_ids") or [])

    if question.id in answered_ids:
        return True

    if question.question_field in extracted:
        return True
    
    for ff in question.fill_fields:
        if ff in extracted:
            return True

    target_val = flat_state.get(question.question_field)
    if target_val is not None and target_val != "__NA__":
        return True

    return False

def fraud_flags_active(flat_state: dict) -> bool:
    """
    Rule E: returns True if any fraud_flags field is set to True.
    """
    fraud = flat_state.get("fraud_flags") or {}
    if isinstance(fraud, dict):
        return any(v is True for v in fraud.values())
    # Also check flat keys prefixed with fraud_flags_
    for k, v in flat_state.items():
        if k.startswith("fraud_flags_") and v is True:
            return True
    return False


class Retriever:

    def __init__(self, bank_path: str | Path):
        """
        Load and index the validated question bank.
        Computes gap_score for each question at load time (O(n²) over bank,
        done once — retrieval itself is O(n)).
        """
        self.bank_path = Path(bank_path)
        self.questions: list[Question] = []
        self._load_bank()
        self._compute_gap_scores()


    def _load_bank(self):
        with open(self.bank_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                q = Question(
                    id=raw["id"],
                    text=raw["text"],
                    question_field=raw["question_field"],
                    category=raw["category"],
                    priority=raw["priority"],
                    triggers=raw.get("triggers", {}),
                    fill_fields=raw.get("targets", {}).get("fill_fields", []),
                    incident_specific="incident_type" in raw.get("triggers", {})
                )
                self.questions.append(q)

    def _compute_gap_scores(self):
        # Build index: field_name -> count of questions requiring it as a prerequisite
        prerequisite_counts: dict[str, int] = {}
        for q in self.questions:
            rfp = q.triggers.get("required_fields_present", [])
            if isinstance(rfp, list):
                for field_name in rfp:
                    prerequisite_counts[field_name] = prerequisite_counts.get(field_name, 0) + 1

        # Also count direct trigger checks (non-special, boolean True)
        for q in self.questions:
            for k, v in q.triggers.items():
                if k in ("incident_type", "required_fields_present"):
                    continue
                if v is True:
                    prerequisite_counts[k] = prerequisite_counts.get(k, 0) + 1

        # Assign gap_score to each question
        for q in self.questions:
            score = 0
            for ff in q.fill_fields:
                score += prerequisite_counts.get(ff, 0)
            score += prerequisite_counts.get(q.question_field, 0)
            q.gap_score = score


    def next_question(self, state: dict) -> Question | None:
        ranked = self.next_questions(state, n=1)
        return ranked[0].question if ranked else None

    def next_questions(self, state: dict, n: int = 5) -> list[ScoredQuestion]:
        flat = flatten_state(state)
        fraud_active = fraud_flags_active(flat)

        # Stage 1: hard filter
        passing = self._stage1_filter(flat)

        # Stage 2: rank
        scored = self._stage2_rank(passing, flat, fraud_active)

        return scored[:n]

    def filter_stats(self, state: dict) -> dict:
        """Returns counts at each stage for diagnostics."""
        flat = flatten_state(state)
        passing = self._stage1_filter(flat)
        return {
            "bank_total": len(self.questions),
            "stage1_pass": len(passing),
            "stage1_removed": len(self.questions) - len(passing),
            "fraud_flags_active": fraud_flags_active(flat),
            "already_extracted": len(flat.get("already_extracted_categories") or []),
            "answered_ids": len(flat.get("answered_question_ids") or []),
        }

    def explain(self, state: dict, question_id: str) -> dict | None:
        flat = flatten_state(state)
        fraud_active = fraud_flags_active(flat)
        q = next((x for x in self.questions if x.id == question_id), None)
        if q is None:
            return {"error": f"Question {question_id} not found in bank"}

        trigger_pass = evaluate_trigger(q.triggers, flat)
        answered = is_field_answered(q, flat)
        passes_s1 = trigger_pass and not answered

        eff_pri = q.priority
        if q.category == "fraud_consistency" and fraud_active:
            eff_pri = min(eff_pri, 1)

        return {
            "id": q.id,
            "text": q.text,
            "category": q.category,
            "base_priority": q.priority,
            "effective_priority": eff_pri,
            "gap_score": q.gap_score,
            "incident_specific": q.incident_specific,
            "triggers": q.triggers,
            "fill_fields": q.fill_fields,
            "trigger_evaluation": {
                k: _explain_trigger(k, v, flat) for k, v in q.triggers.items()
            },
            "field_answered": answered,
            "passes_stage1": passes_s1,
            "fraud_escalation_applied": (
                q.category == "fraud_consistency" and fraud_active and q.priority > 1
            ),
        }

    def _stage1_filter(self, flat: dict) -> list[Question]:
        """
        Returns questions that:
          1. Pass all trigger conditions
          2. Are NOT already answered / extracted (Rule C)
        """
        passing = []
        for q in self.questions:
            if is_field_answered(q, flat):
                continue
            if not evaluate_trigger(q.triggers, flat):
                continue
            passing.append(q)
        return passing

    def _stage2_rank(
        self, questions: list[Question], flat: dict, fraud_active: bool
    ) -> list[ScoredQuestion]:
        """
        Rank by (effective_priority ASC, gap_score DESC, incident_specific DESC, id ASC).
        """
        scored = []
        for q in questions:
            eff_pri = q.priority

            # Rule E: fraud flags elevate fraud_consistency questions to P1
            if q.category == "fraud_consistency" and fraud_active:
                eff_pri = 1

            inc_spec = 1 if q.incident_specific else 0

            # Composite score: lower = better
            # Scale: priority dominates (×1000), then gap (×1, inverted), then specificity
            score = (eff_pri * 1000) - (q.gap_score * 10) - (inc_spec * 5)

            reason_parts = [f"P{eff_pri}"]
            if eff_pri != q.priority:
                reason_parts.append(f"(escalated from P{q.priority} — fraud flags active)")
            if q.gap_score > 0:
                reason_parts.append(f"gap={q.gap_score}")
            if inc_spec:
                reason_parts.append("incident-specific")

            scored.append(ScoredQuestion(
                question=q,
                effective_priority=eff_pri,
                gap_score=q.gap_score,
                incident_specificity=inc_spec,
                score=score,
                reason=", ".join(reason_parts)
            ))

        scored.sort(key=lambda s: (s.score, s.question.id))
        return scored


def _explain_trigger(key: str, expected: Any, flat: dict) -> dict:
    if key == "incident_type":
        actual = flat.get("category")
        passed = actual in (expected if isinstance(expected, list) else [])
        return {"type": "incident_type", "expected": expected, "actual": actual, "passed": passed}

    if key == "required_fields_present":
        results = {}
        for f in (expected if isinstance(expected, list) else []):
            val = flat.get(f)
            results[f] = {"value": val, "filled": val is not None and val != "__NA__"}
        all_passed = all(r["filled"] for r in results.values())
        return {"type": "required_fields_present", "fields": results, "passed": all_passed}

    actual = flat.get(key)
    if isinstance(expected, bool):
        passed = bool(actual) == expected if expected else (actual is False)
        return {"type": "boolean", "expected": expected, "actual": actual, "passed": passed}
    passed = actual == expected
    return {"type": "equality", "expected": expected, "actual": actual, "passed": passed}


if __name__ == "__main__":
    import sys
    from pathlib import Path
    BANK = Path(__file__).parent.parent / "ML_assign" / "question_bank_validated.jsonl"
    if not BANK.exists():
        print(f"Bank not found at {BANK}")
        sys.exit(1)

    print(f"\nLoading question bank from {BANK.name} ...")
    r = Retriever(BANK)
    print(f"Loaded {len(r.questions)} questions.\n")

    # ── Test states ──────────────────────────────────────────────────────────
    TESTS = [
        {
            "name": "Empty state — very first turn",
            "state": {
                "session_control": {
                    "already_extracted_categories": [],
                    "answered_question_ids": [],
                    "turn_count": 0,
                    "terminated": False
                }
            }
        },
        {
            "name": "Collision known, category answered",
            "state": {
                "incident_core": {"category": "collision"},
                "session_control": {
                    "already_extracted_categories": ["category"],
                    "answered_question_ids": ["QINC0001"],
                    "turn_count": 2
                }
            }
        },
        {
            "name": "Theft claim — FIR not yet filed",
            "state": {
                "incident_core": {"category": "theft"},
                "policy": {"policy_active": True, "policy_number": "POL-123"},
                "session_control": {
                    "already_extracted_categories": ["category", "policy_active", "policy_number"],
                    "answered_question_ids": [],
                    "turn_count": 3
                }
            }
        },
        {
            "name": "Collision + third party involved — hit and run",
            "state": {
                "incident_core": {"category": "collision"},
                "third_party": {"third_party_involved": True},
                "legal_and_reporting": {"hit_and_run": True},
                "session_control": {
                    "already_extracted_categories": ["category", "third_party_involved", "hit_and_run"],
                    "answered_question_ids": [],
                    "turn_count": 4
                }
            }
        },
        {
            "name": "Fraud flag active — should escalate fraud questions",
            "state": {
                "incident_core": {"category": "collision"},
                "fraud_flags": {"claim_timeline_inconsistency": True},
                "session_control": {
                    "already_extracted_categories": ["category"],
                    "answered_question_ids": [],
                    "turn_count": 5
                }
            }
        },
    ]

    for test in TESTS:
        print("=" * 65)
        print(f"  TEST: {test['name']}")
        print("=" * 65)

        stats = r.filter_stats(test["state"])
        print(f"  Bank: {stats['bank_total']} | S1 pass: {stats['stage1_pass']} "
              f"| Removed: {stats['stage1_removed']} | Fraud: {stats['fraud_flags_active']}")

        top5 = r.next_questions(test["state"], n=5)
        print(f"  Top 5 ranked questions:")
        for i, sq in enumerate(top5, 1):
            q = sq.question
            print(f"    {i}. [{sq.reason}] {q.id}  \"{q.text[:70]}\"")

        best = r.next_question(test["state"])
        if best:
            print(f"\n  → NEXT QUESTION: [{best.id}] \"{best.text}\"")

        # Show explain for best question
        if best:
            exp = r.explain(test["state"], best.id)
            print(f"\n  Explain {best.id}:")
            print(f"    base_priority:      {exp['base_priority']}")
            print(f"    effective_priority: {exp['effective_priority']}")
            print(f"    gap_score:          {exp['gap_score']}")
            print(f"    incident_specific:  {exp['incident_specific']}")
            print(f"    fraud_escalation:   {exp['fraud_escalation_applied']}")
            print(f"    passes_stage1:      {exp['passes_stage1']}")

        print()