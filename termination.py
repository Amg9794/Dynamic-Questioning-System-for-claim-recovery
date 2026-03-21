
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


MAX_TURNS               = 25
MIN_TURNS_BEFORE_STOP   = 3    # never terminate on turns 1–2
MIN_TURNS_FOR_P1P2_STOP = 6    # P1/P2 exhaustion only valid after turn 6

# All fields marked required:true in claim_state_schema.json
REQUIRED_FIELDS = {
    "category",
    "loss_datetime",
    "vehicle_make",
    "vehicle_model",
    "vehicle_year",
    "vehicle_registration_number",
    "drivable",
    "third_party_involved",
    "policy_number",
    "policy_active",
}

@dataclass
class TerminationDecision:
    should_stop: bool
    reason: str                          # human-readable explanation
    condition_code: str                  # C1 / C2 / C3 / C4 / C5 / CONTINUE
    diagnostics: dict = field(default_factory=dict)  # for audit + test assertions

    def __bool__(self):
        return self.should_stop

    def __repr__(self):
        return (
            f"TerminationDecision(stop={self.should_stop}, "
            f"code={self.condition_code!r}, reason={self.reason!r})"
        )

def _flatten(state: dict) -> dict:
    """Flatten nested claim state to a single-level dict."""
    flat = {}
    def _r(obj, prefix=""):
        if not isinstance(obj, dict):
            return
        for k, v in obj.items():
            if k.startswith("_"):
                continue
            full = f"{prefix}_{k}" if prefix else k
            flat[full] = v
            flat[k] = v          # also store without prefix for easy lookup
            if isinstance(v, dict):
                _r(v, k)
    _r(state)
    return flat


def _get(state: dict, *keys: str, default=None):
    """Get first matching key from flat or nested state."""
    flat = _flatten(state)
    for k in keys:
        if flat.get(k) is not None:
            return flat[k]
    return default


def _session(state: dict) -> dict:
    return state.get("session_control", {})


def _check_c4_safety_cap(turn_count: int) -> TerminationDecision | None:
    """C4: hard turn cap."""
    if turn_count >= MAX_TURNS:
        return TerminationDecision(
            should_stop=True,
            reason=f"Safety cap reached ({turn_count}/{MAX_TURNS} turns). "
                   "Ending session to prevent runaway interviews.",
            condition_code="C4",
            diagnostics={"turn_count": turn_count, "max_turns": MAX_TURNS},
        )
    return None


def _check_c5_policy_inactive(state: dict) -> TerminationDecision | None:
    """
    C5: early exit if policy is explicitly False.
    A lapsed policy means no downstream questions matter — terminate
    immediately and surface the reason so the claimant can be informed.
    """
    policy_active = _get(state, "policy_active")
    if policy_active is False:
        return TerminationDecision(
            should_stop=True,
            reason="Policy is inactive. The claim cannot proceed until the "
                   "policy status is resolved with the insurer.",
            condition_code="C5",
            diagnostics={"policy_active": False},
        )
    return None


def _check_c2_required_fields(state: dict) -> TerminationDecision | None:
    """C2: all required:true schema fields are filled."""
    flat = _flatten(state)
    missing = []
    filled  = []

    for field_name in REQUIRED_FIELDS:
        val = flat.get(field_name)
        if val is None or val == "__NA__":
            missing.append(field_name)
        else:
            filled.append(field_name)

    if not missing:
        return TerminationDecision(
            should_stop=True,
            reason="All required fields collected "
                   f"({len(REQUIRED_FIELDS)}/{len(REQUIRED_FIELDS)}).",
            condition_code="C2",
            diagnostics={
                "required_total": len(REQUIRED_FIELDS),
                "required_filled": len(filled),
                "required_missing": missing,
            },
        )
    return None


def _check_c1_p1p2_exhausted(retriever, state: dict, turn_count: int) -> TerminationDecision | None:
    """
    C1: no Priority-1 or Priority-2 questions remain after the Stage 1 filter.
    Only evaluated after MIN_TURNS_FOR_P1P2_STOP to avoid premature cutoff.
    """
    if turn_count < MIN_TURNS_FOR_P1P2_STOP:
        return None

    remaining = retriever.next_questions(state, n=100)
    p1p2 = [sq for sq in remaining if sq.effective_priority <= 2]

    if not p1p2:
        all_remaining_priorities = sorted({sq.effective_priority for sq in remaining})
        return TerminationDecision(
            should_stop=True,
            reason="All Priority-1 and Priority-2 questions have been answered. "
                   f"{len(remaining)} lower-priority questions remain but are not "
                   "required for claim eligibility determination.",
            condition_code="C1",
            diagnostics={
                "p1p2_remaining": 0,
                "lower_priority_remaining": len(remaining),
                "remaining_priorities": all_remaining_priorities,
            },
        )
    return None


def _check_c3_no_questions(retriever, state: dict) -> TerminationDecision | None:
    """C3: zero questions survive the Stage 1 hard filter."""
    stats = retriever.filter_stats(state)
    if stats["stage1_pass"] == 0:
        return TerminationDecision(
            should_stop=True,
            reason="No questions remain that match the current claim state. "
                   "All applicable questions have been asked or are filtered out.",
            condition_code="C3",
            diagnostics={
                "bank_total": stats["bank_total"],
                "stage1_pass": 0,
                "already_extracted": stats["already_extracted"],
                "answered_ids": stats["answered_ids"],
            },
        )
    return None


def should_terminate(
    state: dict,
    history: list[dict],
    retriever,
) -> TerminationDecision:
    sc          = _session(state)
    turn_count  = sc.get("turn_count", len(history))
    flat        = _flatten(state)


    if turn_count < MIN_TURNS_BEFORE_STOP:
        return TerminationDecision(
            should_stop=False,
            reason="",
            condition_code="CONTINUE",
            diagnostics={"turn_count": turn_count, "min_turns": MIN_TURNS_BEFORE_STOP},
        )

    decision = _check_c4_safety_cap(turn_count)
    if decision:
        return decision

    decision = _check_c5_policy_inactive(state)
    if decision:
        return decision

    decision = _check_c2_required_fields(state)
    if decision:
        return decision
    
    decision = _check_c3_no_questions(retriever, state)
    if decision:
        return decision

    decision = _check_c1_p1p2_exhausted(retriever, state, turn_count)
    if decision:
        return decision

    remaining  = retriever.next_questions(state, n=100)
    p1p2_left  = [sq for sq in remaining if sq.effective_priority <= 2]
    stats      = retriever.filter_stats(state)

    missing_required = [
        f for f in REQUIRED_FIELDS
        if flat.get(f) is None or flat.get(f) == "__NA__"
    ]

    return TerminationDecision(
        should_stop=False,
        reason="",
        condition_code="CONTINUE",
        diagnostics={
            "turn_count": turn_count,
            "stage1_pass": stats["stage1_pass"],
            "p1p2_remaining": len(p1p2_left),
            "required_fields_missing": missing_required,
            "fraud_flags_active": stats["fraud_flags_active"],
        },
    )

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent / "task3"))
    from retriever import Retriever

    BANK = Path(__file__).parent.parent / "task2" / "question_bank_validated.jsonl"
    r = Retriever(BANK)

    def run(label, state, history=None):
        d = should_terminate(state, history or [], r)
        status = "STOP" if d.should_stop else "CONTINUE"
        print(f"  [{status:8}] {label}")
        if d.should_stop:
            print(f"             code={d.condition_code}  reason={d.reason[:70]}")
        else:
            diag = d.diagnostics
            print(f"             P1/P2 left={diag.get('p1p2_remaining','?')}  "
                  f"required missing={diag.get('required_fields_missing','?')}")
        print()

    print("\n" + "="*60)
    print("  TASK 5 — TERMINATION SMOKE TESTS")
    print("="*60 + "\n")

    run("Empty state (turn 0)",
        {"session_control": {"turn_count": 0}})

    run("Turn 3, nothing answered",
        {"session_control": {"turn_count": 3,
         "already_extracted_categories": [], "answered_question_ids": []}})

    run("Policy inactive → C5",
        {"policy": {"policy_active": False},
         "session_control": {"turn_count": 5,
         "already_extracted_categories": ["policy_active"],
         "answered_question_ids": []}})

    run("Safety cap → C4",
        {"session_control": {"turn_count": 26,
         "already_extracted_categories": [], "answered_question_ids": []}})

    run("All required fields filled → C2",
        {
            "incident_core": {"category": "collision", "loss_datetime": "2024-01-15 14:00"},
            "vehicle": {"make": "Hyundai", "model": "Creta", "year": "2022",
                        "registration_number": "MH01AB1234"},
            "damage_assessment": {"drivable": False},
            "third_party": {"third_party_involved": True},
            "policy": {"policy_number": "HDFC-001", "policy_active": True},
            "session_control": {"turn_count": 10,
                "already_extracted_categories": list(REQUIRED_FIELDS),
                "answered_question_ids": []},
        })

    run("Mid-session collision, 3 turns done",
        {
            "incident_core": {"category": "collision"},
            "policy": {"policy_active": True},
            "session_control": {"turn_count": 3,
                "already_extracted_categories": ["category", "policy_active"],
                "answered_question_ids": []},
        })