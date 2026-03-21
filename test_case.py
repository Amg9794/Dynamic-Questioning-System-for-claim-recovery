from __future__ import annotations

import sys
import traceback
from pathlib import Path
from dataclasses import dataclass

# PATH SETUP
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent / "ML_assign"))
sys.path.insert(0, str(BASE.parent / "ML_assign"))

from retriever import Retriever
from termination import should_terminate, REQUIRED_FIELDS

BANK = BASE.parent / "ML_assign" / "question_bank_raw.jsonl"

# TEST HARNESS 

@dataclass
class TestResult:
    name: str
    passed: bool
    message: str
    detail: str = ""

_results: list[TestResult] = []

def test(name: str):
    """Decorator: run function, catch assertion errors, record result."""
    def decorator(fn):
        try:
            fn()
            _results.append(TestResult(name, True, "OK"))
        except AssertionError as e:
            _results.append(TestResult(name, False, str(e)))
        except Exception as e:
            _results.append(TestResult(name, False, f"EXCEPTION: {e}",
                                        traceback.format_exc()))
        return fn
    return decorator

def assert_eq(actual, expected, label=""):
    prefix = f"{label}: " if label else ""
    assert actual == expected, f"{prefix}expected {expected!r}, got {actual!r}"

def assert_in(item, collection, label=""):
    prefix = f"{label}: " if label else ""
    assert item in collection, f"{prefix}{item!r} not found in {collection!r}"

def assert_not_in(item, collection, label=""):
    prefix = f"{label}: " if label else ""
    assert item not in collection, f"{prefix}{item!r} should NOT be in {collection!r}"

def assert_priority_lte(sq_list, max_priority: int, label=""):
    prefix = f"{label}: " if label else ""
    for sq in sq_list:
        assert sq.effective_priority <= max_priority, \
            f"{prefix}question {sq.question.id} has priority {sq.effective_priority} > {max_priority}"

r = Retriever(BANK)

def state_with(**kwargs) -> dict:
    """Build a minimal valid state dict with session_control."""
    s = {
        "session_control": {
            "turn_count": kwargs.pop("turn_count", 5),
            "already_extracted_categories": kwargs.pop("extracted", []),
            "answered_question_ids": kwargs.pop("answered", []),
            "terminated": False,
            "termination_reason": None,
        }
    }
    s.update(kwargs)
    return s

def full_required_state() -> dict:
    """State with every required field filled."""
    return state_with(
        turn_count=12,
        extracted=list(REQUIRED_FIELDS),
        incident_core={"category": "collision", "loss_datetime": "2024-01-15 14:00"},
        vehicle={"make": "Hyundai", "model": "Creta", "year": "2022",
                 "registration_number": "MH01AB1234"},
        damage_assessment={"drivable": False},
        third_party={"third_party_involved": True},
        policy={"policy_number": "HDFC-001", "policy_active": True},
    )


# RETRIEVER TESTS
@test("TC01 — Empty state: first question must be P1")
def tc01():
    """On an empty claim state the retriever must return a Priority-1 question."""
    state = state_with(turn_count=0)
    top = r.next_questions(state, n=5)
    assert len(top) > 0, "No questions returned for empty state"
    assert_eq(top[0].effective_priority, 1, "Top question priority")


@test("TC02 — Rule A: gap score drives ordering within same priority")
def tc02():
    """
    third_party_involved (gap≈200) should rank above police_report_filed (gap≈64)
    at priority 1 when both are unanswered and neither has a trigger condition.
    """
    state = state_with(turn_count=1)
    top20 = r.next_questions(state, n=20)
    p1_ids = [sq.question.id for sq in top20 if sq.effective_priority == 1]
    gap_scores = {sq.question.id: sq.gap_score for sq in top20}

    # QTHI0001 is third_party_involved — highest gap scorer
    assert "QTHI0001" in p1_ids, "QTHI0001 (third_party_involved) should be in top P1 questions"
    # Its gap score should be the highest of the P1 bunch
    qthi_gap = gap_scores.get("QTHI0001", 0)
    for qid, gs in gap_scores.items():
        if qid != "QTHI0001" and qid in p1_ids:
            assert qthi_gap >= gs, \
                f"QTHI0001 gap={qthi_gap} should be ≥ {qid} gap={gs}"


@test("TC03 — Rule B: incident-specific question beats generic at same priority")
def tc03():
    """
    With category=theft and police_report_filed still unknown,
    the theft-specific FIR timeliness question (QTHF0019, incident-specific)
    should rank above a same-priority generic question with no incident_type trigger.
    """
    state = state_with(
        turn_count=4,
        extracted=["category", "third_party_involved", "policy_active"],
        incident_core={"category": "theft"},
        policy={"policy_active": True},
        third_party={"third_party_involved": False},
    )
    top10 = r.next_questions(state, n=10)
    ids = [sq.question.id for sq in top10]

    # QTHF0019 = "Was the FIR filed within 24 hours of discovering the theft?"
    # It has incident_type=["theft"] + priority 1
    assert "QTHF0019" in ids, \
        "Theft-specific FIR timeliness question should appear in top 10 for theft claim"

    # It should rank above any generic P1 question with lower gap score
    thf_rank = ids.index("QTHF0019")
    for i, sq in enumerate(top10):
        q = sq.question
        if not q.incident_specific and sq.effective_priority == 1 and sq.gap_score < top10[thf_rank].gap_score:
            assert i >= thf_rank, \
                f"Generic P1 question {q.id} (gap={sq.gap_score}) ranked above incident-specific QTHF0019"


@test("TC04 — Rule C: answered field is suppressed")
def tc04():
    """
    If 'category' is in already_extracted_categories, QINC0001
    ('What type of incident occurred?') must not appear in results.
    """
    state = state_with(
        turn_count=3,
        extracted=["category"],
        incident_core={"category": "collision"},
    )
    all_ids = {sq.question.id for sq in r.next_questions(state, n=500)}
    assert "QINC0001" not in all_ids, \
        "QINC0001 (category question) should be suppressed after category is extracted"


@test("TC05 — Rule C: question suppressed when target field has a value")
def tc05():
    """
    If policy_active is already True in state, all questions targeting
    policy_active should be filtered out even without explicit extraction list entry.
    """
    state = state_with(
        turn_count=4,
        policy={"policy_active": True, "policy_number": "POL-123"},
    )
    remaining = r.next_questions(state, n=500)
    for sq in remaining:
        q = sq.question
        assert q.question_field != "policy_active", \
            f"{q.id} targets policy_active but that field is already filled"


@test("TC06 — Rule E: fraud flags escalate fraud questions to P1")
def tc06():
    """
    When fraud_flags contains a True value, all fraud_consistency questions
    that survive Stage 1 must have effective_priority = 1.
    """
    state = state_with(
        turn_count=5,
        extracted=["category"],
        incident_core={"category": "collision"},
        fraud_flags={"claim_timeline_inconsistency": True},
    )
    top30 = r.next_questions(state, n=30)
    fraud_qs = [sq for sq in top30 if sq.question.category == "fraud_consistency"]

    assert len(fraud_qs) > 0, "No fraud questions in top 30 — bank may have loading issue"
    for sq in fraud_qs:
        assert sq.effective_priority == 1, \
            f"Fraud question {sq.question.id} should be P1 when fraud flag active, got P{sq.effective_priority}"


@test("TC07 — Collision branch: collision-specific questions activate on category=collision")
def tc07():
    """
    Setting category=collision must unlock collision_dynamics questions that
    have incident_type=['collision'] triggers.
    """
    state_no_cat = state_with(turn_count=2)
    state_collision = state_with(
        turn_count=2,
        extracted=["category"],
        incident_core={"category": "collision"},
    )
    ids_no_cat   = {sq.question.id for sq in r.next_questions(state_no_cat, n=500)}
    ids_collision = {sq.question.id for sq in r.next_questions(state_collision, n=500)}

    # Collision-specific questions should only appear after category is set
    col_only = ids_collision - ids_no_cat
    col_category_qs = [
        sq for sq in r.next_questions(state_collision, n=500)
        if sq.question.category == "collision_dynamics"
        and sq.question.id in col_only
    ]
    assert len(col_category_qs) > 0, \
        "No collision_dynamics questions unlocked after setting category=collision"


@test("TC08 — Theft branch: theft-specific questions gate on category=theft")
def tc08():
    """
    Theft-specific questions (theft_specific category) must NOT appear
    for a collision claim.
    """
    state_collision = state_with(
        turn_count=3,
        extracted=["category"],
        incident_core={"category": "collision"},
    )
    all_ids = {sq.question.id for sq in r.next_questions(state_collision, n=500)}
    theft_qs = [sq for sq in r.next_questions(state_collision, n=500)
                if sq.question.category == "theft_specific"]
    assert len(theft_qs) == 0, \
        f"Theft questions should not appear for collision claim, got: {[sq.question.id for sq in theft_qs[:3]]}"


@test("TC09 — Hit-and-run branch: follow-up questions activate on hit_and_run=True")
def tc09():
    """
    After hit_and_run=True, questions that require hit_and_run as a trigger
    (e.g. plate-noted, hit-and-run FIR) must appear in results.
    """
    state_before = state_with(
        turn_count=4,
        extracted=["category", "third_party_involved"],
        incident_core={"category": "collision"},
        third_party={"third_party_involved": True},
    )
    state_after = state_with(
        turn_count=5,
        extracted=["category", "third_party_involved", "hit_and_run"],
        incident_core={"category": "collision"},
        third_party={"third_party_involved": True, "hit_and_run": True},
        legal_and_reporting={"hit_and_run": True},
    )
    ids_before = {sq.question.id for sq in r.next_questions(state_before, n=500)}
    ids_after  = {sq.question.id for sq in r.next_questions(state_after, n=500)}

    # QTHI0025 = "Did you manage to note the registration plate of the fleeing vehicle?"
    # It has trigger hit_and_run: True
    hit_run_qs = [
        sq for sq in r.next_questions(state_after, n=500)
        if "hit_and_run" in sq.question.triggers
    ]
    assert len(hit_run_qs) > 0, \
        "No hit-and-run follow-up questions activated after hit_and_run=True"


@test("TC10 — Third-party questions suppressed when third_party_involved=False")
def tc10():
    """
    If third_party_involved is False, questions with trigger
    third_party_involved=True must be filtered out.
    """
    state = state_with(
        turn_count=4,
        extracted=["category", "third_party_involved"],
        incident_core={"category": "collision"},
        third_party={"third_party_involved": False},
    )
    remaining = r.next_questions(state, n=500)
    for sq in remaining:
        trig = sq.question.triggers
        if trig.get("third_party_involved") is True:
            raise AssertionError(
                f"{sq.question.id} has trigger third_party_involved=True "
                "but should be suppressed when third_party_involved=False"
            )


# TERMKINATION
@test("TC11 — C4: safety cap terminates at turn 26")
def tc11():
    state = state_with(turn_count=26)
    d = should_terminate(state, [], r)
    assert d.should_stop, "Should terminate at turn 26"
    assert_eq(d.condition_code, "C4", "Condition code")


@test("TC12 — C5: inactive policy terminates immediately")
def tc12():
    state = state_with(
        turn_count=5,
        extracted=["policy_active"],
        policy={"policy_active": False},
    )
    d = should_terminate(state, [], r)
    assert d.should_stop, "Should terminate when policy_active=False"
    assert_eq(d.condition_code, "C5", "Condition code")
    assert "inactive" in d.reason.lower(), "Reason should mention inactive policy"


@test("TC13 — C2: all required fields filled triggers termination")
def tc13():
    state = full_required_state()
    d = should_terminate(state, [], r)
    assert d.should_stop, "Should terminate when all required fields filled"
    assert_eq(d.condition_code, "C2", "Condition code")
    assert_eq(d.diagnostics["required_missing"], [], "No missing required fields")


@test("TC14 — CONTINUE: partial state with P1/P2 remaining does not terminate")
def tc14():
    """Mid-session: category known, policy known, but most required fields still missing."""
    state = state_with(
        turn_count=4,
        extracted=["category", "policy_active"],
        incident_core={"category": "collision"},
        policy={"policy_active": True},
    )
    d = should_terminate(state, [], r)
    assert not d.should_stop, \
        f"Should NOT terminate mid-session, got: code={d.condition_code} reason={d.reason}"
    assert_eq(d.condition_code, "CONTINUE", "Should be CONTINUE")
    assert d.diagnostics["p1p2_remaining"] > 0, "P1/P2 questions should remain"


@test("TC15 — Minimum turn guard: never terminate before turn 3")
def tc15():
    """
    Even with policy_active=False (C5 condition), the minimum turn guard
    should prevent termination at turn 0 and 1.
    Actually, minimum guard only applies to non-blocking conditions.
    C4 and C5 are hard stops — let's verify minimum guard works for
    the non-critical path.
    """
    # Turn 0 — fresh session, nothing answered
    state_t0 = state_with(turn_count=0)
    d = should_terminate(state_t0, [], r)
    assert not d.should_stop, "Should not terminate at turn 0"

    # Turn 1 — one question answered but required fields still missing
    state_t1 = state_with(
        turn_count=1,
        extracted=["category"],
        incident_core={"category": "flood"},
    )
    d = should_terminate(state_t1, [], r)
    assert not d.should_stop, "Should not terminate at turn 1"

    # Turn 2 — still very early
    state_t2 = state_with(
        turn_count=2,
        extracted=["category", "policy_active"],
        incident_core={"category": "flood"},
        policy={"policy_active": True},
    )
    d = should_terminate(state_t2, [], r)
    assert not d.should_stop, "Should not terminate at turn 2"


def print_results():
    passed = sum(1 for r in _results if r.passed)
    failed = sum(1 for r in _results if not r.passed)

    print("\n" + "="*65)
    print("  TASK 6 — TEST RESULTS")
    print("="*65)

    for res in _results:
        icon = "✓" if res.passed else "✗"
        status = "PASS" if res.passed else "FAIL"
        print(f"  {icon} [{status}]  {res.name}")
        if not res.passed:
            print(f"           → {res.message}")
            if res.detail:
                for line in res.detail.strip().split("\n")[-5:]:
                    print(f"             {line}")

    print()
    print(f"  Results: {passed}/{len(_results)} passed  |  {failed} failed")
    print("="*65 + "\n")
    return failed == 0


if __name__ == "__main__":
    ok = print_results()
    sys.exit(0 if ok else 1)