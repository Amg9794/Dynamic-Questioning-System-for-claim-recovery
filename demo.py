#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

# ─── PATHS ───
BASE   = Path(__file__).parent
TASK0  = BASE.parent / "ML_assign"
TASK2  = BASE.parent / "ML_assign"
TASK3  = BASE.parent / "ML_assign"

DEFAULT_BANK   = TASK2 / "question_bank_raw.jsonl"
DEFAULT_SCHEMA = TASK0 / "claim_state_schema.json"
DEFAULT_VOCAB  = TASK0 / "domain_vocabulary.json"

# ─── CONSTANTS 
ANTHROPIC_API  = "your api key"
MODEL          = "claude-sonnet-4-6"
MAX_TOKENS     = 1024
MAX_TURNS      = 25                # hard cap — termination condition 4
MIN_TURNS_BEFORE_TERMINATION = 3  # never terminate on turn 1 or 2

COLORS = {
    "reset":  "\033[0m",
    "bold":   "\033[1m",
    "blue":   "\033[94m",
    "green":  "\033[92m",
    "yellow": "\033[93m",
    "red":    "\033[91m",
    "gray":   "\033[90m",
    "cyan":   "\033[96m",
}

def c(text: str, color: str) -> str:
    """Wrap text in ANSI color if stdout is a tty."""
    if sys.stdout.isatty():
        return f"{COLORS.get(color, '')}{text}{COLORS['reset']}"
    return text


# ─── FRESH CLAIM STATE 

def fresh_state(session_id: str) -> dict:
    return {
        "incident_core": {
            "category": None,
            "loss_datetime": None,
            "loss_location": {"city": None, "state": None, "road_type": None, "landmark": None},
            "reported_to_insurer_datetime": None,
        },
        "vehicle": {
            "make": None, "model": None, "year": None,
            "registration_number": None, "color": None,
            "fuel_type": None, "odometer_reading": None,
            "vehicle_usage_type": None,
        },
        "driver": {
            "owner_is_driver": None, "driver_name": None,
            "license_number": None, "license_valid": None,
            "relationship_to_owner": None,
            "under_influence": None, "breathalyzer_administered": None,
        },
        "damage_assessment": {
            "drivable": None, "towing_required": None,
            "damage_areas": None, "damage_severity": None,
            "airbags_deployed": None, "glass_damage": None,
            "underbody_damage": None, "estimated_repair_cost": None,
            "pre_existing_damage": None,
        },
        "third_party": {
            "third_party_involved": None,
            "vehicle_id": None, "vehicle_make_model": None,
            "driver_name": None, "contact": None,
            "insurer_known": None, "insurer_name": None,
            "injuries_reported": None, "pedestrian_involved": None,
            "hit_and_run": None,
        },
        "legal_and_reporting": {
            "police_report_filed": None, "fir_number": None,
            "police_station_name": None, "witness_available": None,
            "witness_contact": None, "cctv_available": None,
            "dashcam_footage_available": None, "legal_notice_received": None,
        },
        "evidence": {
            "photos_available": None, "video_available": None,
            "repair_estimate_available": None, "documents_submitted": [],
        },
        "policy": {
            "policy_number": None, "policy_active": None,
            "coverage_type": None, "addons": [],
            "ncb_percentage": None, "previous_claims_count": None,
            "policy_expiry_date": None,
        },
        "repair_and_settlement": {
            "workshop_preference": None, "settlement_preference": None,
            "inspection_scheduled": None, "surveyor_assigned": None,
        },
        "incident_specific": {
            "theft": {}, "fire": {}, "flood": {}, "vandalism": {}
        },
        "fraud_flags": {
            "claim_timeline_inconsistency": None,
            "driver_details_inconsistent": None,
            "damage_incident_inconsistency": None,
            "repeated_claim_on_same_vehicle": None,
        },
        "session_control": {
            "session_id": session_id,
            "turn_count": 0,
            "already_extracted_categories": [],
            "answered_question_ids": [],
            "terminated": False,
            "termination_reason": None,
        },
    }


# ─── AUDIT LOG 

class AuditLog:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.entries: list[dict] = []

    def record(self, turn: int, question_id: str, question_text: str,
               user_response: str, state_patch: dict, latency_ms: float):
        self.entries.append({
            "turn": turn,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "question_id": question_id,
            "question_text": question_text,
            "user_response": user_response,
            "state_patch": state_patch,
            "nlu_latency_ms": round(latency_ms, 1),
        })

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "total_turns": len(self.entries),
            "entries": self.entries,
        }


# ─── NLU EXTRACTOR 

class NLUExtractor:
    """
    Calls Claude API to parse a user's free-text response into a structured
    state patch (flat JSON dict of field→value pairs).

    Design principles:
    - Schema and vocabulary injected into every call — Claude knows valid enums
    - Only fields that are explicitly mentioned are returned (no hallucination)
    - Correction signals (negation, "actually", "wait") trigger the correction key
    - Returns {} on API failure (graceful degradation — session continues)
    """

    def __init__(self, schema: dict, vocab: dict, api_key: str):
        self.schema   = schema
        self.vocab    = vocab
        self.api_key  = api_key
        self._system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        # Inject only the most relevant schema parts to keep prompt compact
        incident_types = self.vocab.get("incident_types", {})
        coverage_flags = self.vocab.get("coverage_flags", {})
        damage_areas   = self.vocab.get("damage_areas", {})

        return f"""You are an NLU engine for a vehicle insurance claim intake system.
Your job: parse the user's response to an insurance claim question and return a JSON object
containing ONLY the fields that can be confidently extracted from the response.

RULES:
1. Return ONLY a JSON object. No explanation, no markdown, no prose.
2. Only include fields the user explicitly mentioned — never guess or infer absent fields.
3. Field values must match the types below. Booleans: true/false. Strings: exact enum values.
4. If the user corrects a previous answer, include "_correction": {{"field": "<field_name>", "old_value": "<old>", "new_value": "<new>"}}.
5. If the user is confused, asking a question, or saying nothing useful, return {{}}.
6. If the user mentions facts not in the schema, ignore them.

KNOWN FIELD TYPES AND ENUMS:

incident category (field: "category"):
  enum: {list(incident_types.keys()) if isinstance(incident_types, dict) else incident_types}

coverage_type enum: ["comprehensive", "third_party_only", "own_damage_only"]

fuel_type enum: ["petrol", "diesel", "cng", "electric", "hybrid", "lpg"]

road_type enum: ["highway", "urban", "rural", "parking_lot", "private_property", "unknown"]

damage_severity enum: ["minor", "moderate", "severe", "total_loss"]

settlement_preference enum: ["cashless", "reimbursement"]

Boolean fields (respond true/false):
  policy_active, third_party_involved, hit_and_run, police_report_filed,
  drivable, towing_required, airbags_deployed, glass_damage, underbody_damage,
  witness_available, cctv_available, dashcam_footage_available, owner_is_driver,
  license_valid, under_influence, breathalyzer_administered, photos_available,
  video_available, repair_estimate_available, legal_notice_received,
  pre_existing_damage, pedestrian_involved, injuries_reported,
  insurer_known, inspection_scheduled

String fields (free text):
  vehicle_make, vehicle_model, vehicle_year, registration_number, vehicle_color,
  driver_name, license_number, policy_number, fir_number, police_station_name,
  loss_datetime, loss_location_city, loss_location_state, loss_location_road_type,
  loss_location_landmark, estimated_repair_cost, ncb_percentage,
  previous_claims_count, witness_contact, insurer_name, vehicle_id,
  third_party_driver_name, third_party_contact, workshop_preference

EXAMPLES:
User: "Yes it was a collision, my Maruti Swift hit the divider on the highway"
→ {{"category": "collision", "vehicle_make": "Maruti", "vehicle_model": "Swift",
    "loss_location_road_type": "highway"}}

User: "No police report was filed"
→ {{"police_report_filed": false}}

User: "Actually it wasn't a collision, it was a theft"
→ {{"category": "theft", "_correction": {{"field": "category", "old_value": "collision", "new_value": "theft"}}}}

User: "The car is drivable, front bumper is damaged, moderate damage overall"
→ {{"drivable": true, "damage_severity": "moderate"}}

User: "What do you mean?"
→ {{}}
"""

    def extract(self, question_text: str, user_response: str,
                current_state_summary: str) -> tuple[dict, float]:
        """
        Returns (state_patch_dict, latency_ms).
        On API failure returns ({}, latency_ms).
        """
        t0 = time.time()

        user_msg = (
            f"Current question asked: {question_text}\n\n"
            f"User's response: {user_response}\n\n"
            f"Already known: {current_state_summary}"
        )

        try:
            resp = requests.post(
                ANTHROPIC_API,
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": MODEL,
                    "max_tokens": MAX_TOKENS,
                    "system": self._system_prompt,
                    "messages": [{"role": "user", "content": user_msg}],
                },
                timeout=15,
            )
            latency_ms = (time.time() - t0) * 1000

            if resp.status_code != 200:
                return {}, latency_ms

            data = resp.json()
            raw_text = data["content"][0]["text"].strip()

            # Strip markdown fences if model wrapped in ```json
            if raw_text.startswith("```"):
                raw_text = raw_text.split("```")[1]
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]
            raw_text = raw_text.strip()

            patch = json.loads(raw_text)
            if not isinstance(patch, dict):
                return {}, latency_ms
            return patch, latency_ms

        except Exception:
            latency_ms = (time.time() - t0) * 1000
            return {}, latency_ms


# ─── STATE MANAGER ─

class StateManager:
    """
    Merges NLU patch into the claim state.

    Handles:
    - Flat patch keys → mapped into correct nested group
    - Correction events (Rule D)
    - already_extracted_categories maintenance
    - answered_question_ids tracking
    """

    # Map flat field names → nested path in claim state
    FIELD_MAP: dict[str, tuple[str, str]] = {
        "category":                   ("incident_core",       "category"),
        "loss_datetime":              ("incident_core",       "loss_datetime"),
        "loss_location_city":         ("incident_core",       "loss_location_city"),
        "loss_location_state":        ("incident_core",       "loss_location_state"),
        "loss_location_road_type":    ("incident_core",       "loss_location_road_type"),
        "loss_location_landmark":     ("incident_core",       "loss_location_landmark"),
        "vehicle_make":               ("vehicle",             "make"),
        "vehicle_model":              ("vehicle",             "model"),
        "vehicle_year":               ("vehicle",             "year"),
        "registration_number":        ("vehicle",             "registration_number"),
        "vehicle_color":              ("vehicle",             "color"),
        "fuel_type":                  ("vehicle",             "fuel_type"),
        "vehicle_usage_type":         ("vehicle",             "vehicle_usage_type"),
        "owner_is_driver":            ("driver",              "owner_is_driver"),
        "driver_name":                ("driver",              "driver_name"),
        "license_number":             ("driver",              "license_number"),
        "license_valid":              ("driver",              "license_valid"),
        "under_influence":            ("driver",              "under_influence"),
        "breathalyzer_administered":  ("driver",              "breathalyzer_administered"),
        "drivable":                   ("damage_assessment",   "drivable"),
        "towing_required":            ("damage_assessment",   "towing_required"),
        "damage_severity":            ("damage_assessment",   "damage_severity"),
        "airbags_deployed":           ("damage_assessment",   "airbags_deployed"),
        "glass_damage":               ("damage_assessment",   "glass_damage"),
        "underbody_damage":           ("damage_assessment",   "underbody_damage"),
        "estimated_repair_cost":      ("damage_assessment",   "estimated_repair_cost"),
        "pre_existing_damage":        ("damage_assessment",   "pre_existing_damage"),
        "third_party_involved":       ("third_party",         "third_party_involved"),
        "vehicle_id":                 ("third_party",         "vehicle_id"),
        "third_party_driver_name":    ("third_party",         "driver_name"),
        "third_party_contact":        ("third_party",         "contact"),
        "insurer_known":              ("third_party",         "insurer_known"),
        "insurer_name":               ("third_party",         "insurer_name"),
        "injuries_reported":          ("third_party",         "injuries_reported"),
        "pedestrian_involved":        ("third_party",         "pedestrian_involved"),
        "hit_and_run":                ("third_party",         "hit_and_run"),
        "police_report_filed":        ("legal_and_reporting", "police_report_filed"),
        "fir_number":                 ("legal_and_reporting", "fir_number"),
        "police_station_name":        ("legal_and_reporting", "police_station_name"),
        "witness_available":          ("legal_and_reporting", "witness_available"),
        "witness_contact":            ("legal_and_reporting", "witness_contact"),
        "cctv_available":             ("legal_and_reporting", "cctv_available"),
        "dashcam_footage_available":  ("legal_and_reporting", "dashcam_footage_available"),
        "legal_notice_received":      ("legal_and_reporting", "legal_notice_received"),
        "photos_available":           ("evidence",            "photos_available"),
        "video_available":            ("evidence",            "video_available"),
        "repair_estimate_available":  ("evidence",            "repair_estimate_available"),
        "policy_number":              ("policy",              "policy_number"),
        "policy_active":              ("policy",              "policy_active"),
        "coverage_type":              ("policy",              "coverage_type"),
        "ncb_percentage":             ("policy",              "ncb_percentage"),
        "previous_claims_count":      ("policy",              "previous_claims_count"),
        "policy_expiry_date":         ("policy",              "policy_expiry_date"),
        "workshop_preference":        ("repair_and_settlement", "workshop_preference"),
        "settlement_preference":      ("repair_and_settlement", "settlement_preference"),
        "inspection_scheduled":       ("repair_and_settlement", "inspection_scheduled"),
    }

    def apply_patch(self, state: dict, patch: dict,
                    question: Any, answered_fields: list[str]) -> tuple[dict, list[str]]:
        """
        Merges patch into state.
        Returns (updated_state, list_of_new_fields_extracted).
        """
        state = deepcopy(state)
        new_fields: list[str] = []

        # Handle correction (Rule D)
        correction = patch.pop("_correction", None)
        if correction and isinstance(correction, dict):
            corrected_field = correction.get("field", "")
            # Remove from already_extracted_categories so retriever re-asks if needed
            extracted = state["session_control"]["already_extracted_categories"]
            if corrected_field in extracted:
                extracted.remove(corrected_field)
            # Remove question that originally answered this field
            state["session_control"]["answered_question_ids"] = [
                qid for qid in state["session_control"]["answered_question_ids"]
                if qid != question.id
            ]

        # Apply regular fields
        for flat_key, value in patch.items():
            if flat_key.startswith("_"):
                continue

            mapping = self.FIELD_MAP.get(flat_key)
            if mapping:
                group, field_name = mapping
                if group in state:
                    # Handle nested loss_location
                    if group == "incident_core" and field_name.startswith("loss_location_"):
                        sub_key = field_name.replace("loss_location_", "")
                        if isinstance(state["incident_core"].get("loss_location"), dict):
                            state["incident_core"]["loss_location"][sub_key] = value
                    else:
                        state[group][field_name] = value
                    new_fields.append(flat_key)
            else:
                # Unknown field — store in a catch-all to not lose it
                state.setdefault("_extra", {})[flat_key] = value
                new_fields.append(flat_key)

        # Update session_control
        sc = state["session_control"]

        # Mark question as answered
        if question and question.id not in sc["answered_question_ids"]:
            sc["answered_question_ids"].append(question.id)

        # Add all extracted fields to already_extracted_categories
        for f in new_fields + answered_fields:
            if f not in sc["already_extracted_categories"]:
                sc["already_extracted_categories"].append(f)

        # Also mark question's own target field as extracted
        if question:
            qf = question.question_field
            if qf not in sc["already_extracted_categories"]:
                sc["already_extracted_categories"].append(qf)

        sc["turn_count"] += 1
        return state, new_fields


# ─── TERMINATION CHECKER ───

REQUIRED_FIELDS = {
    "category", "loss_datetime", "vehicle_make", "vehicle_model",
    "vehicle_year", "registration_number", "drivable",
    "third_party_involved", "policy_number", "policy_active",
}


def check_termination(state: dict, retriever, turn: int) -> tuple[bool, str]:
    """
    Returns (should_terminate, reason_string).

    Conditions (from priority_policy.md):
    1. All P1 + P2 questions answered for the current state
    2. All required fields filled
    3. No questions remain after Stage 1 filter
    4. turn_count > MAX_TURNS
    """
    from retriever import flatten_state

    if turn < MIN_TURNS_BEFORE_TERMINATION:
        return False, ""

    # Condition 4: safety cap
    if turn >= MAX_TURNS:
        return True, f"Safety cap reached ({MAX_TURNS} turns)"

    flat = flatten_state(state)

    # Condition 3: no questions pass filter
    stats = retriever.filter_stats(state)
    if stats["stage1_pass"] == 0:
        return True, "No remaining questions match current claim state"

    # Condition 2: all required fields filled
    missing_required = []
    for field in REQUIRED_FIELDS:
        val = flat.get(field)
        if val is None or val == "__NA__":
            missing_required.append(field)

    if not missing_required:
        # Also check no P1 or P2 questions remain
        remaining = retriever.next_questions(state, n=50)
        p1_p2_remaining = [sq for sq in remaining if sq.effective_priority <= 2]
        if not p1_p2_remaining:
            return True, "All required fields collected and all P1/P2 questions answered"

    # Condition 1: no P1/P2 questions remain (even if some required fields missing)
    remaining = retriever.next_questions(state, n=50)
    p1_p2_remaining = [sq for sq in remaining if sq.effective_priority <= 2]
    if not p1_p2_remaining and turn >= 8:
        return True, "All Priority 1 and Priority 2 questions answered"

    return False, ""


# ─── CLAIM SUMMARY ─

def build_summary(state: dict) -> dict:
    """Extract filled fields into a clean structured summary."""
    def not_null(v):
        return v is not None and v != "__NA__" and v != [] and v != {}

    ic   = state.get("incident_core", {})
    veh  = state.get("vehicle", {})
    drv  = state.get("driver", {})
    dam  = state.get("damage_assessment", {})
    tp   = state.get("third_party", {})
    leg  = state.get("legal_and_reporting", {})
    ev   = state.get("evidence", {})
    pol  = state.get("policy", {})
    rep  = state.get("repair_and_settlement", {})
    sc   = state.get("session_control", {})

    summary = {}

    # Incident
    inc = {k: v for k, v in {
        "category":       ic.get("category"),
        "loss_datetime":  ic.get("loss_datetime"),
        "loss_location":  {k2: v2 for k2, v2 in (ic.get("loss_location") or {}).items() if not_null(v2)},
    }.items() if not_null(v)}
    if inc: summary["incident"] = inc

    # Vehicle
    v_d = {k: v for k, v in veh.items() if not_null(v)}
    if v_d: summary["vehicle"] = v_d

    # Driver
    d_d = {k: v for k, v in drv.items() if not_null(v)}
    if d_d: summary["driver"] = d_d

    # Damage
    dm_d = {k: v for k, v in dam.items() if not_null(v)}
    if dm_d: summary["damage"] = dm_d

    # Third party
    tp_d = {k: v for k, v in tp.items() if not_null(v)}
    if tp_d: summary["third_party"] = tp_d

    # Legal
    lg_d = {k: v for k, v in leg.items() if not_null(v)}
    if lg_d: summary["legal_and_reporting"] = lg_d

    # Evidence
    ev_d = {k: v for k, v in ev.items() if not_null(v)}
    if ev_d: summary["evidence"] = ev_d

    # Policy
    po_d = {k: v for k, v in pol.items() if not_null(v)}
    if po_d: summary["policy"] = po_d

    # Repair
    rp_d = {k: v for k, v in rep.items() if not_null(v)}
    if rp_d: summary["repair_and_settlement"] = rp_d

    # Meta
    summary["_session"] = {
        "session_id":    sc.get("session_id"),
        "total_turns":   sc.get("turn_count"),
        "fields_captured": len(sc.get("already_extracted_categories", [])),
        "termination_reason": sc.get("termination_reason"),
    }

    return summary


# ─── CLI RENDERING ─

def print_banner():
    print()
    print(c("╔══════════════════════════════════════════════════════════╗", "blue"))
    print(c("║     Vehicle Insurance Claim Intake —  Demo               ║", "blue"))
    print(c("║                                                          ║", "blue"))
    print(c("╚══════════════════════════════════════════════════════════╝", "blue"))
    print()

def print_question(turn: int, total_remaining: int, q: Any, sq: Any):
    print()
    print(c(f"  Turn {turn}/{MAX_TURNS}  ·  {total_remaining} questions remaining  ·  "
            f"[{sq.reason}]", "gray"))
    print(c(f"  {q.id}  [{q.category}]", "gray"))
    print()
    print(c(f"  ❓  {q.text}", "bold"))
    print()

def print_extracted(patch: dict, latency_ms: float):
    if not patch:
        print(c(f"  (No fields extracted from response  ·  {latency_ms:.0f}ms)", "gray"))
        return
    clean = {k: v for k, v in patch.items() if not k.startswith("_")}
    correction = patch.get("_correction")
    print(c(f"  ✓  Extracted {len(clean)} field(s) in {latency_ms:.0f}ms:", "green"), end=" ")
    print(c(", ".join(f"{k}={repr(v)}" for k, v in clean.items()), "cyan"))
    if correction:
        print(c(f"  ↩  Correction: {correction}", "yellow"))

def print_summary(summary: dict, audit: AuditLog):
    print()
    print(c("═" * 62, "blue"))
    print(c("  CLAIM SUMMARY", "bold"))
    print(c("═" * 62, "blue"))
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print()
    print(c("═" * 62, "blue"))
    print(c("  AUDIT LOG", "bold"))
    print(c("═" * 62, "blue"))
    audit_data = audit.to_dict()
    print(json.dumps(audit_data, indent=2, ensure_ascii=False, default=str))
    print(c("═" * 62, "blue"))
    print()

def print_termination(reason: str):
    print()
    print(c("─" * 62, "yellow"))
    print(c(f"  Session complete: {reason}", "yellow"))
    print(c("─" * 62, "yellow"))


# ─── MAIN LOOP ────

def run_session(
    bank_path: Path,
    schema_path: Path,
    vocab_path: Path,
    session_id: str,
    replay_answers: list[str] | None = None,
):
    # Load dependencies
    sys.path.insert(0, str(TASK3))
    from retriever import Retriever

    print(c("  Loading question bank ...", "gray"), end=" ", flush=True)
    retriever = Retriever(bank_path)
    print(c(f"✓  {len(retriever.questions)} questions", "green"))

    with open(schema_path) as f:
        schema = json.load(f)
    with open(vocab_path) as f:
        vocab = json.load(f)

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print(c("  ⚠  ANTHROPIC_API_KEY not set — NLU extraction disabled (mock mode)", "yellow"))

    nlu = NLUExtractor(schema, vocab, api_key) if api_key else None
    state_manager = StateManager()
    audit = AuditLog(session_id)
    state = fresh_state(session_id)

    replay_index = 0
    turn = 0

    print_banner()
    print(c(f"  Session ID: {session_id}", "gray"))
    print(c("  Type your response to each question. Type 'quit' to exit early.", "gray"))

    while True:
        turn += 1

        # ── Termination check 
        should_stop, reason = check_termination(state, retriever, turn - 1)
        if should_stop:
            state["session_control"]["terminated"] = True
            state["session_control"]["termination_reason"] = reason
            print_termination(reason)
            break

        # ── Select next question ──
        stats = retriever.filter_stats(state)
        scored = retriever.next_questions(state, n=5)
        if not scored:
            state["session_control"]["terminated"] = True
            state["session_control"]["termination_reason"] = "No questions remaining"
            print_termination("No questions remaining")
            break

        sq = scored[0]
        q  = sq.question

        print_question(turn, stats["stage1_pass"], q, sq)

        # ── Get user input ───
        if replay_answers is not None:
            if replay_index < len(replay_answers):
                user_response = replay_answers[replay_index]
                replay_index += 1
                print(c(f"  > {user_response}", "cyan"))
            else:
                # Replay exhausted — end session cleanly
                print(c("\n  [Replay complete — ending session]", "yellow"))
                break
        else:
            try:
                user_response = input(c("  > ", "bold")).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                print(c("\n  Session interrupted by user.", "yellow"))
                break

        if user_response.lower() in ("quit", "exit", "q"):
            print(c("\n  Session ended by user.", "yellow"))
            break
        if not user_response:
            print(c("  (skipped — no input)", "gray"))
            # Still mark question as asked so we don't repeat it
            state["session_control"]["answered_question_ids"].append(q.id)
            state["session_control"]["already_extracted_categories"].append(q.question_field)
            state["session_control"]["turn_count"] += 1
            audit.record(turn, q.id, q.text, "", {}, 0)
            continue

        # ── NLU extraction ───
        if nlu:
            # Build a short summary of known facts for context
            flat = {}
            for group in ("incident_core", "vehicle", "policy", "third_party"):
                for k, v in state.get(group, {}).items():
                    if v is not None and v != "__NA__":
                        flat[k] = v
            state_summary = json.dumps(flat, ensure_ascii=False)[:300]

            patch, latency_ms = nlu.extract(q.text, user_response, state_summary)
        else:
            # Mock mode: try simple keyword extraction for demo
            patch, latency_ms = _mock_extract(q, user_response), 0.0

        print_extracted(patch, latency_ms)

        # ── State update 
        new_fields = patch.get("_extra_fields", [])
        state, extracted = state_manager.apply_patch(
            state, patch, q, answered_fields=new_fields
        )

        # Record in audit
        audit.record(turn, q.id, q.text, user_response, patch, latency_ms)

    # ── End of session ──
    summary = build_summary(state)
    print_summary(summary, audit)

    # Save outputs
    out_dir = BASE / f"session_{session_id[:8]}"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "claim_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    with open(out_dir / "audit_log.json", "w") as f:
        json.dump(audit.to_dict(), f, indent=2, ensure_ascii=False, default=str)
    with open(out_dir / "final_state.json", "w") as f:
        json.dump(state, f, indent=2, ensure_ascii=False, default=str)

    print(c(f"  Outputs saved to: {out_dir}/", "green"))
    print()
    return summary, audit


# ─── MOCK EXTRACTOR (when no API key) 

def _mock_extract(question: Any, response: str) -> dict:
    """
    Very simple keyword extractor for demo/testing without an API key.
    Only handles the most common yes/no and category patterns.
    """
    r = response.lower().strip()
    qf = question.question_field

    # Boolean fields
    bool_fields = {
        "policy_active", "third_party_involved", "hit_and_run", "police_report_filed",
        "drivable", "towing_required", "airbags_deployed", "glass_damage",
        "witness_available", "cctv_available", "photos_available",
        "owner_is_driver", "under_influence", "injuries_reported"
    }
    if qf in bool_fields:
        if any(w in r for w in ("yes", "yeah", "correct", "true", "yep", "certainly")):
            return {qf: True}
        if any(w in r for w in ("no", "nope", "not", "didn't", "wasn't", "false")):
            return {qf: False}

    # Category
    if qf == "category":
        for cat in ("collision", "theft", "fire", "flood", "vandalism"):
            if cat in r:
                return {"category": cat}

    # Vehicle make/model basic extraction
    if qf == "vehicle_make":
        for make in ("maruti", "suzuki", "hyundai", "tata", "honda", "toyota",
                      "mahindra", "kia", "mg", "skoda", "volkswagen", "ford"):
            if make in r:
                return {"vehicle_make": make.title()}

    # Damage severity
    if qf == "damage_severity":
        for sev in ("minor", "moderate", "severe", "total loss", "total_loss"):
            if sev in r:
                return {"damage_severity": sev.replace(" ", "_")}

    # Free text fields — return as-is
    free_text = {"fir_number", "policy_number", "registration_number",
                  "driver_name", "police_station_name", "witness_contact"}
    if qf in free_text and len(response) > 2:
        return {qf: response}

    return {}


# ─── REPLAY TEST ──

SAMPLE_REPLAY = [
    "It was a collision, my Hyundai Creta hit the rear of a truck on the highway",
    "Yes, there was another vehicle involved — a truck",
    "No it was not a hit and run, the truck driver stopped",
    "Yes a police report was filed",
    "FIR number is 123/2024 at MG Road police station",
    "The car is not drivable, severe damage to the front",
    "Yes the airbags deployed",
    "My policy number is HDFC-VEH-2024-001 and yes it is active",
    "Comprehensive coverage",
    "Yes there were no injuries, I am fine",
    "Yes I have photos of the damage",
    "I prefer cashless repair at a Hyundai authorised service center",
]


# ─── ENTRY POINT ──

def main():
    parser = argparse.ArgumentParser(description="Insurance Claim Intake CLI Demo")
    parser.add_argument("--bank",       default=str(DEFAULT_BANK),   help="Path to validated question bank JSONL")
    parser.add_argument("--schema",     default=str(DEFAULT_SCHEMA), help="Path to claim state schema JSON")
    parser.add_argument("--vocab",      default=str(DEFAULT_VOCAB),  help="Path to domain vocabulary JSON")
    parser.add_argument("--session-id", default=None,                help="Session ID (auto-generated if not set)")
    parser.add_argument("--replay",     default=None,                help="Path to replay answers file (one per line)")
    parser.add_argument("--demo",       action="store_true",         help="Run built-in demo replay")
    args = parser.parse_args()

    session_id = args.session_id or str(uuid.uuid4())
    replay_answers = None

    if args.demo:
        replay_answers = SAMPLE_REPLAY
    elif args.replay:
        with open(args.replay) as f:
            replay_answers = [line.strip() for line in f if line.strip()]

    run_session(
        bank_path   = Path(args.bank),
        schema_path = Path(args.schema),
        vocab_path  = Path(args.vocab),
        session_id  = session_id,
        replay_answers = replay_answers,
    )


if __name__ == "__main__":
    main()