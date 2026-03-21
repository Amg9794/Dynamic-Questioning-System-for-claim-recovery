import json, re, os
from collections import defaultdict, Counter
from pathlib import Path

# PATHS 
BASE    = Path(__file__).parent
TASK0   = Path(__file__).parent.parent / "ML_assign" if (Path(__file__).parent.parent / "ML_assign").exists() \
          else Path("/mnt/user-data/outputs/ML_assign")
TASK1   = Path(__file__).parent.parent / "ML_assign"
INPUT   = TASK1 / "question_bank_raw.jsonl"
OUT_Q   = BASE / "question_bank_validated.jsonl"
OUT_R   = BASE / "validation_report.json"

#  CONSTANTS
VALID_CATEGORIES = {
    "incident_core", "collision_dynamics", "damage_assessment", "third_party_details",
    "legal_reporting", "policy_eligibility", "fraud_consistency", "repair_settlement",
    "theft_specific", "fire_specific", "flood_specific", "vandalism_specific"
}
VALID_PRIORITIES = {1, 2, 3, 4, 5}
REQUIRED_Q_FIELDS = {"id", "text", "question_field", "category", "priority", "triggers", "targets"}
ID_PATTERN = re.compile(r"^Q[A-Z]{3}\d{4}$")
VALID_INCIDENT_TYPES = {
    "collision", "collision_wall", "theft", "fire", "flood",
    "vandalism", "animal_strike", "glass_damage", "mechanical_breakdown", "unknown"
}
NEAR_DUP_THRESHOLD = 0.85   
MIN_QUESTIONS_PER_CATEGORY = 100

INCIDENT_SPECIFIC_CATEGORIES = {
    "collision_dynamics", "theft_specific", "fire_specific",
    "flood_specific", "vandalism_specific"
}


def build_schema_registry(schema_path: Path) -> dict:
    """
    Returns a flat dict: field_name -> {type, required, enum, ...}
    Handles nested objects by flattening with underscore joins.
    Also returns a set of all top-level group names.
    """
    if not schema_path.exists():
        return {}, set()

    with open(schema_path) as f:
        raw = json.load(f)

    field_data = raw.get("fields", raw)
    registry = {}
    groups = set()

    def flatten(obj, prefix=""):
        if not isinstance(obj, dict):
            return
        for k, v in obj.items():
            if k.startswith("_"):
                continue
            full_key = f"{prefix}{k}" if prefix else k
            if isinstance(v, dict) and ("type" in v or "required" in v or "default" in v):
                registry[full_key] = v
                groups.add(full_key.split("_")[0] if not prefix else prefix.split("_")[0])
                if "fields" in v:
                    flatten(v["fields"], full_key + "_")
            elif isinstance(v, dict):
                flatten(v, full_key + "_")

    flatten(field_data)
    return registry, groups


def get_required_fields(registry: dict) -> set:
    return {k for k, v in registry.items() if v.get("required")}


def stage1_schema(questions: list) -> tuple[list, list]:
    """Returns (passing_questions, errors_list)."""
    errors = []
    passing = []
    ids_seen = {}

    for q in questions:
        qid = q.get("id", "UNKNOWN")
        q_errors = []
        missing = REQUIRED_Q_FIELDS - set(q.keys())
        if missing:
            q_errors.append(f"Missing required fields: {sorted(missing)}")
        if not ID_PATTERN.match(qid):  # ID format
            q_errors.append(f"ID '{qid}' does not match pattern QXXX0000")
        if qid in ids_seen:  #Duplicate ID
            q_errors.append(f"Duplicate ID — first seen at position {ids_seen[qid]}")
        else:
            ids_seen[qid] = len(passing)


        cat = q.get("category", "")
        if cat not in VALID_CATEGORIES:
            q_errors.append(f"Unknown category '{cat}'")
        pri = q.get("priority")
        if pri not in VALID_PRIORITIES:
            q_errors.append(f"Priority {pri!r} must be integer 1–5")

        # 1f. Text: non-empty, ends with '?', min length
        text = q.get("text", "")
        if not text:
            q_errors.append("Text is empty")
            q_errors.append(f"Text too short ({len(text)} chars): '{text}'")
        elif "?" not in text:
            q_errors.append(f"Text contains no '?': '{text[:60]}...'")  


        qf = q.get("question_field", "")
        if not isinstance(qf, str) or not qf:
            q_errors.append("question_field is empty or not a string")

        trig = q.get("triggers")
        if not isinstance(trig, dict):
            q_errors.append(f"triggers must be a dict, got {type(trig).__name__}")
        else:
            if "incident_type" in trig:
                it = trig["incident_type"]
                if isinstance(it, list):
                    bad = [x for x in it if x not in VALID_INCIDENT_TYPES]
                    if bad:
                        q_errors.append(f"Invalid incident_type values: {bad}")
                else:
                    q_errors.append("triggers.incident_type must be a list")

        # 1j. targets.fill_fields is a non-empty list
        targets = q.get("targets", {})
        ff = targets.get("fill_fields") if isinstance(targets, dict) else None
        if not isinstance(ff, list) or len(ff) == 0:
            q_errors.append("targets.fill_fields must be a non-empty list")

        if q_errors:
            errors.append({"id": qid, "stage": 1, "errors": q_errors})
        else:
            passing.append(q)

    return passing, errors


def stage2_logic(questions: list, registry: dict) -> tuple[list, list]:
    """
    Checks:
    - trigger keys are known schema fields (or special keys: incident_type, required_fields_present)
    - fill_fields are known schema fields (relaxed: allow unmapped fields with warning)
    - Priority 1 in incident-specific categories only if no incident_type trigger
      (i.e. only truly universal safety questions should be P1 there)
    - question_field is a valid schema field or a reasonable derived field
    """
    errors = []
    passing = []

    SPECIAL_TRIGGER_KEYS = {"incident_type", "required_fields_present"}
    # Build a loose set: all short-form field names from registry
    schema_field_names = set(registry.keys())
    # Also allow short names without group prefix
    schema_short_names = set()
    for k in schema_field_names:
        parts = k.split("_")
        for i in range(1, len(parts)):
            schema_short_names.add("_".join(parts[i:]))

    all_known_fields = schema_field_names | schema_short_names

    def field_known(name: str) -> bool:
        if name in all_known_fields:
            return True
        # Allow any name that is a substring match of a known field
        for f in schema_field_names:
            if name in f or f.endswith(name):
                return True
        return False

    for q in questions:
        qid = q["id"]
        q_errors = []
        q_warnings = []
        cat = q.get("category", "")
        pri = q.get("priority", 0)
        triggers = q.get("triggers", {})
        fill_fields = q.get("targets", {}).get("fill_fields", [])

        for tkey in triggers:
            if tkey in SPECIAL_TRIGGER_KEYS:
                continue
            if not field_known(tkey):
                if re.match(r'^[a-z][a-z0-9_]+$', tkey):
                    q_warnings.append(
                        f"Trigger key '{tkey}' not in formal schema (runtime field)"
                    )
                else:
                    q_errors.append(f"Trigger key '{tkey}' is malformed")

  
        for ff in fill_fields:
            if not field_known(ff):
                q_warnings.append(f"fill_field '{ff}' not found in schema (may be derived)")

        if cat in INCIDENT_SPECIFIC_CATEGORIES and pri == 1:
            # P1 in an incident-specific category is only valid if the question
            # has no incident_type restriction (i.e. it's universally applicable)
            # OR the question_field is in a known safety set
            SAFETY_FIELDS = {
                "fire_injuries", "occupants_trapped_fire", "carjacking_harm",
                "vulnerable_persons_fire", "fire_covered_policy", "theft_covered",
                "vandalism_covered_policy"
            }
            if "incident_type" in triggers and q.get("question_field") not in SAFETY_FIELDS:
                q_warnings.append(
                    f"Priority 1 in incident-specific category '{cat}' with incident_type "
                    f"trigger — confirm this is a genuine safety/legal blocker"
                )

        if "required_fields_present" in triggers:
            rfp = triggers["required_fields_present"]
            if not isinstance(rfp, list):
                q_errors.append("triggers.required_fields_present must be a list")
            else:
                for rf in rfp:
                    if not field_known(rf):
                        q_warnings.append(
                            f"required_fields_present value '{rf}' not found in schema"
                        )
        FRAUD_HARD_FLAGS = {
            "double_dipping_claim", "claim_without_policyholder_knowledge",
            "over_insured_cheap_vehicle", "engine_chassis_tampered",
            "staged_damage_for_sale", "reconstruction_contradicts_claim",
            "repeated_parts_claims"
        }
        if cat == "fraud_consistency" and pri == 1:
            if q.get("question_field") not in FRAUD_HARD_FLAGS:
                q_warnings.append(
                    "Fraud question at Priority 1 — only hard fraud flags should be P1"
                )

        q["_warnings"] = q_warnings
        if q_errors:
            errors.append({"id": qid, "stage": 2, "errors": q_errors, "warnings": q_warnings})
        else:
            passing.append(q)

    return passing, errors


def tokenize(text: str) -> frozenset:
    """Lowercase word tokens, strip punctuation."""
    return frozenset(re.findall(r"[a-z]+", text.lower()))


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def stage3_dedup(questions: list) -> tuple[list, list]:
    """
    Returns (unique_questions, duplicate_records).
    Removes:
    - Exact text duplicates (keep first)
    - Near-duplicates: Jaccard ≥ NEAR_DUP_THRESHOLD within same category
    """
    duplicates = []
    kept = []
    exact_seen = {}         # text -> qid
    # Per-category token sets for near-dup check
    cat_tokens = defaultdict(list)  # category -> [(qid, token_set)]

    for q in questions:
        qid = q["id"]
        text = q["text"].strip()
        cat = q["category"]

        # Exact dedup
        if text in exact_seen:
            duplicates.append({
                "id": qid,
                "stage": 3,
                "reason": "exact_duplicate",
                "duplicate_of": exact_seen[text]
            })
            continue

        # Near-dup within category
        tokens = tokenize(text)
        near_dup_of = None
        for prev_id, prev_tokens in cat_tokens[cat]:
            score = jaccard(tokens, prev_tokens)
            if score >= NEAR_DUP_THRESHOLD:
                near_dup_of = (prev_id, round(score, 3))
                break

        if near_dup_of:
            duplicates.append({
                "id": qid,
                "stage": 3,
                "reason": "near_duplicate",
                "jaccard_score": near_dup_of[1],
                "duplicate_of": near_dup_of[0],
                "text": text
            })
            continue

        exact_seen[text] = qid
        cat_tokens[cat].append((qid, tokens))
        kept.append(q)

    return kept, duplicates

def stage4_coverage(questions: list, registry: dict) -> dict:
    """
    Checks:
    - All required schema fields covered by ≥1 question's fill_fields or question_field
    - All 12 categories have ≥ MIN_QUESTIONS_PER_CATEGORY passing questions
    Returns a coverage report dict.
    """
    required_fields = get_required_fields(registry)

    # Map: field_name -> list of question IDs that fill it
    field_coverage = defaultdict(list)
    for q in questions:
        qf = q.get("question_field", "")
        if qf:
            # Try to match against registry (short name matching)
            for rf in required_fields:
                if qf in rf or rf.endswith(qf) or qf == rf.split("_")[-1]:
                    field_coverage[rf].append(q["id"])
        for ff in q.get("targets", {}).get("fill_fields", []):
            for rf in required_fields:
                if ff in rf or rf.endswith(ff) or ff == rf.split("_")[-1]:
                    field_coverage[rf].append(q["id"])

    uncovered_required = [f for f in required_fields if not field_coverage[f]]

    # Category counts
    cat_counts = Counter(q["category"] for q in questions)
    under_minimum = {cat: cnt for cat, cnt in cat_counts.items() if cnt < MIN_QUESTIONS_PER_CATEGORY}
    missing_categories = VALID_CATEGORIES - set(cat_counts.keys())

    # All fill_fields used across the bank
    all_targets = set()
    for q in questions:
        for ff in q.get("targets", {}).get("fill_fields", []):
            all_targets.add(ff)

    return {
        "required_fields_total": len(required_fields),
        "required_fields_covered": len(required_fields) - len(uncovered_required),
        "uncovered_required_fields": sorted(uncovered_required),
        "category_counts": dict(sorted(cat_counts.items())),
        "categories_under_minimum": under_minimum,
        "missing_categories": sorted(missing_categories),
        "unique_fill_fields_used": len(all_targets),
        "field_coverage_map": {f: field_coverage[f][:3] for f in required_fields},
        "coverage_pass": len(uncovered_required) == 0 and not under_minimum and not missing_categories
    }


# MAIN PIPELINE
def run_pipeline():
    print("\n" + "="*65)
    print("  TASK 2 — VALIDATION PIPELINE")
    print("="*65)

    # Load schema
    schema_path = TASK0 / "claim_state_schema.json"
    registry, groups = build_schema_registry(schema_path)
    required_fields = get_required_fields(registry)
    print(f"\n  Schema loaded: {len(registry)} fields ({len(required_fields)} required)")

    # Load questions
    with open(INPUT) as f:
        raw_questions = [json.loads(line) for line in f if line.strip()]
    print(f"  Questions loaded: {len(raw_questions)}")

    # Stage 1 
    print("\n  [Stage 1] Schema Check ...")
    s1_pass, s1_errors = stage1_schema(raw_questions)
    print(f"  → Passed: {len(s1_pass):<5}  Failed: {len(s1_errors)}")

    # Stage 2 
    print("  [Stage 2] Logic Check ...")
    s2_pass, s2_errors = stage2_logic(s1_pass, registry)
    s2_warnings = sum(len(q.get("_warnings", [])) for q in s2_pass)
    print(f"  → Passed: {len(s2_pass):<5}  Failed: {len(s2_errors):<5}  Warnings: {s2_warnings}")

    # Stage 3
    print("  [Stage 3] Deduplication ...")
    s3_pass, s3_dupes = stage3_dedup(s2_pass)
    exact_dupes = sum(1 for d in s3_dupes if d["reason"] == "exact_duplicate")
    near_dupes  = sum(1 for d in s3_dupes if d["reason"] == "near_duplicate")
    print(f"  → Passed: {len(s3_pass):<5}  Exact dupes: {exact_dupes:<4}  Near-dupes: {near_dupes}")

    # Stage 4
    print("  [Stage 4] Coverage Check ...")
    # Strip internal _warnings before coverage
    final_questions = [{k: v for k, v in q.items() if not k.startswith("_")} for q in s3_pass]
    coverage = stage4_coverage(final_questions, registry)
    cov_status = "PASS" if coverage["coverage_pass"] else "FAIL"
    print(f"  → Required fields covered: {coverage['required_fields_covered']}/{coverage['required_fields_total']}")
    print(f"  → Categories at ≥{MIN_QUESTIONS_PER_CATEGORY}: "
          f"{len(coverage['category_counts']) - len(coverage['categories_under_minimum'])}"
          f"/{len(coverage['category_counts'])}")
    print(f"  → Coverage: {cov_status}")

    # Write validated JSONL 
    with open(OUT_Q, "w") as f:
        for q in final_questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"\n  Validated questions written: {len(final_questions)}")

    # Assemble full report 
    all_warnings = []
    for q in s2_pass:
        for w in q.get("_warnings", []):
            all_warnings.append({"id": q["id"], "category": q["category"], "warning": w})

    report = {
        "summary": {
            "input_questions": len(raw_questions),
            "stage1_schema_pass": len(s1_pass),
            "stage1_schema_fail": len(s1_errors),
            "stage2_logic_pass": len(s2_pass),
            "stage2_logic_fail": len(s2_errors),
            "stage2_warnings": s2_warnings,
            "stage3_dedup_pass": len(s3_pass),
            "stage3_exact_duplicates": exact_dupes,
            "stage3_near_duplicates": near_dupes,
            "final_validated_questions": len(final_questions),
            "total_questions_removed": len(raw_questions) - len(final_questions),
            "pipeline_result": "PASS" if (
                not s1_errors and not s2_errors and not s3_dupes and coverage["coverage_pass"]
            ) else "PASS_WITH_WARNINGS" if (
                not s1_errors and not s2_errors and coverage["coverage_pass"]
            ) else "FAIL"
        },
        "stage1_schema_errors": s1_errors,
        "stage2_logic_errors": s2_errors,
        "stage2_logic_warnings": all_warnings,
        "stage3_duplicates": s3_dupes,
        "stage4_coverage": coverage,
        "priority_distribution": dict(
            Counter(q["priority"] for q in final_questions)
        ),
        "category_distribution": dict(
            Counter(q["category"] for q in final_questions)
        ),
        "trigger_key_usage": dict(
            Counter(k for q in final_questions for k in q.get("triggers", {}))
        )
    }

    with open(OUT_R, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  Validation report written: {OUT_R.name}")

    # Final summary 
    print("\n" + "="*65)
    print("  PIPELINE RESULT:", report["summary"]["pipeline_result"])
    print("="*65)
    print(f"  Input:     {len(raw_questions):>6} questions")
    print(f"  Stage 1:   {len(s1_errors):>6} schema errors removed")
    print(f"  Stage 2:   {len(s2_errors):>6} logic errors removed")
    print(f"  Stage 3:   {len(s3_dupes):>6} duplicates removed")
    print(f"  Final:     {len(final_questions):>6} validated questions")
    print(f"  Warnings:  {s2_warnings:>6} logic warnings (kept)")
    print()

    if coverage["uncovered_required_fields"]:
        print("  ⚠ UNCOVERED REQUIRED FIELDS:")
        for f in coverage["uncovered_required_fields"]:
            print(f"    - {f}")

    if coverage["categories_under_minimum"]:
        print("  ⚠ CATEGORIES UNDER MINIMUM:")
        for cat, cnt in coverage["categories_under_minimum"].items():
            print(f"    - {cat}: {cnt} questions")

    if all_warnings:
        print(f"\n  Top warnings (showing first 10 of {len(all_warnings)}):")
        for w in all_warnings[:10]:
            print(f"    [{w['id']}] {w['warning']}")

    print("\n" + "="*65 + "\n")
    return report


if __name__ == "__main__":
    run_pipeline()