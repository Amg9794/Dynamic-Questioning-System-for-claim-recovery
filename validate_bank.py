import json, re
from collections import Counter, defaultdict

REQUIRED_FIELDS = {"id", "text", "question_field", "category", "priority", "triggers", "targets"}
VALID_CATEGORIES = {
    "incident_core", "collision_dynamics", "damage_assessment", "third_party_details",
    "legal_reporting", "policy_eligibility", "fraud_consistency", "repair_settlement",
    "theft_specific", "fire_specific", "flood_specific", "vandalism_specific"
}
VALID_PRIORITIES = {1, 2, 3, 4, 5}

def validate(path):
    questions = []
    errors = []
    warnings = []

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                q = json.loads(line)
                questions.append((i, q))
            except json.JSONDecodeError as e:
                errors.append(f"Line {i}: JSON parse error — {e}")

    ids_seen = {}
    texts_seen = {}

    for lineno, q in questions:
        qid = q.get("id", f"MISSING_ID@line{lineno}")
        missing = REQUIRED_FIELDS - set(q.keys())
        if missing:
            errors.append(f"{qid}: Missing required fields: {missing}")

        if not re.match(r"^Q[A-Z]{3}\d{4}$", qid):
            errors.append(f"{qid}: ID format invalid (expected QXXX0000)")

        if qid in ids_seen:
            errors.append(f"{qid}: Duplicate ID (also at line {ids_seen[qid]})")
        ids_seen[qid] = lineno

        cat = q.get("category", "")
        if cat not in VALID_CATEGORIES:
            errors.append(f"{qid}: Unknown category '{cat}'")

        pri = q.get("priority")
        if pri not in VALID_PRIORITIES:
            errors.append(f"{qid}: Invalid priority {pri!r} (must be 1-5)")

        text = q.get("text", "")
        if not text or len(text) < 10:
            errors.append(f"{qid}: Text too short or empty")

        if text in texts_seen:
            warnings.append(f"{qid}: Duplicate text (same as {texts_seen[text]})")
        texts_seen[text] = qid

        if not q.get("question_field"):
            errors.append(f"{qid}: question_field is empty")

        triggers = q.get("triggers")
        if not isinstance(triggers, dict):
            errors.append(f"{qid}: triggers must be a dict, got {type(triggers)}")

        targets = q.get("targets", {})
        ff = targets.get("fill_fields")
        if not isinstance(ff, list) or len(ff) == 0:
            errors.append(f"{qid}: targets.fill_fields must be a non-empty list")

    # Summary stats
    all_qs = [q for _, q in questions]
    cat_counts = Counter(q["category"] for q in all_qs)
    pri_counts = Counter(q["priority"] for q in all_qs)
    trigger_key_counts = Counter()
    for q in all_qs:
        for k in q.get("triggers", {}).keys():
            trigger_key_counts[k] += 1

    print(f"\n{'='*60}")
    print(f"  QUESTION BANK VALIDATION REPORT")
    print(f"{'='*60}")
    print(f"  Total questions loaded : {len(questions)}")
    print(f"  Errors found           : {len(errors)}")
    print(f"  Warnings found         : {len(warnings)}")
    print(f"{'='*60}")

    print(f"\n  CATEGORY BREAKDOWN")
    for cat in sorted(cat_counts):
        bar = "█" * (cat_counts[cat] // 5)
        print(f"  {cat:<32} {cat_counts[cat]:>4}  {bar}")

    print(f"\n  PRIORITY DISTRIBUTION")
    for p in sorted(pri_counts):
        bar = "█" * (pri_counts[p] // 20)
        print(f"  Priority {p}: {pri_counts[p]:>4} questions  {bar}")

    print(f"\n  TOP TRIGGER KEYS USED")
    for k, cnt in trigger_key_counts.most_common(15):
        print(f"  {k:<40} used in {cnt:>4} questions")

    if errors:
        print(f"\n  ERRORS ({len(errors)})")
        for e in errors[:30]:
            print(f"   {e}")
        if len(errors) > 30:
            print(f"  ... and {len(errors)-30} more")
    else:
        print(f"\n No errors found")

    if warnings:
        print(f"\n  WARNINGS ({len(warnings)})")
        for w in warnings[:10]:
            print(f"  ⚠ {w}")
        if len(warnings) > 10:
            print(f"  ... and {len(warnings)-10} more")

    print(f"\n{'='*60}")
    status = "PASS" if not errors else "FAIL"
    print(f"  RESULT: {status}")
    print(f"{'='*60}\n")
    return len(errors) == 0

if __name__ == "__main__":
    import os
    path = os.path.join(os.path.dirname(__file__), "question_bank_raw.jsonl")
    validate(path)