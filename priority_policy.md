# Priority Policy — Vehicle Insurance Claim Intake

## Overview

Questions are assigned a priority integer from 1 (highest) to 5 (lowest).
The retrieval engine uses priority as one of three ranking signals.
This document defines **what earns each priority level** and **why**.

---

## Priority 1 — Safety, Life, and Legal Blockers

**Rule:** Ask these before anything else, regardless of incident type.

**Criteria:**
- Involves personal injury or fatality (claimant, driver, third party, pedestrian)
- Legal deadline risk (e.g. FIR must be filed within 24 hrs for theft/hit-and-run)
- Policy validity — if policy is lapsed, no downstream questions matter
- Driver was under the influence — this is a coverage exclusion that invalidates the claim

**Examples:**
- "Were there any injuries to you or others involved?"
- "Is your policy currently active?"
- "Was the driver at the time of the incident under the influence of alcohol or drugs?"
- "Was a police report or FIR filed?" (for theft, hit-and-run, fire)

---

## Priority 2 — Core Claim Eligibility and Fraud Consistency

**Rule:** Ask early in the interview, before detailed damage questions.

**Criteria:**
- Owner-driver relationship (unauthorized driver = exclusion)
- Timeline consistency (incident datetime vs. report datetime gap)
- Prior claims on same vehicle in recent period
- Coverage type confirmation (third-party only = no own damage payout)
- Hit-and-run or third-party flags that gate entire question branches

**Examples:**
- "Was the vehicle being driven by the registered owner at the time?"
- "Have you filed any other claims on this vehicle in the past 12 months?"
- "What type of coverage does your policy have?"
- "Was the other vehicle involved identified, or was it a hit-and-run?"

---

## Priority 3 — Incident and Damage Details

**Rule:** The core body of the interview. Activated after eligibility is confirmed.

**Criteria:**
- Incident-specific dynamics (speed, direction of impact, road conditions)
- Damage zone identification and severity
- Drivability and towing requirement
- Third-party vehicle and contact details
- Incident-specific fields (fire source, flood water level, theft entry signs)

**Examples:**
- "Which parts of your vehicle were damaged?"
- "From which direction did the impact occur?"
- "Was your vehicle drivable after the incident, or did it need towing?"
- "What is the estimated severity of the damage?"

---

## Priority 4 — Documentation and Evidence

**Rule:** Ask after incident is well understood.

**Criteria:**
- Availability of photos, videos, dashcam footage
- Documents already in hand (RC book, license, repair estimate)
- Witness details and contact
- CCTV availability at loss location

**Examples:**
- "Do you have photographs of the damage?"
- "Were there any witnesses? Can you share their contact details?"
- "Is the repair estimate from the workshop available?"

---

## Priority 5 — Settlement Preferences and Enrichment

**Rule:** Ask last. These are operational, not eligibility-determining.

**Criteria:**
- Workshop preference (cashless network vs. own garage)
- Settlement mode (cashless vs. reimbursement)
- Inspection scheduling
- Add-on coverage queries (zero-dep, engine protection, etc.)
- Optional vehicle details (color, odometer)

**Examples:**
- "Do you prefer cashless repair at a network garage or reimbursement?"
- "Would you like us to schedule a surveyor inspection?"
- "Does your policy include a zero-depreciation add-on?"

---

## Additional Ordering Rules

### Rule A — Gap Fill First
Among questions at the same priority level, prefer questions that fill
fields explicitly required by other downstream questions' triggers.
(A question that unlocks 3 other questions scores higher than one that doesn't.)

### Rule B — Incident-Specific Before Generic
Among same-priority questions, prefer those whose `incident_type` trigger
matches the current category over generic questions with no incident_type trigger.

### Rule C — Never Ask Answered Fields
If a field appears in `already_extracted_categories` OR
`answered_question_ids`, suppress all questions targeting that field.
This applies even if the user mentioned the fact in passing (not as a direct answer).

### Rule D — Correction Handling
If the user negates or corrects a previously stated fact mid-interview:
- Remove the corrected field from `already_extracted_categories`
- Remove the corresponding question from `answered_question_ids`
- Add the corrected value to state
- Re-evaluate which questions are now eligible

### Rule E — Fraud Red Flags Elevate Priority
If `fraud_flags` fields are set to `true` at any point,
any fraud-consistency question (normally priority 2) gets promoted to priority 1
for that session.

---

## Termination Conditions (used by Task 5)

Stop the interview when ANY of the following is true:
1. All priority-1 and priority-2 questions have been answered
2. All `required: true` fields in the schema are filled
3. No questions remain that pass the Stage 1 hard filter
4. `turn_count` exceeds 25 (safety cap against infinite loops)
