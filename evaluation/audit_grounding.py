"""
Runs the guardrails.py grounding check against all 200 dev-ticket
ground-truth examples, and cross-tabs the result against the mechanical
must_mention/must_not_claim check from harness.py Part B.

The interesting question this answers: of the drafts that PASSED the
mechanical check (all required content present, no forbidden phrases),
how many actually have a hidden grounding problem -- exactly the class of
error DEV-0003 had (citation valid, no forbidden keyword, but a fabricated
number/plan name) that the mechanical check structurally cannot see.

Run from your project root: python evaluation/audit_grounding.py
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.generate import generate
from src.retrieve import retrieve
from src.guardrails import check as run_guardrails


def ticket_text(t):
    return f"{t['subject']}\n{t['body']}".strip() if t.get("subject") else t["body"]


def load_ground_truth_with_ticket_text(
    gt_path="data/ground_truth_responses.json",
    dev_path="data/development_tickets.json",
):
    with open(gt_path) as f:
        ground_truth = json.load(f)
    with open(dev_path) as f:
        dev_tickets = {t["ticket_id"]: t for t in json.load(f)}

    joined = []
    for g in ground_truth:
        ticket = dev_tickets.get(g["ticket_id"])
        if ticket is not None:
            joined.append({**g, "ticket": ticket})
    return joined


if __name__ == "__main__":
    entries = load_ground_truth_with_ticket_text()
    print(f"Auditing {len(entries)} ground-truth examples with the grounding guardrail...")

    mechanical_pass_guardrail_pass = 0
    mechanical_pass_guardrail_block = 0   # <-- the interesting cell
    mechanical_fail_guardrail_pass = 0
    mechanical_fail_guardrail_block = 0

    hidden_problems = []

    for i, e in enumerate(entries):
        text = ticket_text(e["ticket"])
        passages = retrieve(text)
        answer = generate(text, passages)

        if not answer.knows_answer:
            continue

        draft = answer.text.lower()
        missing = [m for m in e["must_mention"] if m.lower() not in draft]
        forbidden_present = [c for c in e["must_not_claim"] if c.lower() in draft]
        mechanical_pass = not missing and not forbidden_present

        guardrail_result = run_guardrails(answer, passages)
        guardrail_pass = not guardrail_result.blocked

        if mechanical_pass and guardrail_pass:
            mechanical_pass_guardrail_pass += 1
        elif mechanical_pass and not guardrail_pass:
            mechanical_pass_guardrail_block += 1
            hidden_problems.append({"ticket_id": e["ticket_id"], "findings": guardrail_result.findings})
        elif not mechanical_pass and guardrail_pass:
            mechanical_fail_guardrail_pass += 1
        else:
            mechanical_fail_guardrail_block += 1

        if (i + 1) % 25 == 0:
            print(f"  ...{i + 1}/{len(entries)} processed")

    print("\n" + "=" * 70)
    print("CROSS-TAB: mechanical check (must_mention/must_not_claim) vs. grounding guardrail")
    print("=" * 70)
    print(f"{'':30}{'guardrail: pass':>18}{'guardrail: BLOCKED':>20}")
    print(f"{'mechanical: pass':<30}{mechanical_pass_guardrail_pass:>18}"
          f"{mechanical_pass_guardrail_block:>20}   <-- hidden problems")
    print(f"{'mechanical: fail':<30}{mechanical_fail_guardrail_pass:>18}"
          f"{mechanical_fail_guardrail_block:>20}")

    print(f"\n{mechanical_pass_guardrail_block} answers passed the mechanical check "
          f"but were blocked by the grounding guardrail --")
    print("these are exactly the DEV-0003-shaped errors: valid citation, no forbidden")
    print("keyword, but a fabricated claim the mechanical check structurally cannot see.")

    if hidden_problems:
        print(f"\nFirst 10 hidden problems:")
        for h in hidden_problems[:10]:
            print(f"  {h['ticket_id']}: {h['findings']}")