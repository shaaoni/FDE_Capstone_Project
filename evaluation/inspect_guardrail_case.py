"""
Diagnostic: for one specific ticket, print the ACTUAL generated draft
side by side with what the grounding guardrail flagged. This is to check
whether the LLM judge is doing its real job (finding unsupported claims
IN THE DRAFT) or malfunctioning (echoing back source-passage content and
mislabeling it).

Run from your project root: python evaluation/inspect_guardrail_case.py DEV-0485
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.generate import generate
from src.retrieve import retrieve
from src.guardrails import check as run_guardrails, _check_numeric_grounding, _check_llm_grounding


def ticket_text(t):
    return f"{t['subject']}\n{t['body']}".strip() if t.get("subject") else t["body"]


if __name__ == "__main__":
    target_id = sys.argv[1] if len(sys.argv) > 1 else "DEV-0485"

    tickets = json.load(open("data/development_tickets.json"))
    ticket = next(t for t in tickets if t["ticket_id"] == target_id)
    text = ticket_text(ticket)

    passages = retrieve(text)
    answer = generate(text, passages)

    print("=" * 70)
    print(f"TICKET {target_id}: {text[:200]}")
    print("=" * 70)
    print("\nACTUAL DRAFT TEXT:")
    print(answer.text)
    print(f"\nCitations: {answer.citations}")

    print("\n" + "=" * 70)
    print("CITED PASSAGES (what the guardrail compares against):")
    print("=" * 70)
    for p in passages:
        if p["doc_id"] in answer.citations:
            print(f"\n[{p['doc_id']}]")
            print(p["text"][:500])

    passages_text = "\n\n".join(p["text"] for p in passages if p["doc_id"] in answer.citations)

    print("\n" + "=" * 70)
    print("NUMERIC CHECK RESULT:")
    print("=" * 70)
    numeric_findings = _check_numeric_grounding(answer.text, passages_text)
    print(numeric_findings if numeric_findings else "(nothing flagged)")

    print("\n" + "=" * 70)
    print("LLM JUDGE RESULT:")
    print("=" * 70)
    llm_findings = _check_llm_grounding(answer.text, passages_text)
    print(llm_findings if llm_findings else "(nothing flagged)")

    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)
    for f in llm_findings:
        found_in_draft = f.lower() in answer.text.lower()
        found_in_source = f.lower() in passages_text.lower()
        print(f"Finding: {f[:80]}")
        print(f"  -> appears in DRAFT: {found_in_draft}  |  appears in SOURCE: {found_in_source}")
        if found_in_source and not found_in_draft:
            print("  -> LIKELY JUDGE MALFUNCTION: this text is from the source, not the draft")