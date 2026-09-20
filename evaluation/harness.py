"""
Evaluation harness (B-05). Two distinct parts, deliberately not merged --
see the conversation/report for why these two files aren't interchangeable:

PART A -- classification, routing, retrieval accuracy
    Run against validation_tickets.json (80 tickets, VAL-####). This is the
    true held-out gate set: it has labels (intent, urgency, expected_route,
    expected_doc_ids) but no reference answer text, so it can grade
    "did the pipeline route/classify/retrieve correctly" but not
    "was the generated answer good."

PART B -- generation quality (must_mention / must_not_claim)
    Run against ground_truth_responses.json (200 tickets, DEV-####, joined
    against development_tickets.json for the actual ticket text). These are
    DEV tickets you've already analyzed -- NOT a blind set -- so use this to
    calibrate/debug generate.py, not as your final reported number. The
    must_not_claim check in particular is a direct test against the
    "confidently incorrect answer" risk from the Section 5 risk register.

Both parts degrade gracefully: if classify()/route()/generate() still raise
NotImplementedError (the scaffold's default), the harness reports that
plainly instead of crashing, so it's usable today and gets more informative
as each module is built.

Run from your project root: python evaluation/harness.py
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.classify import classify
from src.route import route
from src.generate import generate
from src.retrieve import retrieve


def ticket_text(t):
    return f"{t['subject']}\n{t['body']}".strip() if t.get("subject") else t["body"]


# ---------------------------------------------------------------------------
# PART A: classification / routing / retrieval, against the true holdout
# ---------------------------------------------------------------------------

def load_validation_tickets(path="data/validation_tickets.json"):
    with open(path) as f:
        return json.load(f)


def evaluate_pipeline_accuracy(tickets):
    n = len(tickets)
    intent_correct = 0
    urgency_correct = 0
    route_correct = 0
    retrieval_hit_at_k = 0
    n_with_expected_docs = 0

    classify_not_implemented = False
    route_not_implemented = False

    for t in tickets:
        text = ticket_text(t)
        labels = t["labels"]

        # -- classification --
        if not classify_not_implemented:
            try:
                c = classify(text)
                if c.intent == labels["intent"]:
                    intent_correct += 1
                if c.urgency == labels["urgency"]:
                    urgency_correct += 1
            except NotImplementedError:
                classify_not_implemented = True

        # -- retrieval (already built) --
        if labels.get("expected_doc_ids"):
            n_with_expected_docs += 1
            results = retrieve(text)
            retrieved_ids = [r["doc_id"] for r in results]
            if any(doc_id in labels["expected_doc_ids"] for doc_id in retrieved_ids):
                retrieval_hit_at_k += 1

        # -- routing --
        if not route_not_implemented:
            try:
                # route() needs classification + retrieval + threshold; since
                # classify may not exist yet, this will raise the same way.
                c = classify(text)
                passages = retrieve(text)
                r = route(c, passages, threshold=0.80)
                if r.action == labels["expected_route"]:
                    route_correct += 1
            except NotImplementedError:
                route_not_implemented = True

    print("=" * 60)
    print(f"PART A: pipeline accuracy on {n} held-out validation tickets")
    print("=" * 60)

    if classify_not_implemented:
        print("classify(): NOT YET IMPLEMENTED -- intent/urgency accuracy unavailable")
    else:
        print(f"intent accuracy:   {intent_correct}/{n} = {intent_correct/n*100:.1f}%")
        print(f"urgency accuracy:  {urgency_correct}/{n} = {urgency_correct/n*100:.1f}%")

    if route_not_implemented:
        print("route(): NOT YET IMPLEMENTED -- routing accuracy unavailable")
    else:
        print(f"routing accuracy:  {route_correct}/{n} = {route_correct/n*100:.1f}%")

    print(f"retrieval recall@k: {retrieval_hit_at_k}/{n_with_expected_docs} = "
          f"{retrieval_hit_at_k/n_with_expected_docs*100:.1f}% "
          f"(of {n_with_expected_docs} tickets with a known expected doc)")


# ---------------------------------------------------------------------------
# PART B: generation quality, against must_mention / must_not_claim
# ---------------------------------------------------------------------------

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
        if ticket is None:
            continue  # shouldn't happen -- all 200 gt ids are dev ids
        joined.append({**g, "ticket": ticket})
    return joined


def evaluate_generation_quality(entries):
    n = len(entries)
    not_implemented = False
    mentions_ok = 0
    claims_ok = 0
    n_checked = 0
    failures = []

    for e in entries:
        text = ticket_text(e["ticket"])
        try:
            answer = generate(text, retrieved_passages=retrieve(text), prompt_version="v1")
        except NotImplementedError:
            not_implemented = True
            break

        n_checked += 1
        draft = answer.text.lower()

        missing = [m for m in e["must_mention"] if m.lower() not in draft]
        present_forbidden = [c for c in e["must_not_claim"] if c.lower() in draft]

        if not missing:
            mentions_ok += 1
        if not present_forbidden:
            claims_ok += 1
        if missing or present_forbidden:
            failures.append({
                "ticket_id": e["ticket_id"],
                "missing_required": missing,
                "forbidden_present": present_forbidden,
            })

    print("\n" + "=" * 60)
    print(f"PART B: generation quality on {n} dev-ticket ground-truth examples")
    print("(NOTE: these are DEV tickets, already analyzed -- use for")
    print(" calibration, not as your final reported holdout number)")
    print("=" * 60)

    if not_implemented:
        print("generate(): NOT YET IMPLEMENTED -- run this again once it's built")
        return

    print(f"required content present: {mentions_ok}/{n_checked} = {mentions_ok/n_checked*100:.1f}%")
    print(f"forbidden claims absent:  {claims_ok}/{n_checked} = {claims_ok/n_checked*100:.1f}%")
    print(f"(forbidden-claims-absent is the direct test against the 'confidently")
    print(f" incorrect answer' risk from your Section 5 risk register)")

    if failures:
        print(f"\n{len(failures)} failures -- first 5:")
        for f in failures[:5]:
            print(f"  {f['ticket_id']}: missing={f['missing_required']}, "
                  f"forbidden_present={f['forbidden_present']}")


if __name__ == "__main__":
    validation_tickets = load_validation_tickets()
    evaluate_pipeline_accuracy(validation_tickets)

    gt_entries = load_ground_truth_with_ticket_text()
    evaluate_generation_quality(gt_entries)