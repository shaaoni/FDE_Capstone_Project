"""
Calibrates RETRIEVAL_ANSWERABLE_THRESHOLD (route.py) against real labels,
instead of leaving it as a guess. Compares top retrieval scores for
tickets marked answerable_from_docs=True vs. False, then sweeps candidate
thresholds to see which actually separates the two groups.

Run from your project root: python evaluation/calibrate_retrieval_threshold.py
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieve import retrieve


def ticket_text(t):
    return f"{t['subject']}\n{t['body']}".strip() if t.get("subject") else t["body"]


def load_all_labeled_tickets():
    # Use both validation and dev tickets for calibration -- this is a
    # threshold-tuning exercise, not the final reported gate metric, so
    # using dev tickets here (unlike the harness's Part A) is fine.
    tickets = []
    for path in ["data/validation_tickets.json", "data/development_tickets.json"]:
        with open(path) as f:
            tickets.extend(json.load(f))
    return tickets


def top_score(text):
    results = retrieve(text, k=1)
    return results[0]["score"] if results else None


if __name__ == "__main__":
    tickets = load_all_labeled_tickets()

    answerable_scores = []
    not_answerable_scores = []

    for t in tickets:
        score = top_score(ticket_text(t))
        if score is None:
            continue
        if t["labels"]["answerable_from_docs"]:
            answerable_scores.append(score)
        else:
            not_answerable_scores.append(score)

    def summarize(name, scores):
        scores = sorted(scores)
        n = len(scores)
        print(f"{name}: n={n}, min={scores[0]:.3f}, p25={scores[n//4]:.3f}, "
              f"median={scores[n//2]:.3f}, p75={scores[3*n//4]:.3f}, max={scores[-1]:.3f}")

    print("=" * 60)
    print("Top retrieval score distribution, by true answerability label")
    print("=" * 60)
    summarize("answerable_from_docs=True ", answerable_scores)
    summarize("answerable_from_docs=False", not_answerable_scores)

    print("\n" + "=" * 60)
    print("Candidate threshold sweep")
    print("(picks the threshold that best separates the two groups)")
    print("=" * 60)
    print(f"{'threshold':>10}{'accuracy':>10}{'precision':>11}{'recall':>9}")

    best_threshold, best_accuracy = None, -1
    for threshold in [round(x * 0.05, 2) for x in range(-4, 16)]:  # -0.20 to 0.75
        tp = sum(1 for s in answerable_scores if s >= threshold)
        fn = len(answerable_scores) - tp
        fp = sum(1 for s in not_answerable_scores if s >= threshold)
        tn = len(not_answerable_scores) - fp

        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0

        print(f"{threshold:>10.2f}{accuracy:>10.1%}{precision:>11.1%}{recall:>9.1%}")

        if accuracy > best_accuracy:
            best_accuracy, best_threshold = accuracy, threshold

    print(f"\nBest single accuracy: threshold={best_threshold} ({best_accuracy:.1%})")
    print("\nThis is a starting point, not an automatic answer -- accuracy alone")
    print("doesn't tell you whether you'd rather over-escalate (safe, wastes agent")
    print("time) or under-escalate (risky, an unanswerable ticket gets auto-drafted).")
    print("Given your own risk register rates confidently-wrong answers as a top")
    print("priority risk, consider favoring a threshold with higher RECALL on the")
    print("answerable_from_docs=True group even at some cost to overall accuracy --")
    print("that's your call, informed by the numbers above, not this script's.")