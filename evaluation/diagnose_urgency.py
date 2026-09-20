"""
Diagnostic: what is classify() actually predicting for urgency, and how does
it compare to the true label? Run this before changing the prompt, so the
fix targets the real failure pattern instead of a guess.

Run from your project root: python evaluation/diagnose_urgency.py
"""
import json
import sys
import os
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.classify import classify

def ticket_text(t):
    return f"{t['subject']}\n{t['body']}".strip() if t.get("subject") else t["body"]


if __name__ == "__main__":
    tickets = json.load(open("data/validation_tickets.json"))

    true_dist = Counter()
    predicted_dist = Counter()
    confusion = Counter()  # (true, predicted) pairs

    for t in tickets:
        true_urgency = t["labels"]["urgency"]
        result = classify(ticket_text(t))

        true_dist[true_urgency] += 1
        predicted_dist[result.urgency] += 1
        confusion[(true_urgency, result.urgency)] += 1

    print("True urgency distribution:", dict(true_dist))
    print("Predicted urgency distribution:", dict(predicted_dist))
    print()
    header_label = "true \\ predicted"
    print(f"{header_label:<12}{'low':>8}{'medium':>8}{'high':>8}")
    for true_val in ["low", "medium", "high"]:
        row = [confusion.get((true_val, pred_val), 0) for pred_val in ["low", "medium", "high"]]
        print(f"{true_val:<12}{row[0]:>8}{row[1]:>8}{row[2]:>8}")