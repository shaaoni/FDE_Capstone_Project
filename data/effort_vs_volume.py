"""
Section 3 support tool: computes volume share vs. effort share by intent,
so you're transcribing numbers correctly instead of retyping them by hand.

Effort proxy = each intent's share of total resolution minutes across all
500 tickets (same method as before -- this script doesn't change your
approach, it just removes manual-copy errors from it).

This prints numbers only. The "why the two differ" and "evidence for your
estimate" columns are still yours to write -- this just guarantees the
volume%/effort%/avg columns you're writing them against are correct.

Run: python3 effort_vs_volume.py
"""
import json
from collections import defaultdict


def load(path="development_tickets.json"):
    with open(path) as f:
        return json.load(f)


def main():
    tickets = load()
    total_tickets = len(tickets)

    by_intent = defaultdict(list)
    for t in tickets:
        by_intent[t["labels"]["intent"]].append(t["history"]["resolution_time_minutes"])

    total_minutes = sum(v for vals in by_intent.values() for v in vals)

    rows = []
    for intent, vals in by_intent.items():
        n = len(vals)
        vol_share = n / total_tickets * 100
        effort = sum(vals)
        effort_share = effort / total_minutes * 100
        avg_min = effort / n
        rows.append((intent, n, vol_share, effort_share, avg_min))

    rows.sort(key=lambda r: -r[3])  # sorted by effort share, descending

    print(f"total tickets: {total_tickets}, total resolution minutes: {total_minutes}\n")
    print(f"{'intent':<26}{'n':>5}{'vol %':>9}{'effort %':>11}{'avg min':>10}")
    print("-" * 61)
    for intent, n, vol, effort, avg in rows:
        print(f"{intent:<26}{n:>5}{vol:>9.1f}{effort:>11.1f}{avg:>10.1f}")

    print("-" * 61)
    print(f"{'TOTAL':<26}{sum(r[1] for r in rows):>5}"
          f"{sum(r[2] for r in rows):>9.1f}{sum(r[3] for r in rows):>11.1f}")
    print("\n(both percentage columns should sum to ~100.0 -- check before transcribing)")

    # markdown table, ready to paste in and fill the remaining two columns
    print("\n\n--- markdown, paste into workbook and fill last two columns ---\n")
    print("| Ticket category | Volume % | Effort % | Avg resolution (min) | Why the two differ | Evidence |")
    print("|---|---:|---:|---:|---|---|")
    for intent, n, vol, effort, avg in rows:
        label = intent.replace("_", " ").capitalize()
        print(f"| {label} | {vol:.1f}% | {effort:.1f}% | {avg:.1f} | | |")


if __name__ == "__main__":
    main()
