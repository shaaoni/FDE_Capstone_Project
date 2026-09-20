import json
from collections import defaultdict


def summarize(tickets, field):
    groups = defaultdict(list)

    for ticket in tickets:
        groups[ticket.get(field, "unknown")].append(ticket)

    print(f"\n{'=' * 70}")
    print(f"FAIRNESS AUDIT BY {field.upper()}")
    print(f"{'=' * 70}")

    for group, rows in sorted(groups.items()):
        n = len(rows)

        fcr = sum(
            r["history"]["first_contact_resolution"]
            for r in rows
        ) / n

        avg_resolution = sum(
            r["history"]["resolution_time_minutes"]
            for r in rows
        ) / n

        avg_csat = sum(
            r["history"]["csat_rating"]
            for r in rows
        ) / n

        escalation = sum(
            r["history"]["escalated"]
            for r in rows
        ) / n

        print(
            f"{group:15} "
            f"n={n:3d}  "
            f"FCR={fcr * 100:5.1f}%  "
            f"resolution={avg_resolution:7.1f} min  "
            f"CSAT={avg_csat:4.2f}  "
            f"escalation={escalation * 100:5.1f}%"
        )


if __name__ == "__main__":
    with open("data/development_tickets.json") as f:
        tickets = json.load(f)

    summarize(tickets, "customer_tier")
    summarize(tickets, "language_fluency")