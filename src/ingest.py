"""
Normalises raw tickets into one consistent shape for the rest of the
pipeline to consume.

Note: in the dataset provided (data/development_tickets.json), all four
channels (chat, email, forum, docs_comment) already arrive as JSON with
the same field names -- channel, subject, body, received_at, customer_id,
customer_name, customer_tier, customer_region, language_fluency.
So "normalising across channels" here mostly means:
  - validating the shape (missing/empty fields, e.g. many chat tickets
    have an empty "subject")
  - producing one Ticket object the rest of src/ can rely on
  - NOT touching the "labels" or "history" keys, which are ground truth
    for training/evaluation only and must never be fed to the model at
    inference time (that would be leaking the answer to the system
    you're grading).

If you integrate a real support platform later, this is the module that
would absorb each provider's actual schema differences.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
import json


@dataclass
class Ticket:
    ticket_id: str
    channel: str
    subject: str
    body: str
    received_at: datetime
    customer_id: str
    customer_name: str
    customer_tier: str
    customer_region: str
    language_fluency: str

    @property
    def text(self) -> str:
        """Subject + body, the field you'll actually feed to a model."""
        return f"{self.subject}\n{self.body}".strip() if self.subject else self.body


def load_tickets(path: str) -> list[Ticket]:
    with open(path) as f:
        raw = json.load(f)

    tickets = []
    for row in raw:
        tickets.append(Ticket(
            ticket_id=row["ticket_id"],
            channel=row["channel"],
            subject=row.get("subject", "") or "",
            body=row["body"],
            received_at=datetime.fromisoformat(row["received_at"].replace("Z", "+00:00")),
            customer_id=row["customer_id"],
            customer_name=row["customer_name"],
            customer_tier=row["customer_tier"],
            customer_region=row["customer_region"],
            language_fluency=row["language_fluency"],
        ))
    return tickets


if __name__ == "__main__":
    tickets = load_tickets("data/development_tickets.json")
    print(f"loaded {len(tickets)} tickets")
    by_channel = {}
    for t in tickets:
        by_channel[t.channel] = by_channel.get(t.channel, 0) + 1
    print(by_channel)
