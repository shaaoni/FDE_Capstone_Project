"""
Example test to get the suite running in CI. Add tests for classify,
route, generate and guardrails as you build them -- the CI workflow
already runs everything under tests/ on every push.
"""
from src.ingest import load_tickets


def test_load_tickets_returns_expected_count():
    tickets = load_tickets("data/development_tickets.json")
    assert len(tickets) == 500


def test_ticket_text_property_combines_subject_and_body():
    tickets = load_tickets("data/development_tickets.json")
    chat_ticket = next(t for t in tickets if t.channel == "chat")
    # chat tickets in this dataset commonly have an empty subject
    assert chat_ticket.body in chat_ticket.text
