"""
Verifies B-10 (decision logging) end to end.

Calls process_ticket() directly, then reads the actual row back from
storage/decisions.db and checks whether the FR-10 audit fields are populated.

FR-10 audit fields checked here:
- source used
- response/decision
- classification
- routing decision
- human-intervention / guardrail status
"""

import os
import sqlite3
import sys

# Make project root importable when this file is run directly.
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.api import process_ticket, TicketIn
from src.logging_store import init_db
from src.config import DATABASE_URL


DB_PATH = DATABASE_URL.replace(
    "sqlite:///",
    "",
    1,
)


def main():

    init_db()

    test_ticket = TicketIn(
        ticket_id="VERIFY-LOG-001",
        channel="email",
        subject="",
        body=(
            "Our builds that worked last week are now failing "
            "during dependency resolution."
        ),
    )

    print("Calling process_ticket() directly...")

    result = process_ticket(test_ticket)

    print(f"Result: {result}")

    print("\nReading the row back from storage/decisions.db...")

    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row

    row = connection.execute(
        """
        SELECT *
        FROM decisions
        WHERE ticket_id = ?
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (test_ticket.ticket_id,),
    ).fetchone()

    connection.close()

    if row is None:
        print(
            "\nFAILED: no row was written for this ticket_id at all."
        )
        return 1

    row = dict(row)

    print("\nFull row as stored:")

    for key, value in row.items():
        print(f"  {key}: {value!r}")

    print("\n" + "=" * 60)
    print("FR-10 REQUIRED FIELD CHECK")
    print("=" * 60)

    checks = {
        "source used             (sources_used)": row.get(
            "sources_used"
        ),
        "response/decision       (action_taken)": row.get(
            "action_taken"
        ),
        "classification          (prediction)": row.get(
            "prediction"
        ),
        "routing decision        (reason)": row.get(
            "reason"
        ),
        "human-intervention flag (guardrails)": row.get(
            "guardrails"
        ),
    }

    all_present = True

    for label, value in checks.items():

        present = value not in (None, "")

        if not present:
            all_present = False

        status = (
            "PRESENT"
            if present
            else "MISSING/NULL"
        )

        print(
            f"  [{status:<12}] {label} = {value!r}"
        )

    print()

    if all_present:
        print("B-10 VERIFICATION: PASSED")
        return 0

    print("B-10 VERIFICATION: FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())