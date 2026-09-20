"""
The decision log: one row per action the system takes on a ticket, with
enough recorded to reconstruct and defend the decision later. This is
the table your governance/audit work in Stage 3+ reads from, and it's
what you'll show in the video when asked "why did it do that."

Columns follow the Setup Guide exactly. Don't slim this down -- every
column earns its place in the governance write-up.
"""
import sqlite3
import uuid
from datetime import datetime, timezone
from src.config import DATABASE_URL

# DATABASE_URL is sqlite:///./storage/decisions.db -- strip the prefix
# to get a plain file path for sqlite3.connect.
DB_PATH = DATABASE_URL.replace("sqlite:///", "", 1)


def init_db(path: str = DB_PATH) -> None:
    connection = sqlite3.connect(path)
    connection.execute("""
        CREATE TABLE IF NOT EXISTS decisions (
            decision_id     TEXT PRIMARY KEY,
            created_at      TEXT NOT NULL,
            ticket_id       TEXT NOT NULL,
            stage           TEXT NOT NULL,
            prediction      TEXT,
            confidence      REAL,
            threshold       REAL,
            action_taken    TEXT NOT NULL,
            reason          TEXT NOT NULL,
            sources_used    TEXT,
            guardrails      TEXT,
            prompt_version  TEXT,
            requirement_ids TEXT
        )
    """)
    connection.commit()
    connection.close()


def log_decision(
    ticket_id: str,
    stage: str,
    action_taken: str,
    reason: str,
    prediction: str | None = None,
    confidence: float | None = None,
    threshold: float | None = None,
    sources_used: str | None = None,
    guardrails: str | None = None,
    prompt_version: str | None = None,
    requirement_ids: str | None = None,
    path: str = DB_PATH,
) -> str:
    """Writes one decision row and returns its decision_id."""
    decision_id = str(uuid.uuid4())
    connection = sqlite3.connect(path)
    connection.execute(
        """INSERT INTO decisions (
            decision_id, created_at, ticket_id, stage, prediction,
            confidence, threshold, action_taken, reason, sources_used,
            guardrails, prompt_version, requirement_ids
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            decision_id,
            datetime.now(timezone.utc).isoformat(),
            ticket_id, stage, prediction, confidence, threshold,
            action_taken, reason, sources_used, guardrails,
            prompt_version, requirement_ids,
        ),
    )
    connection.commit()
    connection.close()
    return decision_id


if __name__ == "__main__":
    init_db()
    print(f"decision log ready at {DB_PATH}")
