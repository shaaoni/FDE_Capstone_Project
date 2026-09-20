"""
FastAPI application wiring the pipeline together end to end:

ingest -> classify -> retrieve -> route -> generate -> guardrails -> log
"""

from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, start_http_server

from src.classify import classify
from src.retrieve import retrieve
from src.route import route
from src.generate import generate
from src.guardrails import check as run_guardrails
from src.logging_store import log_decision, init_db
from src.config import CONFIDENCE_THRESHOLD


app = FastAPI(title="CloudServe Support Pipeline")


TICKETS = Counter(
    "tickets_processed_total",
    "Tickets processed",
    ["channel", "outcome"],
)

LATENCY = Histogram(
    "response_seconds",
    "End to end response time",
)

GUARDRAIL = Counter(
    "guardrail_blocks_total",
    "Responses blocked",
    ["guardrail"],
)


class TicketIn(BaseModel):
    ticket_id: str
    channel: str
    subject: str = ""
    body: str


@app.on_event("startup")
def startup():
    init_db()
    start_http_server(8001)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/process")
def process_ticket(ticket: TicketIn):
    with LATENCY.time():

        text = (
            f"{ticket.subject}\n{ticket.body}".strip()
            if ticket.subject
            else ticket.body
        )

        # ---------------------------------------------------------------
        # 1. CLASSIFY
        # ---------------------------------------------------------------
        classification = classify(text)

        # ---------------------------------------------------------------
        # 2. RETRIEVE
        # ---------------------------------------------------------------
        passages = retrieve(text)

        # ---------------------------------------------------------------
        # 3. ROUTE
        # ---------------------------------------------------------------
        decision = route(
            classification,
            passages,
            CONFIDENCE_THRESHOLD,
        )

        result = {
            "ticket_id": ticket.ticket_id,
            "action": decision.action,
            "reason": decision.reason,
        }

        # Record what documentation was available to the pipeline.
        # This must be populated even when the ticket is escalated before
        # generation so the audit record can reconstruct the decision.
        sources_used = ",".join(
            p["doc_id"]
            for p in passages
            if p.get("doc_id")
        )

        # Human-review / guardrail audit information.
        guardrail_summary = (
            "not_run (escalated before generation); "
            "human_review_required=True"
        )

        # ---------------------------------------------------------------
        # 4. GENERATE + 5. GUARDRAIL
        # ---------------------------------------------------------------
        if decision.action == "auto_respond":

            answer = generate(
                text,
                passages,
                prompt_version="v1",
            )

            # Generation declined to answer.
            if not answer.knows_answer:

                result["action"] = "escalate"
                result["reason"] = (
                    "generate() could not produce a grounded answer"
                )

                guardrail_summary = (
                    "not_run (generate() declined); "
                    "human_review_required=True"
                )

            else:

                guardrail_result = run_guardrails(
                    answer,
                    passages,
                )

                guardrail_summary = (
                    f"checks_run={guardrail_result.checks_run}, "
                    f"blocked={guardrail_result.blocked}, "
                    f"human_review_required={guardrail_result.blocked}"
                )

                if guardrail_result.blocked:

                    GUARDRAIL.labels(
                        guardrail="generate_check"
                    ).inc()

                    result["action"] = "escalate"

                    result["reason"] = (
                        "Blocked by guardrail: "
                        f"{guardrail_result.findings}"
                    )

                    guardrail_summary += (
                        f", findings={guardrail_result.findings}"
                    )

                else:

                    result["answer"] = answer.text
                    result["citations"] = answer.citations

                    # For a released answer, record only the sources that
                    # were actually cited in the customer response.
                    sources_used = ",".join(
                        answer.citations
                    )

        # ---------------------------------------------------------------
        # 6. DECISION LOG
        # ---------------------------------------------------------------
        log_decision(
            ticket_id=ticket.ticket_id,
            stage="route",
            action_taken=result["action"],
            reason=result["reason"],
            prediction=classification.intent,
            confidence=classification.confidence,
            threshold=CONFIDENCE_THRESHOLD,
            prompt_version="v1",
            sources_used=sources_used,
            guardrails=guardrail_summary,
        )

        # ---------------------------------------------------------------
        # 7. METRICS
        # ---------------------------------------------------------------
        TICKETS.labels(
            channel=ticket.channel,
            outcome=result["action"],
        ).inc()

        return result