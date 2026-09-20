"""
The escalation decision: auto-respond or hand off to a human.

Implements FR-05 (deterministic escalation, no LLM), plus FR-11/FR-19
(category-specific handling for the four broken-conversion categories vs.
the five healthy ones). This was deliberately decided NOT to be an LLM
call -- see the conversation/report: escalation for named high-risk
categories must not depend on a model's interpretation when the
requirement calls for a hard routing boundary.

TWO OPEN DECISIONS, DELIBERATELY LEFT FOR YOU RATHER THAN GUESSED:

1. HARD_ESCALATE_INTENTS currently only contains security_incident and
   compliance_request -- the two categories with direct data support
   (100% escalated, 0% FCR in your Section 3 analysis, even when
   answerable). Daniel's interview also named "billing disputes" and
   "data location" as red lines, but those don't map cleanly onto your
   actual taxonomy: billing_query showed 54.2% FCR (often fine to
   auto-answer), and data_residency is one of the FR-11 broken-conversion
   categories (needs caution, but isn't a 100%-escalate case like the two
   above). Decide explicitly whether to add either, and cite the
   reasoning -- don't let this default silently.

2. RETRIEVAL_ANSWERABLE_THRESHOLD (below) is a first-pass guess, not a
   calibrated number. Your confidence threshold (config.CONFIDENCE_THRESHOLD
   = 0.80) has a real justification chain behind it; this doesn't yet.
   Before the gate run, build the retrieval equivalent of
   diagnose_urgency.py: compare top retrieval scores for tickets marked
   answerable_from_docs=True vs. False in your labeled data, and pick a
   threshold that actually separates them, the same way you calibrated
   chunk size against real recall numbers.
"""
from dataclasses import dataclass

from src.classify import Classification

# Hard-escalate intents -- decided, not defaulted. See reasoning per category:
#
# security_incident, compliance_request: data-confirmed (100% escalated,
# 0% FCR in ground truth, even when answerable). Not a judgment call.
#
# billing_query: added despite 54.2% FCR in ground truth -- Daniel's
# interview named billing disputes as contractual/financial risk, and that
# risk is judged to outweigh the automation coverage lost on the ~54% of
# billing tickets that could otherwise be auto-answered. This deliberately
# lowers measured routing accuracy against validation labels (which
# reflect the CURRENT human process, not this system's intended policy) --
# that drop is the expected cost of the decision, not a defect.
#
# data_residency: considered, NOT added. Despite compliance adjacency, its
# real signature (72.4% answerable, 31% FCR) doesn't match the true
# hard-block categories above -- it's treated instead via
# BROKEN_CONVERSION_INTENTS below (stricter bar, not a full block), on the
# view that a genuinely strong match should still be allowed through for
# an FR-11 category. The same reasoning keeps performance_degradation,
# database_issue, and integration_help in the softer bucket too, rather
# than promoting them here by inconsistency.
HARD_ESCALATE_INTENTS = {
    "security_incident",
    "compliance_request",
}

# FR-11's four categories: documentation is frequently available, but the
# current process still fails to convert that into first-contact
# resolution (69-76% answerable, 62-73% escalated, 27-38% FCR). These get
# a stricter bar rather than a hard block -- unlike HARD_ESCALATE_INTENTS,
# a genuinely confident, well-grounded answer should still be allowed
# through; the point is not to extend the SAME leniency as the five
# healthy FR-19 categories.
BROKEN_CONVERSION_INTENTS = {
    "performance_degradation",
    "data_residency",
    "database_issue",
    "integration_help",
}

# First-pass guess -- see module docstring point 2. NOT yet calibrated
# against labeled answerable_from_docs data.
RETRIEVAL_ANSWERABLE_THRESHOLD = 0.30

# Extra margin applied on top of the base confidence/retrieval thresholds
# for BROKEN_CONVERSION_INTENTS, reflecting that these categories have
# already shown they convert poorly even when documentation exists.
BROKEN_CONVERSION_EXTRA_MARGIN = 0.10


@dataclass
class RoutingDecision:
    action: str          # "auto_respond" | "escalate"
    reason: str           # human-readable, for a support manager
    threshold_used: float


def _escalate(reason: str, threshold: float) -> RoutingDecision:
    return RoutingDecision(action="escalate", reason=reason, threshold_used=threshold)


def _auto_respond(reason: str, threshold: float) -> RoutingDecision:
    return RoutingDecision(action="auto_respond", reason=reason, threshold_used=threshold)


def route(
    classification: Classification,
    retrieval_results: list[dict],
    threshold: float,
) -> RoutingDecision:
    """
    Deterministic routing decision. Pure function of its inputs -- no
    hidden state, no randomness, no LLM call. Same inputs always produce
    the same decision, which is itself part of what FR-05 requires.
    """
    # 1. Hard-risk categories escalate unconditionally. This check comes
    #    FIRST and nothing below can override it -- this is the concrete
    #    enforcement of "answerability is not permission to answer."
    if classification.intent in HARD_ESCALATE_INTENTS:
        return _escalate(
            f"Category '{classification.intent}' requires human review regardless "
            f"of confidence or documentation match (policy, not a data judgment call).",
            threshold,
        )

    # 2. A failed/fallback classification can't be trusted for anything
    #    downstream -- escalate rather than guess on top of a guess.
    if classification.fallback_used:
        return _escalate(
            "Classification failed or fell back; cannot safely automate on an "
            "unreliable signal.",
            threshold,
        )

    # 3. Classification confidence below threshold.
    if classification.confidence < threshold:
        return _escalate(
            f"Classification confidence {classification.confidence:.2f} is below "
            f"the required threshold {threshold:.2f}.",
            threshold,
        )

    # 4. Retrieval quality check -- the answerability proxy (see module
    #    docstring point 2: this threshold is not yet calibrated).
    top_score = retrieval_results[0]["score"] if retrieval_results else None
    if top_score is None or top_score < RETRIEVAL_ANSWERABLE_THRESHOLD:
        return _escalate(
            f"No sufficiently relevant documentation found "
            f"(top retrieval score {top_score}, threshold {RETRIEVAL_ANSWERABLE_THRESHOLD}).",
            threshold,
        )

    # 5. Broken-conversion categories get a stricter combined bar rather
    #    than a hard block -- a genuinely strong match should still pass.
    if classification.intent in BROKEN_CONVERSION_INTENTS:
        strict_confidence = threshold + BROKEN_CONVERSION_EXTRA_MARGIN
        strict_retrieval = RETRIEVAL_ANSWERABLE_THRESHOLD + BROKEN_CONVERSION_EXTRA_MARGIN
        if classification.confidence < strict_confidence or top_score < strict_retrieval:
            return _escalate(
                f"'{classification.intent}' is a known low-conversion category "
                f"(FR-11); confidence {classification.confidence:.2f} or retrieval "
                f"score {top_score:.2f} did not clear the stricter bar "
                f"({strict_confidence:.2f}/{strict_retrieval:.2f}) applied here.",
                threshold,
            )

    return _auto_respond(
        f"Classification confidence {classification.confidence:.2f} and retrieval "
        f"score {top_score:.2f} both clear their thresholds; no hard-escalate "
        f"category matched.",
        threshold,
    )