"""
Checks that can BLOCK a generated response before release (FR-09, FR-15,
FR-18-adjacent). This is a guardrail that actually blocks, not one that
only warns -- "a guardrail that only warns is not a guardrail."

FIRST REAL CHECK IMPLEMENTED HERE: grounding/faithfulness verification.

This exists because of a concrete, confirmed failure found during manual
review of generate.py's own smoke test: DEV-0003's answer claimed a
"6 months" retention period for a "standard plan," citing DOC-SEC-003 and
DOC-DATA-001 -- but DOC-SEC-003 explicitly states it applies only to
Business and Enterprise plans and never mentions 6 months. Neither
generate.py's citation validation nor its forbidden-claims keyword match
caught this, because both check DIFFERENT things (source exists; no known-
bad phrase appears) than whether the draft's actual content is supported.

TWO ROUNDS OF CONFIRMED BUGS, FOUND AND FIXED DURING TESTING:

Round 1: the first version of the LLM-judge check had a ~98% block rate
against 200 real ground-truth drafts. Manual inspection (DEV-0485) showed
the judge was echoing back verbatim SOURCE PASSAGE sentences and
mislabeling them as "unsupported claims." Fixed with an exact-substring
echo filter.

Round 2: after fixing round 1, the block rate was still ~54% (75/140).
Manual inspection of the new findings showed TWO more patterns: the judge
flagging close paraphrases of the source (DEV-0485, DEV-0331) despite
being explicitly told not to, and flagging generic boilerplate/politeness
language with no checkable factual content at all (DEV-0428: "please
don't hesitate to reach out"). Fixed with a word-overlap paraphrase filter
and a generic-phrasing filter.

KNOWN REMAINING LIMITATION: the paraphrase filter uses blunt word-overlap
and will miss a paraphrase that uses substantially different vocabulary
for the same claim (confirmed: DEV-0331's "not uncommon...several minutes
to complete" vs. source's "commonly...several minutes" only partially
overlaps and may still slip through in some cases). This is a deliberate
precision/recall tradeoff, not an oversight -- see the function's own
docstring for the reasoning.

TWO LAYERS, RUN TOGETHER:

1. Deterministic numeric/duration check (cheap, no API call): extracts
   number+time-unit claims from the draft and verifies each appears
   somewhere in the cited passages. Caught DEV-0003's "6 months" claim
   directly. Known gap: digit-only regex misses spelled-out numbers.

2. LLM-as-judge grounding check, with three filters (echo, paraphrase,
   generic-non-factual) applied to its raw output before trusting it.
   Still not exhaustive -- treat this as a real improvement over having
   nothing, not a guarantee.
"""
import json
import re
from dataclasses import dataclass

import requests

from src.config import OPENROUTER_API_KEY, MODEL_NAME
from src.generate import GeneratedAnswer

NUMBER_TIME_PATTERN = re.compile(
    r"\b(\d+)\s*[-\s]?(day|days|month|months|year|years|hour|hours|minute|minutes|week|weeks)\b",
    re.IGNORECASE,
)


@dataclass
class GuardrailResult:
    blocked: bool
    checks_run: list[str]
    findings: list[str]


def _extract_number_time_claims(text: str) -> list[tuple[str, str]]:
    claims = []
    for match in NUMBER_TIME_PATTERN.finditer(text):
        number, unit = match.group(1), match.group(2).lower().rstrip("s")
        claims.append((number, unit))
    return claims


def _check_numeric_grounding(draft_text: str, passages_text: str) -> list[str]:
    draft_claims = _extract_number_time_claims(draft_text)
    passage_claims = set(_extract_number_time_claims(passages_text))

    findings = []
    for number, unit in draft_claims:
        if (number, unit) not in passage_claims:
            findings.append(
                f"Draft claims '{number} {unit}' but no cited passage contains "
                f"that specific number+unit -- likely unsupported/fabricated."
            )
    return findings


def _is_verbatim_source_echo(finding: str, passages_text: str, threshold: int = 15) -> bool:
    """
    True if `finding` is essentially quoted source text rather than a real
    unsupported claim. A genuine "unsupported claim" describes something
    the DRAFT said; source text appearing in the source cannot, by
    definition, be unsupported by that source.
    """
    normalized_finding = finding.strip().lower().rstrip(".")
    normalized_passages = passages_text.lower()
    return normalized_finding in normalized_passages


def _is_paraphrase_of_source(finding: str, passages_text: str, overlap_threshold: float = 0.5) -> bool:
    """
    Catches a SECOND confirmed false-positive pattern (found after fixing
    the verbatim-echo bug, via manual inspection of DEV-0485/DEV-0428/
    DEV-0331): the judge flagging close paraphrases of source content as
    "unsupported," despite being explicitly told not to. Word-overlap
    ratio is a blunt instrument -- it will occasionally miss a genuine
    paraphrased fabrication, or over-filter a short finding that happens
    to share common words with the source. That tradeoff is deliberate:
    given the demonstrated failure mode is over-flagging, not under-
    flagging, erring toward fewer false positives here is the right side
    to be wrong on for a first pass.
    """
    finding_words = set(re.findall(r"[a-z]+", finding.lower()))
    finding_words = {w for w in finding_words if len(w) > 3}
    if not finding_words:
        return False
    passage_words = set(re.findall(r"[a-z]+", passages_text.lower()))
    overlap = len(finding_words & passage_words) / len(finding_words)
    return overlap >= overlap_threshold


def _is_generic_non_factual(finding: str) -> bool:
    """
    Catches the THIRD confirmed false-positive pattern: the judge flagging
    generic transitional/boilerplate language ("please don't hesitate to
    reach out", "this will ensure a smooth transition") that makes no
    checkable factual claim at all. Heuristic: a real factual claim worth
    grounding almost always contains something concrete -- a number, a
    specific named policy/feature, or a doc reference. Sentences with none
    of that are far more likely to be filler than fabrication.
    """
    has_digit = bool(re.search(r"\d", finding))
    has_doc_ref = bool(re.search(r"DOC-[A-Z]+-\d+", finding))
    generic_phrases = [
        "don't hesitate", "reach out", "smooth transition", "let us know",
        "feel free to", "please contact", "further assistance", "if you have any",
    ]
    is_generic_phrasing = any(p in finding.lower() for p in generic_phrases)
    return is_generic_phrasing and not has_digit and not has_doc_ref


def _check_llm_grounding(draft_text: str, passages_text: str) -> list[str]:
    judge_prompt = f"""You are a fact-checking guardrail. Compare the DRAFT
answer against the SOURCE passages it was supposedly grounded in.

List every specific factual claim made IN THE DRAFT (plan names, policies,
capabilities, numbers, promises) that is NOT directly supported by the
source text. Do not list sentences that come FROM the source itself --
you are checking what the DRAFT asserts, not summarizing the source.
Do not flag reasonable paraphrasing of something the source DOES say.
Only flag claims the source does not support at all, or contradicts.

SOURCE PASSAGES:
{passages_text}

DRAFT:
{draft_text}

Return ONLY valid JSON: {{"unsupported_claims": ["<claim, quoted from the DRAFT>", ...]}}
If everything is supported, return {{"unsupported_claims": []}}.
"""
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": judge_prompt}],
                "temperature": 0,
                "response_format": {"type": "json_object"},
            },
            timeout=30,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()
        if content.startswith("```"):
            content = content.split("```")[1].removeprefix("json").strip()
        result = json.loads(content)
        raw_findings = result.get("unsupported_claims", [])
    except (requests.RequestException, KeyError, json.JSONDecodeError, IndexError) as e:
        return [f"(grounding judge unavailable: {e} -- deterministic check is the only coverage this run)"]

    filtered = [
        f for f in raw_findings
        if not _is_verbatim_source_echo(f, passages_text)
        and not _is_paraphrase_of_source(f, passages_text)
        and not _is_generic_non_factual(f)
    ]
    return filtered


def check(answer: GeneratedAnswer, retrieved_passages: list[dict]) -> GuardrailResult:
    checks_run = []
    findings = []

    if not answer.knows_answer or not answer.text.strip():
        return GuardrailResult(blocked=False, checks_run=[], findings=[])

    passages_text = "\n\n".join(
        p["text"] for p in retrieved_passages if p["doc_id"] in answer.citations
    )

    try:
        checks_run.append("numeric_grounding")
        findings.extend(_check_numeric_grounding(answer.text, passages_text))

        checks_run.append("llm_grounding")
        findings.extend(_check_llm_grounding(answer.text, passages_text))
    except Exception as e:
        return GuardrailResult(
            blocked=True,
            checks_run=checks_run,
            findings=[f"guardrail internal error, failing closed: {e}"],
        )

    return GuardrailResult(
        blocked=len(findings) > 0,
        checks_run=checks_run,
        findings=findings,
    )


if __name__ == "__main__":
    fabricated_answer = GeneratedAnswer(
        text="Yes, you can export the access records covering the last six months. "
             "This period is within the retention window of our standard plan, "
             "which is 6 months.",
        citations=["DOC-SEC-003", "DOC-DATA-001"],
        knows_answer=True,
    )
    fabricated_passages = [
        {"doc_id": "DOC-SEC-003", "text": "Audit logs are retained for ninety days "
         "on Business plans and one year on Enterprise. Applies to Business and "
         "Enterprise plans."},
        {"doc_id": "DOC-DATA-001", "text": "The underlying export file is retained "
         "for seven days after generation."},
    ]
    result1 = check(fabricated_answer, fabricated_passages)
    print("Test 1 (should BLOCK - real fabrication):", result1.blocked)
    for f in result1.findings:
        print("  -", f)
    assert result1.blocked, "Regression: the filters must not hide real fabrications"

    good_passages = [
        {"doc_id": "DOC-ACCT-001", "text": "A user cannot access a project they "
         "should be able to. The role was granted at project scope rather than "
         "organisation scope. The invitation was never accepted and the account "
         "does not exist. A group membership grants broader access than the "
         "direct role suggests. Check the effective permissions view for the "
         "user, which resolves direct roles and group memberships together. "
         "Confirm the invitation was accepted."},
    ]
    good_answer = GeneratedAnswer(
        text="It's possible that the role was granted at project scope rather "
             "than organisation scope, which is a common cause of this issue. "
             "You could check the effective permissions view, which resolves "
             "direct roles and group memberships together.",
        citations=["DOC-ACCT-001"],
        knows_answer=True,
    )
    result2 = check(good_answer, good_passages)
    print("\nTest 2 (should PASS - well-grounded answer):", not result2.blocked)
    for f in result2.findings:
        print("  -", f)

    print("\nAll guardrail regression tests passed." if result1.blocked and not result2.blocked else "\nREGRESSION FAILURE")
