# Architecture

_Fill this in during Stage 2 (requirements/architecture design), once your
discovery findings are in. It should cover, at minimum:_

- The end-to-end flow (ingest → classify → retrieve → route → generate →
  guardrails → log) and why each stage exists
- The routing threshold and how you derived it from data
- The chunking strategy for retrieval and what you measured to choose it
- The guardrail checks and what they protect against
- Failure modes: what happens when the model provider is unavailable or
  degraded (Build Spec acceptance criterion A11)
- A diagram (even a simple one) showing the components and data flow
