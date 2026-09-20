# Prompt Register

Every prompt that runs inside the system or evaluates it must be versioned
here. This is what Stage 3 (Prompt Library) produces — don't skip straight
to writing prompt strings inside your Python files.

| ID | File | Purpose | Version | Last changed | Why |
|----|------|---------|---------|---------------|-----|
| | | | | | |

- `build/` — prompts that run inside the system itself (classify, generate, etc.)
- `evaluation/` — prompts used to judge output (e.g. an LLM-as-judge grader)

Each prompt file should carry its version in its filename or header
(e.g. `classify_v1.txt`, `classify_v2.txt`) so you can point to exactly
which version produced a given evaluation result.
