# CloudServe Support Pipeline

_A brief description of the system goes here once it's built — what it
does, and the one-sentence version of the problem it solves. Write this
last, not first._

## Setup

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 3. Protect your API key before it exists
echo ".env" >> .gitignore

# 4. Configure
cp .env.example .env
# then edit .env and add your OPENROUTER_API_KEY (openrouter.ai, free tier)

# 5. Build the vector store (run once, or whenever data/documentation.json changes)
python -m src.retrieve

# 6. Initialize the decision log
python -m src.logging_store
```

## Verify your key works

```bash
python -c "
import os, requests
from dotenv import load_dotenv
load_dotenv()
r = requests.post(
    'https://openrouter.ai/api/v1/chat/completions',
    headers={'Authorization': f\"Bearer {os.environ['OPENROUTER_API_KEY']}\"},
    json={'model': os.environ['MODEL_NAME'],
          'messages': [{'role': 'user', 'content': 'Reply with the word ready.'}]},
    timeout=30,
)
print(r.status_code); print(r.json())
"
```

## Run the API

```bash
uvicorn src.api:app --reload
# health check:   curl localhost:8000/health
# metrics:        curl localhost:8001/metrics
```

## Run the full evaluation set (the gate)

```bash
# TODO once evaluation/harness.py is built (Stage 4/5):
python -m evaluation.harness --input data/validation_tickets.json --output evaluation/results/
```

## Run tests

```bash
python -m pytest tests/ -v
```

## Project structure

```
src/            pipeline components (ingest, classify, retrieve, route, generate, guardrails, logging_store, api)
prompts/        versioned prompt register (build/ and evaluation/)
tests/          pytest suite, run in CI on every push
evaluation/     harness that runs the full ticket set unattended + dated results
docs/           architecture notes
data/           small samples only — full datasets are not committed
storage/        generated at runtime (chroma index, sqlite db) — never committed
```

## Status

This is a scaffold. `classify.py`, `route.py`, `generate.py`, and
`guardrails.py` currently raise `NotImplementedError` — see the docstring
in each file for what design decisions need to be made (and justified)
before implementing them.
