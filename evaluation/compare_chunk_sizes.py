"""
Compares retrieval quality across chunk-size configurations using real
ground truth: labels.expected_doc_ids tells you which KB article SHOULD
be retrieved for each answerable ticket. This turns "which chunk size is
better" from a subjective read into a measured recall@k comparison.

Only tickets with answerable_from_docs=True and a non-empty expected_doc_ids
are used -- there's no ground truth to check against otherwise.

Run from your project root: python evaluation/compare_chunk_sizes.py
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieve import build_index, retrieve

CONFIGS = [
    {"name": "500/75", "chunk_size": 500, "chunk_overlap": 75, "persist_directory": "storage/chroma_500"},
    {"name": "800/120", "chunk_size": 800, "chunk_overlap": 120, "persist_directory": "storage/chroma_800"},
]

K = 5  # how many results to check for the correct doc
SAMPLE_SIZE = 100  # set to None to use every eligible ticket


def load_eval_tickets(path="data/development_tickets.json"):
    with open(path) as f:
        tickets = json.load(f)
    eligible = [
        t for t in tickets
        if t["labels"].get("answerable_from_docs") and t["labels"].get("expected_doc_ids")
    ]
    if SAMPLE_SIZE:
        eligible = eligible[:SAMPLE_SIZE]
    return eligible


def evaluate_config(config, tickets):
    print(f"\nBuilding index at chunk_size={config['chunk_size']}, "
          f"overlap={config['chunk_overlap']} -> {config['persist_directory']} ...")
    build_index(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
        persist_directory=config["persist_directory"],
    )

    hit_at_1 = 0
    hit_at_k = 0
    misses = []

    for t in tickets:
        text = f"{t['subject']}\n{t['body']}".strip() if t["subject"] else t["body"]
        expected = set(t["labels"]["expected_doc_ids"])
        results = retrieve(text, k=K)
        retrieved_ids = [r["doc_id"] for r in results]

        if retrieved_ids and retrieved_ids[0] in expected:
            hit_at_1 += 1
        if any(doc_id in expected for doc_id in retrieved_ids):
            hit_at_k += 1
        else:
            misses.append((t["ticket_id"], t["labels"]["intent"], expected, retrieved_ids))

    n = len(tickets)
    print(f"  recall@1: {hit_at_1}/{n} = {hit_at_1/n*100:.1f}%")
    print(f"  recall@{K}: {hit_at_k}/{n} = {hit_at_k/n*100:.1f}%")
    return {"config": config["name"], "recall_at_1": hit_at_1/n, "recall_at_k": hit_at_k/n, "misses": misses}


if __name__ == "__main__":
    tickets = load_eval_tickets()
    print(f"Evaluating on {len(tickets)} answerable tickets with known expected_doc_ids")

    results = [evaluate_config(c, tickets) for c in CONFIGS]

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for r in results:
        print(f"{r['config']:<12} recall@1={r['recall_at_1']*100:5.1f}%  "
              f"recall@{K}={r['recall_at_k']*100:5.1f}%  "
              f"misses={len(r['misses'])}")

    print("\nThis prints the numbers. Deciding which config to ship -- and whether")
    print("the difference is even large enough to matter -- is your call, not this")
    print("script's. If you want to see WHAT was missed, inspect result['misses']")
    print("for either config above (ticket_id, intent, expected doc, what came back).")
    print("\nNOTE: this script wrote to storage/chroma_500 and storage/chroma_800,")
    print("NOT the default storage/chroma that src/api.py actually reads from.")
    print("Once you've picked a config, rebuild the real index with it:")
    print('  python -c "from src.retrieve import build_index; build_index(chunk_size=..., chunk_overlap=...)"')