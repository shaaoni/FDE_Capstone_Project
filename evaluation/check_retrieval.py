"""
Retrieval sanity check: pulls a real ticket and shows what the retriever
finds for it, so you can eyeball whether the top passages actually match
what the ticket is asking about. Run from your project root.

Usage: python check_retrieval.py [ticket_index]
"""
import json
import sys
from src.retrieve import retrieve

tickets = json.load(open("data/development_tickets.json"))
idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
sample = tickets[idx]

print(f"TICKET #{idx} [{sample['labels']['intent']}]:")
print(sample["body"][:200])
print()

results = retrieve(sample["body"], k=3)
for i, r in enumerate(results, 1):
    print(f"{i}. [{r['score']:.3f}] {r['title']} ({r['doc_id']})")
    print("   ", r["text"][:150].replace("\n", " "))
    print()