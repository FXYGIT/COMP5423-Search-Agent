---
name: search-recovery
description: Use after empty, repetitive, irrelevant, or conflicting retrieval results, or when a candidate answer cannot be verified.
---

# Search recovery

1. Inspect the last query and returned snippets. Identify whether the failure came from overly narrow phrasing, ambiguous names, missing synonyms, or an unsupported assumption.
2. Change one search dimension: remove a weak constraint, use an alias, search an exact distinctive phrase, or switch the method parameter of retrieve. Do not repeat an unchanged query.
3. Use open_document or find_in_document to check promising evidence before discarding a candidate.
4. Compare new document IDs and facts with earlier results. If no useful evidence appears, try a different clue instead of merely increasing top_k.
5. Keep contradictory evidence visible. Never invent a bridge between unrelated documents.
6. Stop when evidence supports an answer, the call budget is nearly exhausted, or further searches no longer add evidence. Report unresolved uncertainty.
