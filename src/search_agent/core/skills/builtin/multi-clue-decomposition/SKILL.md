---
name: multi-clue-decomposition
description: Use when a question identifies an entity through multiple independent clues, dates, relationships, or constraints that must all hold.
---

# Multi-clue decomposition

1. Separate the question into constraints; distinguish identifying clues from facts to verify.
2. Search the most distinctive clue first with the unified retrieve tool. Avoid one oversized query containing every clue.
3. Open promising documents and record candidate entities with supporting document IDs.
4. Search each remaining constraint with the candidate's name. A shared keyword alone is not confirmation.
5. Reject candidates contradicted by a date, relationship, or other required condition. Revise the query using the missing constraint.
6. Return an answer only when the collected evidence supports all essential constraints. Cite document IDs and disclose unresolved ambiguity.
