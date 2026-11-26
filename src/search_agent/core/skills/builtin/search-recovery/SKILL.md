---
name: search-recovery
description: Use after empty, repetitive, irrelevant, or conflicting retrieval results, or when a candidate answer cannot be verified.
---

# Search recovery

## Purpose

Use this procedure after retrieval is empty, repetitive, irrelevant, contradictory, or
unable to verify a candidate. It is also appropriate when several calls return the same
documents without adding a new fact. Do not load it for a normal first search.

## Diagnose before retrying

1. Inspect the last query, retrieval method, returned document IDs, and snippets.
2. Classify the failure:
   - **too narrow**: too many constraints or overly exact wording;
   - **vocabulary mismatch**: alias, abbreviation, spelling, or synonym differs;
   - **ambiguous**: a name or phrase has several meanings;
   - **unsupported assumption**: the query presupposes a fact not in evidence;
   - **source conflict**: documents provide incompatible dates, names, or relationships;
   - **localization failure**: the right document is known but the passage was not found.
3. State the diagnosis internally before making the next call. Do not retry an unchanged
   query or merely increase `top_k` without changing the evidence strategy.

## Recovery procedure

1. Change exactly one search dimension:
   - remove one weak constraint;
   - replace a term with an alias or synonym;
   - search a rare exact phrase;
   - search the candidate name with the missing constraint;
   - change the `retrieve` method when the configured method is unsuitable.
2. Compare the new document IDs with earlier results. Prefer genuinely new evidence,
   but revisit a repeated document when a new query may expose a different passage.
3. Open promising documents before discarding a candidate. Use `find_in_document` for a
   known document and `inspect_document` for focused reading around a query.
4. Preserve contradictions in the working evidence. A plausible bridge between two
   unrelated snippets is not evidence.
5. If recovery still fails, try a different clue rather than repeating the same tactic.
   For a multi-constraint question, load `multi-clue-decomposition` only when the task
   now requires systematic candidate comparison.
6. Stop when the answer is supported, the search budget is nearly exhausted, or two
   materially different recovery strategies produce no useful evidence. Report the
   failed strategy and unresolved uncertainty instead of inventing an answer.

## DeepSeek-V4-Flash decision rule

Select this skill from query state, not from the user's wording alone: empty results,
low-relevance repetition, conflicting facts, or an unverifiable candidate are triggers.
If the next action is obvious and evidence is already sufficient, do not load the skill.
Unload it once recovery succeeds or when switching to an unrelated research task.
