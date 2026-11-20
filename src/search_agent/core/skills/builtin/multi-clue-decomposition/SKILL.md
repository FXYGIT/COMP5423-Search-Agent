---
name: multi-clue-decomposition
description: Use when a question identifies an entity through multiple independent clues, dates, relationships, or constraints that must all hold.
---

# Multi-clue decomposition

## Purpose

Use this procedure when the question identifies one person, organization, work, event,
or other entity through several independent clues. Typical clues include a date,
occupation, relationship, location, title, quotation, or sequence of events. Do not use
this skill for a simple fact lookup with one distinctive clue.

## Operating procedure

1. Parse the question into a numbered constraint table. Mark each clue as:
   - **identifying**: useful for finding candidates;
   - **verification**: must be proved before answering;
   - **context**: useful background but not required.
   Preserve dates, negations, and relationship direction exactly.
2. Call `retrieve` with only the most distinctive identifying clue. Prefer a rare name,
   exact phrase, unusual title, or specific date over a long query containing every clue.
   The purpose of this call is candidate generation, not final proof.
3. Use `open_document` or `inspect_document` on promising results. Record each candidate
   and the document ID supporting it. Reject a candidate immediately if the source clearly
   conflicts with a required constraint.
4. For each remaining required clue, search using the candidate's name plus that one clue.
   Change only one search dimension at a time. Use `find_in_document` when the candidate
   document is known and an exact phrase, date, or relationship needs locating.
5. Maintain an evidence matrix: one row per required constraint, with document IDs and
   the exact supporting passage. A shared keyword, approximate date, or search-result
   co-occurrence is not confirmation.
6. If a candidate fails one constraint, return to candidate generation or search the
   missing constraint with an alternative spelling or synonym. Do not silently relax a
   required clue.
7. Answer only when every essential constraint is supported by evidence. Cite the
   document IDs, distinguish direct evidence from an inference, and state any unresolved
   ambiguity. If no candidate satisfies all constraints, say that the evidence is
   insufficient rather than guessing.

## Query and budget discipline

- Never repeat an unchanged query.
- Prefer several focused searches to one oversized query.
- Do not treat snippets as proof when the source document can be opened.
- Stop when one candidate satisfies all required constraints or when remaining searches
  no longer add evidence. Report the unresolved constraints if stopping early.

## DeepSeek-V4-Flash decision rule

Select this skill when the user asks “who/which entity” and gives multiple clues that
must jointly identify the answer. Do not load it merely because a question contains
multiple sentences. After the evidence matrix is complete, unload the skill before an
unrelated follow-up search.
