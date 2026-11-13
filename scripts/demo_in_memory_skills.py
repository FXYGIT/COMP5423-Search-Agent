"""Run the skills module against two in-memory documents.

This demo does not require the BrowseComp-Plus corpus or validation set. It uses
the LLM configured in .env and lets the model decide whether to load a skill.
"""

from __future__ import annotations

import json
from typing import Any

from langchain.tools import tool
from langchain_core.messages import HumanMessage

from search_agent.core import create_search_agent
from search_agent.core.config import get_config
from search_agent.core.tools import (
    list_skills,
    load_skill,
    read_skill_resource,
    run_skill_script,
    unload_skill,
)

DOCUMENTS = {
    "doc-ada": {
        "title": "The Northbridge Learning Trust",
        "text": (
            "In 1998, Dr. Ada Mercer served as deputy director of the Northbridge "
            "Public Library. In 2004, Mercer and physicist Leo Hart founded the "
            "Northbridge Learning Trust, a nonprofit that organizes public science "
            "lectures and school laboratory grants."
        ),
    },
    "doc-leo": {
        "title": "Leo Hart and public science",
        "text": (
            "Physicist Leo Hart collaborated with Dr. Ada Mercer when they founded "
            "the Northbridge Learning Trust in 2004. The trust supports school "
            "laboratories and public education."
        ),
    },
}


@tool
def retrieve(query: str) -> dict[str, Any]:
    """Search the two in-memory documents and return matching evidence."""
    terms = {term.lower() for term in query.split() if len(term) > 2}
    results = []
    for doc_id, document in DOCUMENTS.items():
        haystack = f"{document['title']} {document['text']}".lower()
        matches = sum(term in haystack for term in terms)
        if matches:
            results.append(
                {
                    "doc_id": doc_id,
                    "title": document["title"],
                    "score": matches,
                    "snippet": document["text"],
                }
            )
    results.sort(key=lambda result: result["score"], reverse=True)
    return {"status": "ok", "result": results}


@tool
def open_document(doc_id: str) -> dict[str, Any]:
    """Read one in-memory document by ID."""
    document = DOCUMENTS.get(doc_id)
    if document is None:
        return {"status": "error", "message": f"Unknown document: {doc_id}", "result": None}
    return {"status": "ok", "result": {"doc_id": doc_id, **document}}


def print_trajectory(state: dict[str, Any]) -> None:
    print("\n========== TRAJECTORY ==========")
    for index, entry in enumerate(state.get("trajectory", []), start=1):
        data = entry.model_dump() if hasattr(entry, "model_dump") else entry
        role = data.get("role")
        print(f"\n[{index}] {role}")
        if role == "assistant":
            for call in data.get("tool_calls") or []:
                function = call["function"]
                print(f"TOOL CALL: {function['name']}")
                print(f"ARGUMENTS: {function['arguments']}")
            if data.get("content"):
                print(f"ANSWER: {data['content']}")
        elif role == "tool":
            print(f"TOOL RESULT ({data.get('name')}): {data.get('content', '')[:800]}")
        elif role == "system" and "Active skill:" in data.get("content", ""):
            print("ACTIVE SKILL INJECTED")
            print(data["content"][:800])

    print("\n========== STATE ==========")
    print(
        json.dumps(
            {
                "active_skill": state.get("active_skill"),
                "tool_calls": state.get("tool_calls"),
                "total_tokens": state.get("total_tokens"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    config = get_config()
    if not (config.llm.api_key.get_secret_value() and config.llm.base_url and config.llm.model):
        raise SystemExit("Missing LLM__API_KEY, LLM__BASE_URL, or LLM__MODEL in .env.")

    question = (
        "This is a required multi-clue investigation. Do not answer from the first "
        "search result. Before using retrieve, call list_skills, then load the "
        "multi-clue-decomposition skill because the answer depends on independent "
        "date, role, collaboration, and organization-purpose clues. After loading "
        "the skill, follow its procedure: search focused clues, open both documents, "
        "verify every constraint separately, and cite the document IDs. The person "
        "held a public library leadership role in 1998, later co-founded an "
        "organization in 2004 with physicist Leo Hart, and that organization "
        "supports school laboratories and public science education. Do not provide "
        "the final answer until all clues are verified."
    )
    system_prompt = (
        "You are testing the skills module with two in-memory documents. "
        "Use list_skills and load_skill before retrieve for this deliberately "
        "multi-clue question. A valid demo run must show a real load_skill call with "
        "the exact name multi-clue-decomposition, followed by retrieve and "
        "open_document. Do not claim a skill was used unless the trajectory shows "
        "the call and an injected active-skill system message. "
        "After verification, answer briefly with document IDs."
    )

    agent = create_search_agent(
        config=config,
        tools=[
            retrieve,
            open_document,
            load_skill,
            list_skills,
            read_skill_resource,
            run_skill_script,
            unload_skill,
        ],
        system_prompt=system_prompt,
    )
    state = agent.invoke({"messages": [HumanMessage(content=question)]})
    print_trajectory(state)


if __name__ == "__main__":
    main()
