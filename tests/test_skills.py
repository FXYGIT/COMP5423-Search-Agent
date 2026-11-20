from __future__ import annotations

import time
from pathlib import Path

import pytest

from search_agent.core.middlewares.skills import SkillMiddleware
from search_agent.core.skills import SkillLoader, SkillScriptRunner
from search_agent.core.tools import load_skill, unload_skill


def write_skill(root: Path, name: str = "demo") -> Path:
    skill = root / name
    (skill / "scripts").mkdir(parents=True)
    (skill / "references").mkdir()
    (skill / "assets").mkdir()
    (skill / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: A test skill\n---\n\n# {name}\n\nFollow this."
    )
    return skill


def test_discover_parses_frontmatter_without_loading_body(tmp_path: Path) -> None:
    skill = write_skill(tmp_path)
    (skill / "SKILL.md").write_text(
        "---\nname: demo\ndescription: Find useful evidence\n---\n\nSECRET BODY"
    )

    discovered = SkillLoader(tmp_path).discover()

    assert discovered[0].name == "demo"
    assert discovered[0].description == "Find useful evidence"
    assert not hasattr(discovered[0], "instructions")


def test_load_and_resources_are_read_on_demand(tmp_path: Path) -> None:
    skill = write_skill(tmp_path)
    (skill / "references" / "notes.txt").write_text("reference text")
    (skill / "assets" / "image.bin").write_bytes(b"\x00\xff")
    loader = SkillLoader(tmp_path)

    assert "# demo" in loader.load("demo")
    resources = {resource.path: resource.kind for resource in loader.list_resources("demo")}
    assert resources["references/notes.txt"] == "reference"
    assert resources["assets/image.bin"] == "asset"
    assert loader.read_resource("demo", "references/notes.txt").content == "reference text"
    asset = loader.read_resource("demo", "assets/image.bin")
    assert asset.encoding == "base64"
    with pytest.raises(ValueError):
        loader.read_resource("demo", "../SKILL.md")


def test_script_runner_captures_arguments_and_times_out(tmp_path: Path) -> None:
    skill = write_skill(tmp_path)
    (skill / "scripts" / "echo.py").write_text(
        "import sys\nprint('|'.join(sys.argv[1:]), flush=True)\n"
    )
    (skill / "scripts" / "slow.py").write_text("import time\ntime.sleep(2)\nprint('finished')\n")
    runner = SkillScriptRunner(tmp_path)

    result = runner.run("demo", "scripts/echo.py", ["one", "two"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "one|two"
    started = time.monotonic()
    timeout = runner.run("demo", "scripts/slow.py", [], timeout_seconds=1)
    assert time.monotonic() - started < 2
    assert timeout.timed_out
    assert timeout.exit_code is None


def test_skill_switching_and_unloading_update_state() -> None:
    first = load_skill.invoke(
        {
            "type": "tool_call",
            "name": "load_skill",
            "args": {"name": "multi-clue-decomposition"},
            "id": "load-1",
        }
    )
    second = load_skill.invoke(
        {
            "type": "tool_call",
            "name": "load_skill",
            "args": {"name": "search-recovery"},
            "id": "load-2",
        }
    )
    unloaded = unload_skill.invoke(
        {
            "type": "tool_call",
            "name": "unload_skill",
            "args": {},
            "id": "unload-1",
        }
    )

    assert first.update["active_skill"] == "multi-clue-decomposition"
    assert second.update["active_skill"] == "search-recovery"
    assert second.update["skill_instructions"].startswith("# Search recovery")
    assert unloaded.update["active_skill"] is None
    assert unloaded.update["skill_instructions"] == ""


def test_middleware_catalog_and_active_instructions_are_request_scoped() -> None:
    middleware = SkillMiddleware()
    assert "multi-clue-decomposition" in middleware._catalog()
    assert "search-recovery" in middleware._catalog()
