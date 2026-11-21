"""扩展工具入口；业务工具返回占位结果，原生 write_todos 由 middleware 注册。

文档与检索工具由工厂函数创建，并绑定共享的 DocumentStore / DocumentReader / 检索后端。
"""

from .find_in_document import create_find_in_document_tool
from .inspect_document import create_inspect_document_tool
from .load_skill import load_skill
from .open_document import create_open_document_tool
from .retrieve import create_retrieve_tool
from .skill_resources import list_skills, read_skill_resource, run_skill_script, unload_skill

__all__ = [
    "create_retrieve_tool",
    "create_open_document_tool",
    "create_find_in_document_tool",
    "create_inspect_document_tool",
    "load_skill",
    "list_skills",
    "read_skill_resource",
    "run_skill_script",
    "unload_skill",
]
