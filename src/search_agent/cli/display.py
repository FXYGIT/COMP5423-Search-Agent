"""终端展示层：只消费消息和状态，不决定 Agent 的检索、技能或规划行为。"""

import json
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from search_agent.core.state import SearchState


class TerminalView:
    """工具与状态使用 Rich 面板；模型文本逐段原样输出，保证真实流式展示。

    plain 关闭面板和颜色，适合重定向；verbose 显示完整工具输出。
    默认仅截短终端预览，原始 ToolMessage 和传给模型的内容不变。
    """

    def __init__(self, *, plain: bool = False, verbose: bool = False) -> None:
        self.plain = plain
        self.verbose = verbose
        self.console = Console(highlight=False, markup=False)
        self.text_started = False

    def panel(self, title: str, content: str, style: str = "cyan") -> None:
        """显示一个已转义的文字面板；工具内容不解释为 Rich markup。"""
        if self.plain:
            print(f"\n[{title}]\n{content}", flush=True)
        else:
            self.console.print(Panel(Text(content), title=Text(title), border_style=style))

    def welcome(self) -> None:
        self.panel(
            "SEARCH AGENT · LangChain",
            "连续追问 · 流式回答 · 工具过程 · 任务规划\n"
            "/help 查看命令   /clear 清空会话   /exit 退出\n"
            "骨架模式：检索、文档、技能执行与摘要算法待实现。",
        )

    def token(self, text: str) -> None:
        """收到一段就输出一段，不等待模型生成结束。"""
        if not self.text_started:
            if self.plain:
                print("\n助手：")
            else:
                self.console.print(Text("\n助手", style="bold cyan"))
            self.text_started = True
        print(text, end="", flush=True)

    def end_text(self) -> None:
        if self.text_started:
            print(flush=True)
        self.text_started = False

    def tool_call(self, name: str, args: dict[str, Any]) -> None:
        """完整参数就绪时、工具执行前展示，避免解析不完整 JSON chunk。"""
        self.panel(f"调用工具 · {name}", json.dumps(args, ensure_ascii=False, indent=2))

    def tool_result(self, name: str, content: str, *, failed: bool = False) -> None:
        """区分真实工具错误、未实现状态和正常结果，长结果仅截短显示。"""
        status = "结果"
        try:
            payload = json.loads(content)
        except (ValueError, TypeError):
            payload = None
        if isinstance(payload, dict) and payload.get("status") == "not_implemented":
            status = "未实现 · not_implemented"
        elif failed:
            status = "执行失败"
        if not self.verbose and len(content) > 1600:
            content = content[:1600] + "\n…（展示已截短；--verbose 显示完整结果）"
        self.panel(f"工具{status} · {name}", content, "yellow" if status != "结果" else "green")

    def todos(self, state: SearchState) -> None:
        """显示真实 state.todos；未规划时不虚构任务。"""
        items = state.get("todos", [])
        labels = {"pending": "待处理", "in_progress": "进行中", "completed": "已完成"}
        lines = [f"{i}. [{labels[t['status']]}] {t['content']}" for i, t in enumerate(items, 1)]
        self.panel("任务计划", "\n".join(lines) or "尚未创建任务计划。")

    def skills(self, state: SearchState) -> None:
        self.panel("当前技能", state.get("active_skill") or "未激活技能；技能加载仍为骨架。")

    def evidence(self, state: SearchState) -> None:
        """显示结构化证据及来源；接受内存模型和序列化后的字典形式。"""
        evidence = state.get("selected_evidence", [])
        if not evidence:
            self.panel("支持证据", "暂无已收集证据；文档与检索仍为骨架。")
            return
        table = Table(title="支持证据", expand=True)
        for label in ("文档", "原文位置", "证据", "来源"):
            table.add_column(label)
        for item in evidence:
            data = item if isinstance(item, dict) else item.model_dump()
            fields = (
                str(data["doc_id"]),
                f"{data['start']}:{data['end']}",
                str(data["text"]),
                str(data.get("source_url") or "—"),
            )
            if self.plain:
                print(" | ".join(fields), flush=True)
            else:
                table.add_row(*(Text(value) for value in fields))
        if not self.plain:
            self.console.print(table)

    def status(self, state: SearchState) -> None:
        """未实现的统计显示“未统计”，避免伪报 0 token 或 0 次工具调用。"""
        self.panel(
            "会话状态",
            f"消息：{len(state.get('messages', []))}\n"
            f"任务：{len(state.get('todos', []))}\n"
            f"技能：{state.get('active_skill') or '无'}\n"
            f"摘要：{state.get('summary') or '未生成（摘要算法待实现）'}\n"
            f"工具调用：{state.get('tool_calls', '未统计')}\n"
            f"API tokens：{state.get('total_tokens', '未统计')}",
        )

    def help(self) -> None:
        self.panel(
            "命令",
            "/help       查看命令\n/clear      清空完整会话状态\n"
            "/todos      查看任务计划\n/skills     查看当前技能\n"
            "/evidence   查看支持证据\n/status     查看状态与统计\n"
            "/exit       退出（也支持 exit / quit）\n"
            "生成中 Ctrl+C 中断当前轮；输入时 Ctrl+C / Ctrl+D 退出。",
        )
