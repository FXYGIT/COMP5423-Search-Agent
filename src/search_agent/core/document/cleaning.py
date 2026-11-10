"""固定语料的清洗：剥离元数据头、重复标题与导航/页脚噪声。

清洗规则刻意保守：只删除"明确匹配模板"的行，并做字符保留率保护——
清洗后正文少于原正文 50% 时回退到原文，避免误删整篇内容。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

HEADER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.S)

# 明确的导航/页脚/账号类模板行。
# 注意：不包含 "related articles / read more / click here / about us" 这类行——
# 它们经常包含实体名，属于可用信息，误删会损失证据。
BOILERPLATE = re.compile(
    r"(?:"
    r"log ?in|sign ?in|sign ?up|create (?:an )?account|my account|account (?:info|settings)|"
    r"dashboard|publications|logged in as|user ?name|password|"
    r"cookie(?:s)?(?: policy)?|privacy policy|terms (?:of )?(?:service|use)|"
    r"all rights reserved|copyright\s*©|©\s*\d{4}|"
    r"subscribe|newsletter|share (?:this|on)|follow us|advertisement|sponsored|"
    r"skip to (?:main )?content|"
    r"you are here|breadcrumb|site ?map|accessibility|contact us"
    r")",
    re.I,
)
ONLY_URL = re.compile(r"^\s*(?:https?://|www\.)\S+\s*$", re.I)
MENU_SEPARATORS = re.compile(r"[|›»]")
# 只有含导航词汇的管道分隔行才当作菜单，避免误删表格数据。
NAV_WORDS = re.compile(
    r"\b(?:home|about|contact|login|log in|sign in|sign up|search|menu|site ?map|"
    r"help|privacy|terms|subscribe|newsletter|archive|sections?|topics?)\b",
    re.I,
)
SENTENCE_PUNCT = re.compile(r"[.!?]")
ZERO_WIDTH = re.compile(r"[\u200b-\u200d\ufeff]")
MULTI_BLANK = re.compile(r"\n{3,}")


@dataclass
class CleanedDocument:
    """清洗结果：结构化字段与正文。"""

    title: str
    author: str
    date: str
    text: str
    changed: bool
    removed_ratio: float


def parse_fields(text: str) -> tuple[dict[str, str], str]:
    """解析 ``--- key: value ---`` 头，返回字段字典与正文。

    支持 title / date / author 等任意 ``key: value`` 行；没有头时返回空字典。
    """
    match = HEADER.match(text)
    if not match:
        return {}, text
    fields: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().lower()
        value = value.strip()
        if key and value:
            fields[key] = value
    return fields, text[match.end() :]


def parse_header(text: str) -> tuple[str, str, str]:
    """兼容旧接口：返回 (title, date, body)。"""
    fields, body = parse_fields(text)
    return fields.get("title", ""), fields.get("date", ""), body


def _looks_like_menu(line: str) -> bool:
    """判定菜单/面包屑行；表格行（以 | 开头）一律保留。"""
    stripped = line.strip()
    if not stripped or len(stripped) > 120:
        return False
    if stripped.startswith("|"):
        return False
    if SENTENCE_PUNCT.search(stripped):
        return False
    if "›" in stripped or "»" in stripped:
        return True
    if len(MENU_SEPARATORS.findall(stripped)) >= 2:
        return bool(NAV_WORDS.search(stripped))
    return False


def _is_boilerplate(line: str) -> bool:
    """判定导航/页脚行：短行允许任意位置命中，较长行只接受行首命中。"""
    stripped = line.strip()
    if not stripped:
        return False
    if len(stripped) <= 80:
        return bool(BOILERPLATE.search(stripped))
    if len(stripped) <= 160:
        return bool(BOILERPLATE.match(stripped))
    return False


def _normalise_title(title: str) -> str:
    return re.sub(r"\s+", " ", title).strip().lower()


def remove_title_repeat(title: str, body: str) -> str:
    """删除正文开头重复的标题行（仅当前 3 行内高度相似）。"""
    if not title:
        return body
    target = _normalise_title(title)
    lines = body.splitlines()
    removed = 0
    index = 0
    while index < len(lines) and index < 6:
        line = _normalise_title(lines[index])
        if not line:
            index += 1
            continue
        if line == target or (len(line) > 20 and line in target):
            del lines[index]
            removed += 1
            continue
        index += 1
    return "\n".join(lines) if removed else body


def clean_document(text: str, *, remove_boilerplate: bool = False) -> CleanedDocument:
    """执行清洗；返回结构化字段、正文与统计。

    默认只做"无损"处理：解析元数据头、删除正文开头重复的标题、规范空白。
    行级导航/页脚过滤只在 ``remove_boilerplate=True`` 时启用——它可能删掉
    含实体名的页脚/链接行，必须作为独立实验验证。
    """
    fields, body = parse_fields(text)
    title = fields.get("title", "")
    author = fields.get("author", "")
    date = fields.get("date", "")
    original_body = body
    body = ZERO_WIDTH.sub("", body)
    body = remove_title_repeat(title, body)

    if remove_boilerplate:
        kept: list[str] = []
        for line in body.splitlines():
            stripped = line.strip()
            if not stripped:
                kept.append("")
                continue
            if ONLY_URL.match(stripped):
                continue
            if _looks_like_menu(stripped):
                continue
            if _is_boilerplate(stripped):
                continue
            kept.append(line.rstrip())
        body = "\n".join(kept)

    cleaned = MULTI_BLANK.sub("\n\n", body).strip()

    # 保护：清洗后正文不足原来一半，视为误删，回退到原文。
    if len(original_body) > 0 and len(cleaned) < 0.5 * len(original_body):
        cleaned = original_body.strip()

    removed_ratio = 1.0 - (len(cleaned) / len(original_body) if original_body else 1.0)
    return CleanedDocument(
        title=title,
        author=author,
        date=date,
        text=cleaned,
        changed=cleaned != original_body.strip(),
        removed_ratio=removed_ratio,
    )


def clean_passage(text: str, *, remove_boilerplate: bool = True) -> str:
    """清洗单个段落/片段（不含完整文档头），供建库编码前调用。"""
    text = ZERO_WIDTH.sub("", text)
    fields, body = parse_fields(text)
    body = remove_title_repeat(fields.get("title", ""), body)
    if remove_boilerplate:
        kept: list[str] = []
        for line in body.splitlines():
            stripped = line.strip()
            if not stripped:
                kept.append("")
                continue
            if ONLY_URL.match(stripped) or _looks_like_menu(stripped) or _is_boilerplate(stripped):
                continue
            kept.append(line.rstrip())
        body = "\n".join(kept)
    return MULTI_BLANK.sub("\n\n", body).strip()


def embedding_text(
    title: str, text: str, *, with_title: bool, author: str = "", date: str = ""
) -> str:
    """构造用于编码的文本；是否前置标题/日期由实验开关决定。"""
    header_parts = []
    if with_title and title:
        header_parts.append(title)
    if date:
        header_parts.append(date)
    if author:
        header_parts.append(author)
    if not header_parts:
        return text
    return "\n".join(header_parts) + "\n" + text
