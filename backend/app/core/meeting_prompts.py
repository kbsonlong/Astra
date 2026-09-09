"""Built-in system prompt templates for meeting minutes generation."""
from __future__ import annotations

import json
import os
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MeetingPromptTemplate:
    id: str
    name: str
    description: str
    chunk_system_prompt: str
    merge_system_prompt: str


DEFAULT_MEETING_PROMPT_ID = "standard"

MEETING_PROMPT_TEMPLATES: tuple[MeetingPromptTemplate, ...] = (
    MeetingPromptTemplate(
        id="standard",
        name="标准纪要",
        description="提取会议结论、明确行动项和待确认事项，适合常规周会。",
        chunk_system_prompt=(
            "你是严谨的会议纪要助手。基于带说话人标签(S1/S2/...)的逐字稿片段，"
            "只提取本片段明确说出的信息，不补充常识，不修正无法确认的专有名词，不臆造事实。"
            "区分‘已确认结论’、‘明确提出的行动项’和‘讨论中/待确认事项’："
            "只有明确承诺、明确要求或明确决定的内容才能列为行动项；没有明确负责人的行动项写‘负责人：待确认’，"
            "不要根据谁说得多、谁赞同来推断负责人。保留原始 S1/S2 标签和原文中的人名。"
            "同一事项只能出现一次：已列为行动项的内容不要再放入关键结论或待确认事项；"
            "关键结论只写已经形成的结论，不写‘需要做/应当做’的行动句。"
            "关键结论最多 8 条、明确行动项最多 8 条、待确认事项最多 5 条，优先保留最具体的内容。"
            "输出简体中文 markdown，严格使用以下三个小节："
            "### 关键结论、### 明确行动项、### 待确认事项。"
            "没有内容的小节写‘无’。不要输出发言人性格评价，不要使用 ``` 代码块。"
        ),
        merge_system_prompt=(
            "你是严谨的会议纪要总编。下面是同一场长会议分段生成的要点。"
            "请合并为一份简体中文 markdown 纪要，去除重复内容，但不得新增逐字稿中没有的事实、"
            "决定、日期、负责人、团队或工具名称。只有原文明确提出、明确承诺或明确决定的事项才保留为行动项；"
            "负责人不明确时写‘负责人：待确认’，不得根据 S1/S2 的发言多少进行推断。"
            "不确定或仅为讨论/建议的内容放入待确认事项。保留原始人名和 S1/S2 标签。"
            "同一事项只能归入一个栏目：会议要点只放已形成的结论，明确行动项只放明确要求/承诺，"
            "待确认事项只放尚未决定的问题；行动项不得重复出现在其他栏目，待确认事项也不得重复行动项。"
            "同义或高度相似的条目只保留一条，确认过的结论不能再次列为待确认事项。"
            "会议要点最多 8 条、明确行动项最多 8 条、待确认事项最多 5 条。"
            "发言概览只概括各 S# 实际讨论的主题，不评价性格，不虚构分工。"
            "严格只输出以下四个小节，不要输出 ``` 代码块，不要重复任何小节："
            "## 会议要点、## 明确行动项、## 待确认事项、## 发言概览。"
            "每个小节都必须存在；没有内容写‘无’。"
        ),
    ),
    MeetingPromptTemplate(
        id="decisions",
        name="决策与行动项",
        description="突出已定决策、负责人和截止时间，适合项目评审或推进会。",
        chunk_system_prompt=(
            "你是项目会议记录员。只依据带说话人标签(S1/S2/...)的逐字稿片段提取明确内容，"
            "不得补充、推断或创造事实。重点识别已经拍板的决策、明确承诺的行动项、负责人和截止时间。"
            "只有原文明确决定、要求或承诺的内容才能记录；负责人或截止时间不明确时写‘待确认’，"
            "不要根据发言人身份、发言多少或语气推断。保留人名和 S1/S2 标签。"
            "输出简体中文 markdown，严格使用三个小节："
            "### 已确认决策、### 行动项、### 风险与待确认。"
            "行动项使用‘事项｜负责人｜截止时间’格式；没有内容写‘无’。"
            "每个小节最多 8 条，去除重复，不要输出性格评价或 ``` 代码块。"
        ),
        merge_system_prompt=(
            "你是项目会议纪要负责人。请根据同一场会议的分段要点合并结果，只保留逐字稿明确支持的内容，"
            "不得新增事实、日期、负责人、团队或工具名称。优先突出拍板决策和可执行行动项；"
            "行动项使用‘事项｜负责人｜截止时间’，未知字段写‘待确认’，不得自行推断。"
            "讨论中的建议、未拍板方案和风险放入风险与待确认。相同事项只保留一条。"
            "严格只输出四个小节：## 会议决策、## 行动项、## 风险与待确认、## 发言概览。"
            "每节都必须存在，没有内容写‘无’；不要输出 ``` 代码块，不评价发言人。"
        ),
    ),
    MeetingPromptTemplate(
        id="concise",
        name="简洁摘要",
        description="压缩为少量核心信息，适合快速浏览或较短会议。",
        chunk_system_prompt=(
            "你是简洁的会议摘要助手。根据带说话人标签(S1/S2/...)的逐字稿片段，"
            "只记录明确说出的事实、结论和安排，不补充常识，不推断负责人，不修正不确定的专有名词。"
            "去除背景铺陈和重复表达，优先保留最重要的结果。"
            "输出简体中文 markdown，严格使用三个小节："
            "### 一句话摘要、### 关键事项、### 后续安排。"
            "总计最多 10 条，后续安排只有明确提出或承诺的事项才能记录；没有内容写‘无’，"
            "保留必要的人名和 S1/S2 标签，不要输出 ``` 代码块。"
        ),
        merge_system_prompt=(
            "你是会议摘要编辑。请把分段摘要压缩成一份便于快速阅读的简体中文 markdown。"
            "只保留输入中有明确依据的事实，不新增决定、日期、负责人或背景；删除重复和次要细节。"
            "后续安排只保留明确提出或承诺的事项，负责人不明确时写‘负责人：待确认’，不得推断。"
            "严格只输出三个小节：## 会议摘要、## 关键事项、## 后续安排。"
            "总计最多 12 条，每节都必须存在，没有内容写‘无’，不要输出 ``` 代码块。"
        ),
    ),
)

_TEMPLATES_BY_ID = {template.id: template for template in MEETING_PROMPT_TEMPLATES}


class MeetingPromptTemplateStore:
    """Persist user-defined templates separately from immutable built-ins."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()

    def _read_custom(self) -> list[MeetingPromptTemplate]:
        if not self.path.is_file():
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read meeting prompt templates: {self.path}") from exc
        if not isinstance(payload, list):
            raise ValueError("meeting prompt templates must be a JSON array")
        templates: list[MeetingPromptTemplate] = []
        for item in payload:
            if not isinstance(item, dict):
                raise ValueError("each meeting prompt template must be a JSON object")
            templates.append(
                self._make_custom(
                    template_id=str(item.get("id", "")),
                    name=str(item.get("name", "")),
                    description=str(item.get("description", "")),
                    chunk_system_prompt=str(item.get("chunk_system_prompt", "")),
                    merge_system_prompt=str(item.get("merge_system_prompt", "")),
                )
            )
        return templates

    @staticmethod
    def _make_custom(
        *,
        template_id: str,
        name: str,
        description: str,
        chunk_system_prompt: str,
        merge_system_prompt: str,
    ) -> MeetingPromptTemplate:
        if not template_id.startswith("custom-"):
            raise ValueError("custom meeting prompt template IDs must start with custom-")
        if not 1 <= len(name.strip()) <= 80:
            raise ValueError("template name must be between 1 and 80 characters")
        if len(description) > 200:
            raise ValueError("template description must be at most 200 characters")
        if not 1 <= len(chunk_system_prompt.strip()) <= 20_000:
            raise ValueError("chunk system prompt must be between 1 and 20000 characters")
        if not 1 <= len(merge_system_prompt.strip()) <= 20_000:
            raise ValueError("merge system prompt must be between 1 and 20000 characters")
        return MeetingPromptTemplate(
            id=template_id,
            name=name.strip(),
            description=description.strip(),
            chunk_system_prompt=chunk_system_prompt.strip(),
            merge_system_prompt=merge_system_prompt.strip(),
        )

    def _write_custom(self, templates: list[MeetingPromptTemplate]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump([template.__dict__ for template in templates], handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            os.replace(temporary, self.path)
        except Exception:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise

    def list_templates(self) -> list[MeetingPromptTemplate]:
        return [*MEETING_PROMPT_TEMPLATES, *self._read_custom()]

    def get(self, template_id: str) -> MeetingPromptTemplate:
        if template_id in _TEMPLATES_BY_ID:
            return _TEMPLATES_BY_ID[template_id]
        for template in self._read_custom():
            if template.id == template_id:
                return template
        raise ValueError(f"unknown meeting prompt template: {template_id!r}")

    def create(
        self,
        *,
        name: str,
        description: str,
        chunk_system_prompt: str,
        merge_system_prompt: str,
    ) -> MeetingPromptTemplate:
        templates = self._read_custom()
        template = self._make_custom(
            template_id=f"custom-{uuid.uuid4().hex[:12]}",
            name=name,
            description=description,
            chunk_system_prompt=chunk_system_prompt,
            merge_system_prompt=merge_system_prompt,
        )
        templates.append(template)
        self._write_custom(templates)
        return template

    def update(
        self,
        template_id: str,
        *,
        name: str,
        description: str,
        chunk_system_prompt: str,
        merge_system_prompt: str,
    ) -> MeetingPromptTemplate:
        if template_id in _TEMPLATES_BY_ID:
            raise ValueError("built-in meeting prompt templates are read-only")
        templates = self._read_custom()
        updated = self._make_custom(
            template_id=template_id,
            name=name,
            description=description,
            chunk_system_prompt=chunk_system_prompt,
            merge_system_prompt=merge_system_prompt,
        )
        for index, template in enumerate(templates):
            if template.id == template_id:
                templates[index] = updated
                self._write_custom(templates)
                return updated
        raise ValueError(f"unknown meeting prompt template: {template_id!r}")

    def delete(self, template_id: str) -> None:
        if template_id in _TEMPLATES_BY_ID:
            raise ValueError("built-in meeting prompt templates are read-only")
        templates = self._read_custom()
        remaining = [template for template in templates if template.id != template_id]
        if len(remaining) == len(templates):
            raise ValueError(f"unknown meeting prompt template: {template_id!r}")
        self._write_custom(remaining)


def get_meeting_prompt_template(
    template_id: str,
    *,
    custom_templates_path: str | Path | None = None,
) -> MeetingPromptTemplate:
    """Return a selected template or raise ValueError for an unknown ID."""
    normalized = template_id.strip()
    if custom_templates_path is not None:
        try:
            return MeetingPromptTemplateStore(custom_templates_path).get(normalized)
        except ValueError as exc:
            choices = ", ".join(template.id for template in MeetingPromptTemplateStore(custom_templates_path).list_templates())
            raise ValueError(
                f"unknown meeting prompt template: {template_id!r}; choices: {choices}"
            ) from exc
    try:
        return _TEMPLATES_BY_ID[normalized]
    except KeyError as exc:
        choices = ", ".join(_TEMPLATES_BY_ID)
        raise ValueError(
            f"unknown meeting prompt template: {template_id!r}; choices: {choices}"
        ) from exc


def list_meeting_prompt_templates(
    custom_templates_path: str | Path | None = None,
) -> list[dict[str, object]]:
    """Serialize built-in and custom templates for the editor UI."""
    templates = (
        MeetingPromptTemplateStore(custom_templates_path).list_templates()
        if custom_templates_path is not None
        else list(MEETING_PROMPT_TEMPLATES)
    )
    return [
        {
            "id": template.id,
            "name": template.name,
            "description": template.description,
            "chunk_system_prompt": template.chunk_system_prompt,
            "merge_system_prompt": template.merge_system_prompt,
            "builtin": template.id in _TEMPLATES_BY_ID,
            "editable": template.id not in _TEMPLATES_BY_ID,
        }
        for template in templates
    ]
