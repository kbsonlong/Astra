"""会议逐字稿的确定性纠错和受限 LLM 候选配置。"""
from __future__ import annotations

from collections.abc import Iterable, Mapping


# 只收录已经人工确认过的 ASR 误识别，不做泛化同音词替换。
DEFAULT_CORRECTION_RULES: dict[str, str] = {
    "后视网络模式": "host 网络模式",
    "下限了五台机器": "下线了五台机器",
    "用更长的资源支撑更多的业务": "用更少的资源支撑更多的业务",
    "一个谷歌表哥": "一个 Google 表格",
    "你可以嫁给豆包": "你可以交给豆包",
    "后视": "host",
    "单科": "单机",
    "多科": "多机",
    "集群机": "集群",
    "宗师": "忠思",
    "下限": "下线",
    "冷资源中心": "云资源中心",
    "光单": "关单",
    "空单": "工单",
    "谷歌表哥": "Google 表格",
    "嫁给豆包": "交给豆包",
}


def apply_text_rules(text: str, rules: Mapping[str, str]) -> str:
    """按最长词优先做确定性替换。"""
    corrected = text
    for source, target in sorted(rules.items(), key=lambda item: len(item[0]), reverse=True):
        if source:
            corrected = corrected.replace(source, target)
    return corrected


def parse_correction_rules(values: Iterable[str]) -> dict[str, str]:
    """解析配置中的 ``source->target`` 候选映射。"""
    rules: dict[str, str] = {}
    for value in values:
        if "->" in value:
            source, target = value.split("->", 1)
        elif "=>" in value:
            source, target = value.split("=>", 1)
        else:
            continue
        source, target = source.strip(), target.strip()
        if source and target:
            rules[source] = target
    return rules


def candidate_outputs(text: str, rules: Mapping[str, str]) -> set[str]:
    """生成仅允许候选词替换的结果集合，用于校验 LLM 输出。"""
    outputs = {text}
    for source, target in rules.items():
        if source and source in text:
            outputs.update(output.replace(source, target) for output in tuple(outputs))
    return outputs
