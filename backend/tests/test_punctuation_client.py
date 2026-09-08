import pytest

from app.models.punctuation_client import (
    FunASRPunctuationClient,
    clean_repeated_punctuation,
)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("你好。。", "你好。"),
        ("真的吗？？", "真的吗？"),
        ("嗯，，嗯，。", "嗯，嗯。"),
        ("很好！！，继续", "很好！继续"),
        ("正文\n没有重复标点", "正文\n没有重复标点"),
    ],
)
def test_clean_repeated_punctuation(source: str, expected: str) -> None:
    assert clean_repeated_punctuation(source) == expected


@pytest.mark.anyio
async def test_funasr_punctuation_client_loads_once_and_extracts_text() -> None:
    loads: list[dict[str, object]] = []

    class Model:
        def generate(self, **kwargs: object) -> list[dict[str, str]]:
            return [{"text": "你好，，世界。。"}]

    def factory(**kwargs: object) -> Model:
        loads.append(kwargs)
        return Model()

    client = FunASRPunctuationClient(
        "ct-punc-c", device="mps", model_factory=factory
    )

    assert await client.restore("你好世界") == "你好，世界。"
    assert await client.restore("再见") == "你好，世界。"
    assert len(loads) == 1
    assert loads[0] == {
        "model": "ct-punc-c",
        "device": "mps",
        "disable_update": True,
    }
