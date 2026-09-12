"""One cache, one invalidation rule.

Encoding a reference is the dominant cost of a cloned utterance, so both
cloning engines cache it. They used to cache it separately, with two different
rules for when an entry goes stale — the 80M compared the recording's
fingerprint, MOSS trusted `forget()` alone. A divergence like that is silent:
the wrong branch still produces audio, in the wrong voice.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from cortex_speech.engine.conditioning import ConditioningCache
from cortex_speech.references import Reference

SRC = Path(__file__).resolve().parent.parent / "src/cortex_speech/engine"


def _reference(fingerprint: str = "aaaa") -> Reference:
    return Reference(
        id="wanwan",
        name="灣灣",
        transcript="你好。",
        raw_transcript="你好",
        language="zh",
        gender="female",
        seconds=6.0,
        created=0.0,
        audio_path=Path("/nowhere/wanwan.wav"),
        fingerprint=fingerprint,
    )


class _Encoder:
    """Counts how often the expensive step actually runs."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, reference: Reference) -> str:
        self.calls += 1
        return f"codes-{reference.fingerprint}-{self.calls}"


class TestCaching:
    def test_the_first_call_encodes_and_the_second_does_not(self) -> None:
        cache: ConditioningCache[str] = ConditioningCache()
        encode = _Encoder()
        first = cache.get(_reference(), encode)
        assert cache.get(_reference(), encode) == first
        assert encode.calls == 1

    def test_references_do_not_share_an_entry(self) -> None:
        cache: ConditioningCache[str] = ConditioningCache()
        encode = _Encoder()
        other = Reference(**{**_reference().__dict__, "id": "other"})
        assert cache.get(_reference(), encode) != cache.get(other, encode)
        assert encode.calls == 2


class TestInvalidation:
    def test_a_replaced_recording_is_re_encoded(self) -> None:
        """A deleted reference re-uploaded under the same name reuses the id.

        Serving the previous voice under it is the failure that only the
        fingerprint check catches.
        """
        cache: ConditioningCache[str] = ConditioningCache()
        encode = _Encoder()
        before = cache.get(_reference("aaaa"), encode)
        after = cache.get(_reference("bbbb"), encode)
        assert after != before
        assert encode.calls == 2

    def test_editing_a_transcript_keeps_the_encoding(self) -> None:
        """The transcript reaches the model as text; re-encoding is waste."""
        cache: ConditioningCache[str] = ConditioningCache()
        encode = _Encoder()
        before = cache.get(_reference(), encode)
        edited = Reference(**{**_reference().__dict__, "transcript": "早安。"})
        assert cache.get(edited, encode) == before
        assert encode.calls == 1

    def test_forget_drops_the_entry(self) -> None:
        cache: ConditioningCache[str] = ConditioningCache()
        encode = _Encoder()
        cache.get(_reference(), encode)
        cache.forget("wanwan")
        cache.get(_reference(), encode)
        assert encode.calls == 2

    def test_forgetting_something_uncached_is_not_an_error(self) -> None:
        """`forget_reference` fans out to every loaded engine, hit or miss."""
        ConditioningCache[str]().forget("never-seen")


class TestNoEngineKeepsItsOwn:
    """The check that keeps the two rules from diverging again.

    Both engines cache by reference id. An engine that grows its own dict for
    it passes every other test here while restoring exactly the inconsistency
    this cache replaced, so the shape is pinned rather than the behaviour.
    """

    @pytest.mark.parametrize("module", ["clone.py", "moss.py"])
    def test_an_engine_with_reference_voices_uses_the_shared_cache(
        self, module: str
    ) -> None:
        source = (SRC / module).read_text(encoding="utf-8")
        assert "ConditioningCache" in source, f"{module} caches conditioning itself"

    @pytest.mark.parametrize("module", ["clone.py", "moss.py", "preset.py"])
    def test_no_engine_annotates_a_dict_keyed_by_reference_id(
        self, module: str
    ) -> None:
        """`self._x: dict[str, ...]` in an engine is the shape being banned."""
        tree = ast.parse((SRC / module).read_text(encoding="utf-8"))
        offenders = [
            ast.unparse(node)
            for node in ast.walk(tree)
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Attribute)
            and ast.unparse(node.annotation).startswith("dict[str,")
        ]
        assert not offenders, f"{module} keeps its own cache: {offenders}"
