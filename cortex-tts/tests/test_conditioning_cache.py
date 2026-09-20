"""One cache, one invalidation rule.

Encoding a reference is the dominant cost of a cloned utterance, so every
cloning engine caches it — in the one cache here, under the one rule for when
an entry goes stale. An engine keeping its own would be a second rule, and a
divergence between two is silent: the wrong branch still produces audio, in
the wrong voice.
"""

from __future__ import annotations

import ast
import io
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from cortex_speech.engine.conditioning import ConditioningCache
from cortex_speech.references import Reference, ReferenceStore

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
        """The encoding is of the audio alone; re-encoding is waste.

        An engine whose conditioning object also carries the transcript
        (OmniVoice) replaces that copy on use — see
        `test_transcript_refresh.py`.
        """
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

    @pytest.mark.parametrize("module", ["moss.py"])
    def test_an_engine_with_reference_voices_uses_the_shared_cache(
        self, module: str
    ) -> None:
        source = (SRC / module).read_text(encoding="utf-8")
        assert "ConditioningCache" in source, f"{module} caches conditioning itself"

    @pytest.mark.parametrize("module", ["moss.py", "preset.py"])
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


class _JsonSidecar:
    """The smallest thing that satisfies `Sidecar`."""

    suffix = ".fake.json"

    def dump(self, value: str, path: Path) -> None:
        path.write_text(json.dumps(value))

    def load(self, path: Path) -> str:
        return json.loads(path.read_text())


class TestOnDisk:
    """An unloaded engine that comes back must not pay the encode again."""

    def test_a_fresh_cache_reads_what_the_last_one_wrote(self, tmp_path) -> None:
        encode = _Encoder()
        first = ConditioningCache[str](_JsonSidecar(), tmp_path).get(
            _reference(), encode
        )
        again = ConditioningCache[str](_JsonSidecar(), tmp_path).get(
            _reference(), encode
        )
        assert again == first
        assert encode.calls == 1
        assert (tmp_path / "wanwan.aaaa.fake.json").is_file()

    def test_a_replaced_recording_misses_and_sweeps_the_old_file(
        self, tmp_path
    ) -> None:
        encode = _Encoder()
        ConditioningCache[str](_JsonSidecar(), tmp_path).get(_reference("aaaa"), encode)
        ConditioningCache[str](_JsonSidecar(), tmp_path).get(_reference("bbbb"), encode)
        assert encode.calls == 2
        assert [p.name for p in tmp_path.iterdir()] == ["wanwan.bbbb.fake.json"]

    def test_an_unreadable_file_is_replaced_not_fatal(self, tmp_path) -> None:
        (tmp_path / "wanwan.aaaa.fake.json").write_text("{not json")
        encode = _Encoder()
        value = ConditioningCache[str](_JsonSidecar(), tmp_path).get(
            _reference(), encode
        )
        assert encode.calls == 1
        assert json.loads((tmp_path / "wanwan.aaaa.fake.json").read_text()) == value

    def test_forget_removes_the_file_too(self, tmp_path) -> None:
        cache = ConditioningCache[str](_JsonSidecar(), tmp_path)
        cache.get(_reference(), _Encoder())
        cache.forget("wanwan")
        assert list(tmp_path.iterdir()) == []

    def test_only_this_engine_s_files_are_touched(self, tmp_path) -> None:
        other = tmp_path / "wanwan.aaaa.other.npz"
        other.write_bytes(b"")
        cache = ConditioningCache[str](_JsonSidecar(), tmp_path)
        cache.get(_reference(), _Encoder())
        cache.forget("wanwan")
        assert other.is_file()

    def test_without_a_directory_nothing_is_written(self, tmp_path) -> None:
        ConditioningCache[str](_JsonSidecar()).get(_reference(), _Encoder())
        assert list(tmp_path.iterdir()) == []


class TestTheStoreSweepsSidecars:
    """Removing a reference while no engine is loaded must not leave files."""

    def test_remove_takes_every_engine_s_sidecar_with_it(self, tmp_path) -> None:
        store = ReferenceStore(tmp_path)
        rate = 24000
        t = np.arange(int(rate * 2.5)) / rate
        clip = np.concatenate([0.2 * np.sin(2 * np.pi * 220 * t), np.zeros(rate // 2)])
        buffer = io.BytesIO()
        sf.write(buffer, clip.astype(np.float32), rate, format="WAV", subtype="PCM_16")
        reference = store.add(
            name="Wanwan", transcript="你好。", audio=buffer.getvalue()
        )
        for suffix in (".omni.pt", ".moss.json"):
            (
                store.root / f"{reference.id}.{reference.fingerprint}{suffix}"
            ).write_bytes(b"")
        neighbour = store.add(
            name="Wanwan", transcript="你好。", audio=buffer.getvalue()
        )
        (store.root / f"{neighbour.id}.{neighbour.fingerprint}.omni.pt").write_bytes(
            b""
        )

        assert store.remove(reference.id)

        left = sorted(p.name for p in store.root.iterdir())
        assert left == sorted(
            [
                "references.json",
                f"{neighbour.id}.wav",
                f"{neighbour.id}.{neighbour.fingerprint}.omni.pt",
            ]
        )
