"""Which execution provider was asked for, and which one arrived.

These are not the same question, and the gap between them is the whole reason
this module exists. Measured on a host with a GTX 1650:
`get_available_providers()` listed `CUDAExecutionProvider`, and creating a
session fell back to the CPU with nothing but a warning, because the wheel's
CUDA libraries were a major version ahead of the driver. A provider list is a
claim; a session is evidence.
"""

from __future__ import annotations

import pytest

from cortex_speech.providers import (
    EXECUTION_PROVIDERS,
    ProviderUnavailableError,
    in_use,
    requested,
    verify,
)


class _Session:
    def __init__(self, *providers: str) -> None:
        self._providers = list(providers)

    def get_providers(self) -> list[str]:
        return self._providers


CUDA = "CUDAExecutionProvider"
CPU = "CPUExecutionProvider"


class TestWhatToAskFor:
    def test_cpu_never_looks_for_a_gpu(self) -> None:
        assert requested("cpu") == "cpu"

    def test_cuda_asks_even_when_the_runtime_denies_having_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Someone who names cuda gets a real failure, not a quiet cpu."""
        monkeypatch.setattr("cortex_speech.providers._cuda_is_offered", lambda: False)
        assert requested("cuda") == "cuda"

    def test_auto_asks_only_when_the_runtime_offers_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("cortex_speech.providers._cuda_is_offered", lambda: True)
        assert requested("auto") == "cuda"
        monkeypatch.setattr("cortex_speech.providers._cuda_is_offered", lambda: False)
        assert requested("auto") == "cpu"

    def test_every_choice_is_a_declared_one(self) -> None:
        assert set(EXECUTION_PROVIDERS) == {"auto", "cpu", "cuda"}


class TestWhatArrived:
    def test_all_sessions_on_cuda_is_cuda(self) -> None:
        assert in_use([_Session(CUDA, CPU), _Session(CUDA, CPU)]) == "cuda"

    def test_one_session_that_fell_back_makes_it_cpu(self) -> None:
        """A graph that fell back while its neighbours did not is not a GPU
        deployment, it is a slow one wearing the label."""
        assert in_use([_Session(CUDA, CPU), _Session(CPU)]) == "cpu"

    def test_no_sessions_is_cpu_not_a_crash(self) -> None:
        """Asked before anything is loaded, which the health endpoint does."""
        assert in_use([]) == "cpu"


class TestRefusingASilentDowngrade:
    def test_cuda_that_became_cpu_raises(self) -> None:
        """The failure this module exists to make loud."""
        with pytest.raises(ProviderUnavailableError) as err:
            verify("cuda", "cpu")
        assert "onnxruntime-gpu" in str(err.value), "it should say what to check"

    def test_cuda_that_stayed_cuda_passes(self) -> None:
        assert verify("cuda", "cuda") == "cuda"

    @pytest.mark.parametrize("actual", ["cpu", "cuda"])
    def test_auto_accepts_whatever_arrived(self, actual: str) -> None:
        """That is what auto means; the health endpoint reports which."""
        assert verify("auto", actual) == actual

    def test_cpu_is_never_upgraded_behind_the_caller(self) -> None:
        assert verify("cpu", "cpu") == "cpu"


class TestReleasingSessions:
    """ONNX Runtime has no `close()`. What returns a CUDA arena is the C++
    session losing its last Python reference, and the wrapper holds several —
    so release nulls the same names the wrapper's own `_reset_session` does.
    Measured on a GTX 1650 with the collector off and the engine pinned by a
    cycle: 2480 MiB before, 110 after."""

    def test_the_names_are_the_ones_onnxruntime_itself_nulls(self) -> None:
        import inspect

        import onnxruntime as ort

        from cortex_speech.providers import _SESSION_REFERENCES

        source = inspect.getsource(ort.InferenceSession._reset_session)  # noqa: SLF001
        missing = [name for name in _SESSION_REFERENCES if name not in source]
        assert not missing, f"onnxruntime {ort.__version__} no longer holds {missing}"

    def test_a_released_session_says_so_when_used(self) -> None:
        from cortex_speech.providers import SessionClosedError, release_sessions

        class _Wrapper(_Session):
            def __init__(self) -> None:
                super().__init__(CUDA)
                self._sess = object()
                self._inputs_meta = ["x"]

            def run(self, *args: object) -> object:
                return self._sess.run(*args)  # type: ignore[attr-defined]

        class _Runtime:
            def __init__(self) -> None:
                self.sessions = {"a": _Wrapper(), "b": _Wrapper()}
                self.other = _Wrapper()

        runtime = _Runtime()
        assert release_sessions(runtime) == 3
        assert runtime.other._inputs_meta is None  # noqa: SLF001
        with pytest.raises(SessionClosedError):
            runtime.sessions["a"].run(None, {})
        assert release_sessions(runtime) == 0, "release must be idempotent"


class TestRunOptions:
    """Asked of the session, because what was requested is not what it got.

    Shrinking an arena the session does not have is an invalid argument and
    fails the run rather than being ignored, and a model with several graphs
    can have some on the GPU and some not.
    """

    def test_a_cuda_session_shrinks_its_arena(self) -> None:
        from cortex_speech.providers import run_options

        options = run_options(_Session(CUDA, CPU))
        assert options is not None
        assert (
            options.get_run_config_entry("memory.enable_memory_arena_shrinkage")
            == "gpu:0"
        )

    def test_a_session_that_fell_back_carries_nothing(self) -> None:
        from cortex_speech.providers import run_options

        assert run_options(_Session(CPU)) is None
