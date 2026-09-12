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
