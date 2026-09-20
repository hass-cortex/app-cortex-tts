"""Running out of device memory is recognised in every spelling the runtime has.

An ONNX Runtime arena never gives memory back short of losing the session, so
the one thing that helps is dropping the engine — and only a failure that is
recognised gets that. Measured on a 4 GB GTX 1650 with three models resident:
cuBLAS and cuDNN, not the arena, were what failed, and the engine was kept.
"""

from __future__ import annotations

import pytest

from cortex_speech.device import exhausted


class TestExhaustionIsRecognised:
    @pytest.mark.parametrize(
        "message",
        [
            "Failed to allocate memory for requested buffer of size 1048576",
            "CUDA out of memory",
            "CUBLAS failure 3: the resource allocation failed ; GPU=0",
            "CUDNN failure 4000: CUDNN_STATUS_INTERNAL_ERROR ; GPU=0",
            "CUDNN failure 2: CUDNN_STATUS_ALLOC_FAILED ; GPU=0",
        ],
    )
    def test_each_spelling_the_runtime_uses(self, message: str) -> None:
        assert exhausted(RuntimeError(message))

    def test_through_the_chain_the_runtime_wraps_it_in(self) -> None:
        inner = RuntimeError("CUBLAS failure 3: the resource allocation failed")
        outer = RuntimeError("stream failed")
        outer.__cause__ = inner
        assert exhausted(outer)

    def test_a_failure_that_is_not_memory_is_not(self) -> None:
        assert not exhausted(
            RuntimeError("Non-zero status code returned while running Add")
        )
