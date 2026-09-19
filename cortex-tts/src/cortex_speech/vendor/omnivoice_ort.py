# pyright: reportMissingImports=false
"""Build OmniVoice with an ONNX graph where its transformer should be.

Upstream runs the whole pipeline in PyTorch. `rhasspy/omnivoice-onnx` publishes
the language model alone as a block-wise int4 ONNX export, and the wyoming-piper
backend it was made for uses it by replacing `OmniVoice.forward`. This module
does the same, and takes one step further: because `forward` is the only thing
that reads the transformer's weights, the module is built on the meta device
and the checkpoint's 2.45 GB of them are never downloaded or materialised.

What is still real, and has to be: the Higgs Audio V2 tokenizer that turns
codes into a waveform, the text tokenizer, and the feature extractor.

Measured on a 4-vCPU i7-9750H: peak resident 1.4 GB against 4.7 GB loading the
pipeline whole.

The pragma sits above this docstring, not below it, because pyright reads a
file-level one only before any other code — and a docstring is code. Below it,
as it was, the suppression is silently ignored. torch and transformers arrive
with the `omnivoice` extra, which CI does not install: a couple of gigabytes to
type-check 150 lines of glue. Nothing else is suppressed file-wide; this file is
ours, so every other rule and all of ruff still apply to it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..providers import CUDA_OPTIONS, run_options


def _cpu_omnivoice() -> type:
    """Return an OmniVoice whose `device` is cpu though its weights are not.

    Everything the generate path allocates — input ids, masks, the prompt —
    is placed on `self.device`, which reads the first parameter's device. With
    the transformer on meta that answers `meta`, and the first tensor handed to
    the ONNX graph raises "Cannot copy out of meta tensor".

    Importing the module is also what registers `omnivoice` with `AutoConfig`,
    so this has to happen before the checkpoint's config is read.
    """
    from .omnivoice.modeling import OmniVoice

    class CpuOmniVoice(OmniVoice):  # type: ignore[misc, valid-type]
        @property
        def device(self) -> torch.device:
            return torch.device("cpu")

    return CpuOmniVoice


def _onnx_forward(session: Any, options: Any = None) -> Any:
    """Return a `forward` that runs the exported graph.

    The graph takes a 2-D padding mask because its attention is bidirectional,
    while the pipeline builds a 4-D block mask — so the real-token mask is
    recovered from the block mask's rows: a key position is real when its row
    attends to more than itself.

    Takes no `self`, and is assigned to the instance as a plain function rather
    than bound with `types.MethodType`. A bound method stored on its own
    `__self__` is a reference cycle, and this closure holds the ONNX session,
    so the model's GPU arena would outlive the last name for it and wait for
    the collector — 618 MiB, on a card that has 4096.

    `options` carries the arena shrinkage on CUDA. Without it this session's
    arena only grows: measured on a 4 GB GTX 1650, a paced reply took it from
    690 MiB at load to 3690 in 38 seconds, then failed to place a 32 MiB
    buffer for a twelve-character request. Every runtime here asks for the
    shrinkage, and a card that is given back between runs is a card the next
    reply can still use.
    """
    from .omnivoice.modeling import OmniVoiceModelOutput

    accepted = {spec.name for spec in session.get_inputs()}

    def forward(
        input_ids: torch.Tensor,
        audio_mask: torch.Tensor,
        labels: Any = None,
        attention_mask: torch.Tensor | None = None,
        document_ids: Any = None,
        position_ids: torch.Tensor | None = None,
    ) -> Any:
        batch, _codebooks, length = input_ids.shape
        if attention_mask is not None and attention_mask.dim() == 4:
            mask = (attention_mask[:, 0, :, :].sum(-1) > 1).to(torch.int64)
        elif attention_mask is not None and attention_mask.dim() == 2:
            mask = attention_mask.to(torch.int64)
        else:
            mask = torch.ones(batch, length, dtype=torch.int64)
        if position_ids is None:
            position_ids = (
                torch.arange(length, dtype=torch.int64)
                .unsqueeze(0)
                .expand(batch, length)
                .contiguous()
            )
        feeds = {
            "input_ids": input_ids.cpu().numpy().astype(np.int64),
            "audio_mask": audio_mask.cpu().numpy().astype(bool),
            "attention_mask": mask.cpu().numpy().astype(np.int64),
            # Narrowed by the `is None` branch above. With torch
            # unresolved the checker cannot follow that, so it is told
            # here rather than for the whole file.
            "position_ids": position_ids.cpu()  # pyright: ignore[reportOptionalMemberAccess]
            .numpy()
            .astype(np.int64),
        }
        logits = session.run(
            ["logits"], {k: v for k, v in feeds.items() if k in accepted}, options
        )[0]
        return OmniVoiceModelOutput(logits=torch.from_numpy(logits))

    return forward


def load(
    directory: Path,
    *,
    onnx_model: str,
    audio_tokenizer_dir: str,
    execution_provider: str = "cpu",
    num_threads: int = 0,
) -> tuple[Any, Any]:
    """Assemble the hybrid model. Returns it and the ONNX session.

    The session is returned rather than hidden inside the model because
    `providers.sessions_of` has to be able to see it: an engine that reports a
    provider it did not get is the failure that module exists to prevent.
    """
    import onnxruntime as ort
    from transformers import (
        AutoConfig,
        AutoFeatureExtractor,
        AutoModel,
        AutoTokenizer,
        HiggsAudioV2TokenizerModel,
    )

    from .omnivoice.duration import RuleDurationEstimator

    omni_voice = _cpu_omnivoice()
    config = AutoConfig.from_pretrained(str(directory))
    # The transformer is never read: `forward` below replaces it outright.
    with torch.device("meta"):
        language_model = AutoModel.from_config(config.llm_config)
    model = omni_voice(config, llm=language_model)
    # Computed in `__init__` rather than read from the checkpoint, and the one
    # tensor the generate path uses outside `forward`, so it has to be real.
    model.codebook_layer_offsets = (
        torch.arange(config.num_audio_codebook) * config.audio_vocab_size
    )

    model.text_tokenizer = AutoTokenizer.from_pretrained(str(directory))
    tokenizer_path = str(directory / audio_tokenizer_dir)
    # No `device_map`: upstream passes one, which makes `transformers` demand
    # accelerate to place a model that is going on the CPU either way.
    model.audio_tokenizer = HiggsAudioV2TokenizerModel.from_pretrained(tokenizer_path)
    model.feature_extractor = AutoFeatureExtractor.from_pretrained(tokenizer_path)
    model.sampling_rate = model.feature_extractor.sampling_rate
    model.duration_estimator = RuleDurationEstimator()

    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if num_threads > 0:
        options.intra_op_num_threads = num_threads
    providers = (
        [("CUDAExecutionProvider", CUDA_OPTIONS), "CPUExecutionProvider"]
        if execution_provider == "cuda"
        else ["CPUExecutionProvider"]
    )
    session = ort.InferenceSession(str(directory / onnx_model), options, providers)
    model.forward = _onnx_forward(session, run_options(session))
    return model, session
