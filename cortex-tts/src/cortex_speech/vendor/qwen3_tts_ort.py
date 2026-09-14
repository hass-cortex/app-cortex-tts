"""Qwen3-TTS over its exported ONNX sub-models.

Derived from `inference.py` in `onnx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice`
(Apache-2.0), which mirrors `Qwen3TTSForConditionalGeneration.generate` with
`non_streaming_mode=True` and was verified there against PyTorch at 100% greedy
parity. Unlike the other files in this directory it is not byte-for-byte
upstream, because upstream's artefact is a demo script rather than a runtime:

- the manifest is gone — the catalog entry names the bundle, and a manifest
  that also names a file this app deliberately does not download would fail
  the load,
- the execution provider is the caller's, not the manifest's,
- the tokenizer is handed in, because the checkpoint ships none assembled,
- generation yields frames instead of returning them, so audio can leave
  before the utterance is finished,
- the CLI, the self-test and the parity harnesses are not carried.

The sub-model signatures, the prompt layout and the sampling are upstream's and
are the part to diff when it moves.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SAMPLE_RATE = 24000

# The codec decoder is exported at a fixed length: 25 frames in, 2 seconds of
# audio out. It is also the streaming granularity, because a block decoded on
# its own is bit-identical to the same block inside a longer decode.
DECODER_FRAMES = 25

# One frame carries this many residual-quantiser codes, predicted one at a
# time because each is conditioned on the ones before it.
CODE_GROUPS = 16

# Files the engine opens, by the name this module knows them under. Every one
# is required; the optional two below are what separates the cloning
# checkpoint from the custom-voice one.
_SESSIONS = (
    "text_embed",
    "codec_embed",
    "residual_embed",
    "talker_cache",
    "code_predictor",
    "tok_decoder",
)
_CLONING_SESSIONS = ("tok_encoder", "speaker_encoder")

# Reference audio is encoded a second at a time: the encoder is exported at a
# fixed 24000-sample input.
_ENCODER_SAMPLES = SAMPLE_RATE


class Qwen3TtsError(RuntimeError):
    """The bundle cannot do what was asked of it."""


@dataclass(frozen=True)
class ReferenceConditioning:
    """A reference recording, in the three forms the clone prompt needs.

    Attributes:
        codes: The recording as codec frames, `[frames, CODE_GROUPS]`.
        x_vector: The speaker embedding, shaped `[1, 1, hidden]`.
        transcript: What the recording says, which the prompt reads beside it.
    """

    codes: np.ndarray
    x_vector: np.ndarray
    transcript: str


def _penalise(logits: np.ndarray, seen: list[int], penalty: float) -> np.ndarray:
    """Divide the logits of already-chosen codes, upstream's formulation."""
    if penalty == 1.0 or not seen:
        return logits
    index = np.array(sorted(set(seen)), dtype=np.int64)
    scores = logits[index]
    logits[index] = np.where(scores < 0, scores * penalty, scores / penalty)
    return logits


def _sample(
    logits: np.ndarray,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    rng: np.random.Generator,
) -> int:
    """Pick one code. `temperature <= 0` is greedy."""
    logits = logits.astype(np.float64)
    if temperature <= 0:
        return int(np.argmax(logits))
    logits = logits / temperature
    if top_k > 0:
        keep = min(top_k, logits.shape[-1])
        threshold = np.partition(logits, -keep)[-keep]
        logits = np.where(logits < threshold, -np.inf, logits)
    logits -= logits.max()
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum()
    if top_p < 1.0:
        order = np.argsort(probabilities)[::-1]
        cut = int(np.searchsorted(np.cumsum(probabilities[order]), top_p)) + 1
        mask = np.zeros_like(probabilities)
        mask[order[:cut]] = probabilities[order[:cut]]
        probabilities = mask / mask.sum()
    return int(rng.choice(len(probabilities), p=probabilities))


class Qwen3TtsOnnx:
    """The exported sub-models, wired into one text-to-speech pass."""

    def __init__(
        self,
        directory: Path,
        tokenizer: Any,
        *,
        onnx_subdir: str = "cpu_int4",
        execution_provider: str = "cpu",
        num_threads: int = 0,
    ) -> None:
        """Open every session in the bundle.

        Args:
            directory: The downloaded bundle.
            tokenizer: A `tokenizers.Tokenizer` for the checkpoint's text.
            onnx_subdir: Which exported variant the bundle carries. The graphs
                are identical across the export's `cpu_*` and `cuda_*`
                directories — only the manifest differs — so this names a
                precision, not a device.
            execution_provider: `cpu` or `cuda`.
            num_threads: ONNX Runtime intra-op threads; 0 lets it decide.
        """
        import onnxruntime as ort

        self._config = json.loads(
            (directory / "config.json").read_text(encoding="utf-8")
        )
        self._talker_config = self._config["talker_config"]
        self._tokenizer = tokenizer

        options = ort.SessionOptions()
        options.log_severity_level = 3
        if num_threads > 0:
            options.intra_op_num_threads = num_threads
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if execution_provider == "cuda"
            else ["CPUExecutionProvider"]
        )
        root = directory / onnx_subdir

        def open_session(name: str) -> Any:
            return ort.InferenceSession(str(root / f"{name}.onnx"), options, providers)

        self.sessions: dict[str, Any] = {name: open_session(name) for name in _SESSIONS}
        for name in _CLONING_SESSIONS:
            if (root / f"{name}.onnx").is_file():
                self.sessions[name] = open_session(name)

        # The KV cache is fed back under the talker's own input names, which
        # follow the three real inputs. Reading them off the graph keeps the
        # 28-layer count out of this file.
        self._past_names = [
            spec.name for spec in self.sessions["talker_cache"].get_inputs()
        ][3:]

    # -- what the bundle can do -------------------------------------------

    @property
    def model_type(self) -> str:
        """`custom_voice`, `voice_design` or `base`, as the checkpoint says."""
        return str(
            self._config.get("tts_model_type")
            or self._talker_config.get("tts_model_type")
            or "unknown"
        )

    @property
    def speakers(self) -> dict[str, int]:
        """Built-in speaker names and the codec ids that select them."""
        return dict(self._talker_config.get("spk_id") or {})

    @property
    def languages(self) -> dict[str, int]:
        """Language names the talker can be told to read as."""
        return dict(self._talker_config.get("codec_language_id") or {})

    @property
    def can_clone(self) -> bool:
        """Whether the bundle carries the encoders cloning needs."""
        return all(name in self.sessions for name in _CLONING_SESSIONS)

    # -- the sub-models ----------------------------------------------------

    def _embed_text(self, ids: Any) -> np.ndarray:
        return self.sessions["text_embed"].run(
            None, {"text_ids": np.asarray(ids, np.int64)}
        )[0]

    def _embed_codec(self, ids: Any) -> np.ndarray:
        return self.sessions["codec_embed"].run(
            None, {"codec_ids": np.asarray(ids, np.int64)}
        )[0]

    def _step_embed(self, codes: Any) -> np.ndarray:
        """Sum the group embeddings of one frame: the talker's next input."""
        return self.sessions["residual_embed"].run(
            None, {"codec_ids": np.asarray(codes, np.int64)}
        )[0]

    def _talker(
        self,
        inputs_embeds: np.ndarray,
        position_ids: np.ndarray,
        attention_mask: np.ndarray,
        past: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
        feed = {
            "inputs_embeds": inputs_embeds.astype(np.float32),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }
        feed.update(zip(self._past_names, past, strict=True))
        outputs = self.sessions["talker_cache"].run(None, feed)
        return outputs[0], outputs[1], list(outputs[2:])

    def _predict_group(self, hidden: np.ndarray, codes: np.ndarray) -> np.ndarray:
        return self.sessions["code_predictor"].run(
            None,
            {
                "talker_hidden": hidden.astype(np.float32),
                "codec_ids": np.asarray(codes, np.int64),
            },
        )[0]

    def _ids(self, text: str) -> np.ndarray:
        return np.asarray(
            [self._tokenizer.encode(text, add_special_tokens=False).ids], np.int64
        )

    # -- decoding ----------------------------------------------------------

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Turn exactly `DECODER_FRAMES` frames into mono float32 audio.

        A shorter block is padded by repeating it and the extra audio dropped,
        which is upstream's own tail handling.
        """
        frames = codes.shape[0]
        if frames == 0:
            return np.zeros(0, dtype=np.float32)
        block = codes
        if frames < DECODER_FRAMES:
            block = codes[np.arange(DECODER_FRAMES) % frames]
        wave = self.sessions["tok_decoder"].run(
            None, {"audio_codes": block[None].astype(np.int64)}
        )[0]
        wave = np.asarray(wave, dtype=np.float32).reshape(-1)
        if frames < DECODER_FRAMES:
            wave = wave[: int(round(len(wave) * frames / DECODER_FRAMES))]
        return wave

    def _encode_reference(self, wave: np.ndarray) -> np.ndarray:
        """Reference audio at 24 kHz to codes, a second at a time."""
        blocks = []
        for start in range(0, max(len(wave), 1), _ENCODER_SAMPLES):
            window = wave[start : start + _ENCODER_SAMPLES]
            if len(window) < _ENCODER_SAMPLES:
                window = np.pad(window, (0, _ENCODER_SAMPLES - len(window)))
            blocks.append(
                self.sessions["tok_encoder"].run(
                    None,
                    {
                        "audio": window.reshape(1, 1, _ENCODER_SAMPLES).astype(
                            np.float32
                        )
                    },
                )[0][0]
            )
        return np.concatenate(blocks, axis=0).astype(np.int64)

    # -- generation --------------------------------------------------------

    def _special_embeds(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Embeddings for the three text control tokens the prompt is built from."""
        special = self._embed_text(
            [
                [
                    self._config["tts_bos_token_id"],
                    self._config["tts_eos_token_id"],
                    self._config["tts_pad_token_id"],
                ]
            ]
        )
        return special[:, 0:1], special[:, 1:2], special[:, 2:3]

    def _codec_prefix(
        self, language: str | None, speaker_embed: np.ndarray | None
    ) -> np.ndarray:
        """The codec-side prelude: whether to think about a language, and who speaks.

        `speaker_embed` occupies one position — the embedding of a built-in
        speaker's id for the custom-voice checkpoint, the reference recording's
        x-vector for the cloning one, and nothing at all for voice design.
        """
        talker = self._talker_config
        language_id = self.languages.get((language or "").lower())
        if language_id is None:
            tags = [
                [
                    talker["codec_nothink_id"],
                    talker["codec_think_bos_id"],
                    talker["codec_think_eos_id"],
                ]
            ]
        else:
            tags = [
                [
                    talker["codec_think_id"],
                    talker["codec_think_bos_id"],
                    language_id,
                    talker["codec_think_eos_id"],
                ]
            ]
        parts = [self._embed_codec(tags)]
        if speaker_embed is not None:
            parts.append(speaker_embed)
        parts.append(
            self._embed_codec([[talker["codec_pad_id"], talker["codec_bos_id"]]])
        )
        return np.concatenate(parts, axis=1)

    def _speaker_embed(self, speaker: str) -> np.ndarray:
        """One built-in speaker, as the codec embedding of its id."""
        speaker_id = self.speakers.get(speaker.lower())
        if speaker_id is None:
            raise Qwen3TtsError(f"unknown speaker {speaker!r}")
        return self._embed_codec([[speaker_id]])

    def _prompt_ids(self, text: str) -> np.ndarray:
        """Tokenise the assistant template the talker's prompt is cut from."""
        ids = self._ids(
            f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        )
        if ids.shape[1] < 9:
            raise Qwen3TtsError("text is too short to fill the assistant template")
        return ids

    def _role_block(
        self,
        prompt_ids: np.ndarray,
        codec_input: np.ndarray,
        bos_embed: np.ndarray,
        pad_embed: np.ndarray,
    ) -> np.ndarray:
        """The three role tokens, summed against the codec prelude beside them.

        The talker reads a text stream and a codec stream added position by
        position, so the shorter prelude is padded out to meet the role tokens.
        """
        pad_block = np.concatenate(
            [np.repeat(pad_embed, codec_input.shape[1] - 2, axis=1), bos_embed], axis=1
        )
        role = self._embed_text(prompt_ids[:, :3])
        return np.concatenate([role, pad_block + codec_input[:, :-1]], axis=1)

    def _prefill(
        self,
        text: str,
        *,
        language: str | None,
        speaker: str | None,
        instruct: str | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build the talker's prompt. Returns it and the per-step trailing embed."""
        bos_embed, eos_embed, pad_embed = self._special_embeds()
        speaker_embed = self._speaker_embed(speaker) if speaker else None
        codec_input = self._codec_prefix(language, speaker_embed)
        codec_pad = self._talker_config["codec_pad_id"]
        codec_bos = self._talker_config["codec_bos_id"]

        prompt_ids = self._prompt_ids(text)
        talker_input = self._role_block(prompt_ids, codec_input, bos_embed, pad_embed)

        body_ids = prompt_ids[:, 3:-5]
        body = self._embed_text(body_ids)
        text_block = np.concatenate([body, eos_embed], axis=1) + self._embed_codec(
            [[codec_pad] * (body_ids.shape[1] + 1)]
        )
        start_block = pad_embed + self._embed_codec([[codec_bos]])
        talker_input = np.concatenate([talker_input, text_block, start_block], axis=1)

        if instruct:
            prefix = self._embed_text(
                self._ids(f"<|im_start|>user\n{instruct}<|im_end|>\n")
            )
            talker_input = np.concatenate([prefix, talker_input], axis=1)

        return talker_input, pad_embed[:, 0]

    def _prefill_clone(
        self,
        text: str,
        reference: ReferenceConditioning,
        *,
        language: str | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build the in-context-learning prompt that clones a recording.

        The recording enters twice: as an x-vector where a built-in speaker's
        id would go, and as its own codec frames paired with its transcript, so
        the talker reads a worked example before the text it has to say.
        """
        bos_embed, eos_embed, pad_embed = self._special_embeds()
        codec_input = self._codec_prefix(language, reference.x_vector)
        codec_pad = self._talker_config["codec_pad_id"]
        codec_bos = self._talker_config["codec_bos_id"]

        prompt_ids = self._prompt_ids(text)
        talker_input = self._role_block(prompt_ids, codec_input, bos_embed, pad_embed)

        transcript_ids = self._ids(
            f"<|im_start|>assistant\n{reference.transcript}<|im_end|>\n"
        )[:, 3:-2]
        body_ids = np.concatenate([transcript_ids, prompt_ids[:, 3:-5]], axis=1)
        text_block = np.concatenate([self._embed_text(body_ids), eos_embed], axis=1)
        text_block = text_block + self._embed_codec([[codec_pad] * text_block.shape[1]])
        # The per-frame sum of a reference frame's group embeddings is the same
        # quantity the talker is fed after each generated frame.
        codec_block = np.concatenate(
            [
                self._embed_codec([[codec_bos]]),
                self._step_embed(reference.codes)[None],
            ],
            axis=1,
        )
        example = np.concatenate([text_block, codec_block + pad_embed], axis=1)
        return np.concatenate([talker_input, example], axis=1), pad_embed[:, 0]

    def condition(self, wave: np.ndarray, transcript: str) -> ReferenceConditioning:
        """Encode a 24 kHz mono recording into everything cloning needs.

        Kept apart from generation because it depends only on the recording:
        the caller does this once per reference and reuses it for every reply.
        """
        if not self.can_clone:
            raise Qwen3TtsError("this bundle carries no encoders, so it cannot clone")
        if not transcript:
            raise Qwen3TtsError("cloning needs the reference recording's transcript")
        x_vector = self.sessions["speaker_encoder"].run(
            None, {"audio": wave[None].astype(np.float32)}
        )[0]
        return ReferenceConditioning(
            codes=self._encode_reference(wave),
            x_vector=x_vector.reshape(1, 1, self._talker_config["hidden_size"]),
            transcript=transcript,
        )

    def _empty_past(self) -> list[np.ndarray]:
        talker = self._talker_config
        shape = (1, talker["num_key_value_heads"], 0, talker["head_dim"])
        return [np.zeros(shape, np.float32) for _ in self._past_names]

    def frames(
        self,
        text: str,
        *,
        speaker: str | None = None,
        language: str | None = None,
        instruct: str | None = None,
        reference: ReferenceConditioning | None = None,
        max_frames: int = 2048,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.05,
        seed: int = 0,
    ) -> Iterator[np.ndarray]:
        """Yield one `[CODE_GROUPS]` frame at a time until end-of-speech.

        The talker samples the first code of each frame; the code predictor
        fills the other fifteen, one at a time because each is conditioned on
        the ones already chosen. That inner loop is where most of the time
        goes, and it is fifteen model calls per frame however fast the host is.
        """
        if reference is not None:
            talker_input, trailing = self._prefill_clone(
                text, reference, language=language
            )
        else:
            talker_input, trailing = self._prefill(
                text, language=language, speaker=speaker, instruct=instruct
            )
        yield from self._decode_loop(
            talker_input,
            trailing,
            max_frames=max_frames,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
        )

    def _decode_loop(
        self,
        talker_input: np.ndarray,
        trailing: np.ndarray,
        *,
        max_frames: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        seed: int,
    ) -> Iterator[np.ndarray]:
        talker = self._talker_config
        codec_eos = talker["codec_eos_token_id"]
        vocab = talker["vocab_size"]
        # The top of the codec vocabulary holds prompt-only tags — speaker
        # ids, language ids, the think markers. Only end-of-speech may be
        # sampled from that range.
        suppressed = np.array(
            [i for i in range(vocab - 1024, vocab) if i != codec_eos], dtype=np.int64
        )
        rng = np.random.default_rng(seed)

        past = self._empty_past()
        length = talker_input.shape[1]
        positions = np.broadcast_to(np.arange(length), (3, 1, length)).copy()
        logits, hidden, past = self._talker(
            talker_input, positions, np.ones((1, length), np.int64), past
        )

        chosen: list[int] = []
        for _ in range(max_frames):
            first = logits[0, -1].astype(np.float64).copy()
            first[suppressed] = -np.inf
            first = _penalise(first, chosen, repetition_penalty)
            code = _sample(
                first, temperature=temperature, top_k=top_k, top_p=top_p, rng=rng
            )
            if code == codec_eos:
                return
            chosen.append(code)

            state = hidden[0, -1][None].astype(np.float32)
            frame = np.zeros((1, CODE_GROUPS), dtype=np.int64)
            frame[0, 0] = code
            for group in range(1, CODE_GROUPS):
                group_logits = self._predict_group(state, frame)
                frame[0, group] = _sample(
                    group_logits[0, group - 1],
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    rng=rng,
                )
            yield frame[0].copy()

            step = self._step_embed(frame)[:, None] + trailing[:, None]
            positions = np.broadcast_to(np.array([length]), (3, 1, 1)).copy()
            logits, hidden, past = self._talker(
                step, positions, np.ones((1, length + 1), np.int64), past
            )
            length += 1
