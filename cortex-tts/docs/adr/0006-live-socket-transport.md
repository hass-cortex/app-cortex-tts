# ADR 0006 — The live socket: MP3 without a length, bounded frames, legible delivery

Status: accepted. Pacing itself is ADR 0001.

## Context

Every reply the integration speaks goes over `/api/speak/live`, one
WebSocket per reply, text frames in and audio frames out. What the socket
carries beside audio decides whether a stall can be diagnosed at all.

## Decision

- **Chunked audio is the WebSocket's alone.** `/api/speak/live` is the only
  route that sends audio as it is produced — a buffered reply included,
  which is held to the end and levelled like a file before release.
  `/v1/audio/speech` answers a finished file for callers that are not Home
  Assistant. Two routes because those are two questions.
- **A streamed format must not have to declare a length.** The stream is
  MP3: a bare frame sequence with no container, no length field and no
  index. WAV is offered but not the default — the maximal length its header
  declares is read by a general-purpose player as a six-hour file it then
  waits to buffer. FLAC and OGG are not offered. The `ready` frame carries
  `bitrate`, the one measurement that exists before the first sample, so a
  byte count can become a duration downstream.
- **A stream has its own level control.** `encode` peak-normalises a
  finished waveform; a stream has none, so `StreamGain` holds a gain that
  only ever falls, far enough to keep each chunk under the ceiling. Scaling
  each chunk to its own peak would pump.
- **One audio frame is bounded, not one reply.** A receiver buffers a frame
  whole before it sees any of it — aiohttp's default cap is 4 MB — and a
  buffered reply's release is the whole reply at once. `MAX_FRAME_BYTES`
  slices at 512 KB; the bytes are a stream and the receiver concatenates.
- **Everything that can fail must fail before the first byte.** The model,
  the voice and the encoder are resolved before `ready`; after audio has
  started an error can only truncate it.
- **The delivery is legible from outside, or it is not reviewable.** A
  `batch` frame says what a request carries and which way the reply is
  spoken; a `rendered` frame says what it cost; `done` says how it went,
  with the lowest lead and where it fell. None of that is recoverable from
  the audio, and while the bank is held there is no audio. The admin UI's
  live panel draws its timeline from those frames and from what the audio
  clock did: it asks for WAV and schedules the raw PCM on a Web Audio clock
  rather than an `<audio>` element, because an element decides the timing
  for itself and the timing is the subject. A host with no audio device
  reports a running context whose clock never advances (measured in headless
  Chrome), so the panel withdraws every playback figure rather than count a
  playback that is not happening.
- **The mode is settled at `ready` and never revised**, and it is one of two
  words, `streaming` or `buffered`, repeated in `batch` and `done`. The
  integration reads it into an enum sensor, and an enum handed a state
  outside its options raises rather than degrades — the reply never plays
  at all — so the mode words are why `api_version` moves.
- **A browser carries the key in the subprotocol list** (`["cortex-tts",
key]`), because a `WebSocket` constructor sets no headers; behind ingress
  it sends none. **Ingress bypasses the key, and only ingress**:
  `deps.is_ingress` requires both the Supervisor's `X-Ingress-Path` header
  and the Supervisor's peer address (`172.30.32.2`), since the header alone
  is forgeable by anything that can reach the port.

## Consequences

- The frame protocol is documented in `docs/api.md` ("Speaking live") and
  is the part of the API OpenAPI cannot describe.
- A new field a listener needs goes into `ready` if it exists before the
  first sample and into `done` otherwise; nothing about a reply is inferred
  from the audio.
