# Reference audio is decoded, cached and limited in one place, not per engine

Adding a second cloning engine turned three private details of the 80M engine
into things two engines need. Each was about to be written twice, and each was
measured to be worth extracting.

## One decoder, no torch

The MOSS runtime loads a reference with `torchaudio.load()` — one call, at
module scope, and the only reason its ONNX path imports torch at all. That
import costs ~214 MB of resident memory that a bundled voice never uses, on a
model whose 1988 MB is already the reason it does not fit the target host.

`soundfile` and `scipy` are already dependencies of this app, and between them
they decode, resample and downmix. So the vendored MOSS runtime gets that one
function replaced and drops torch entirely.

The 80M is left alone. Its reference handling lives in `vendor/hojo80.py`,
which is upstream's code kept diffable on purpose; rewriting it to share this
decoder would trade a real property for a tidier import graph. The shared
decoder serves the new engine and the next one, not a retrofit.

## Conditioning is cached, and `forget()` is what invalidates it

MOSS re-encodes the reference recording on every single request: 960 ms for a
125-frame prompt, paid again for every sentence, because the runtime takes a
file path rather than the codes it derives from it. Measured at the 2-thread
anchor, caching those codes is the difference between RTF 0.84 and the 1.19 an
uncached path shows — the model is not slow, the plumbing is.

So conditioning derived from a reference — Hojo's speaker embedding, MOSS's
prompt codes — is cached by reference id in `engine/conditioning.py`, and
`forget()`, which already existed on the protocol for precisely this
lifecycle, is what drops it. Neither engine implements its own cache.

One cache means one invalidation rule, which is the reason to share it rather
than to save the twenty lines. An entry also carries the recording's
fingerprint and is dropped when it changes, so the two paths that can put
different audio behind a reference id — a re-upload under a reused slug, an
engine that outlives a `forget()` it never received — cannot leave an engine
synthesising the previous voice. The fingerprint covers the audio only:
correcting a transcript must not force a re-encode, because the transcript
reaches the model as text, not through the encoder.

## The reference is levelled on the way in, not the output on the way out

MOSS clips. Three of ten Home-Assistant-shaped replies rendered with its
Taiwanese-accent voice peaked past full scale (1.061, 1.197, 1.264).

The first instinct — limit peaks in `engine/join.py`, where every waveform
passes on its way out — is wrong twice over. `audio.normalize_level` already
scales the output to a target peak before encoding, so nothing clips at the
int16 conversion; and the distortion is _generated_, baked into the samples the
model produced, which scaling the whole utterance down cannot undo. A limiter
there would fix nothing and hide the symptom.

The cause is at the other end. Output level tracks the reference recording's
loudness monotonically, across eight references:

    reference −12.2 dBFS RMS → peak 1.078      reference −19.6 → peak 0.785
    reference −12.6          → peak 1.053      reference −21.5 → peak 0.625
    reference −15.2          → peak 1.133      reference −24.3 → peak 0.248

Levelling all eight to −21 dBFS and re-rendering removed the clipping
completely: the highest peak across the set fell to 0.905. It also explains a
detail that looked like a model defect — the bundled voice the benchmark
happened to use, `Junhao`, is the loudest of the six Chinese presets (0.909
against `Lingyu`'s 0.366), so "MOSS clips" was partly "that reference is hot".

So the seam is a shared reference decoder that resamples, downmixes and
**levels** in one place, applied when a recording is decoded for conditioning
rather than when it is stored. Non-destructive: what the user uploaded is what
plays back from `/api/references/{id}/audio`.

Measured on MOSS only. The 80M reaches its reference through vendored upstream
code that stays byte-diffable against upstream, so it keeps its own path and
this decoder serves the new engine and whatever comes after it.
