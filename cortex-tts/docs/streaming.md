# Keeping up

Whether a model can speak a reply while the rest is still being written, and
how the app decides that on your host. The decision is the app's, per reply,
from what it has measured; the integration's one setting — **Speaking mode**,
per model — is `auto` or `buffered`.

## What the listener needs

A streamed reply is smooth when the audio already handed to the player always
outlasts the wait for the next piece. Call that the **lead**: audio sent minus
time elapsed since the first byte. Every request the app makes changes it in
two ways:

- A request costs a **fixed part** before any audio exists — the prompt, the
  reference recording being re-encoded — plus a part **per second of audio**,
  which is the real-time factor.
- An engine that hands a request over whole (Hojo, OmniVoice) produces
  nothing until it is done, so the whole cost is a gap the previous audio has
  to cover. One that emits audio while rendering (MOSS-TTS-Nano, Qwen3-TTS)
  pays only the fixed part up front and then gains or loses at the rate it
  renders.

Neither number survives a change of host, execution provider, thread count or
reference recording. So the app does not carry them: it fits them from the
requests it has actually served, per model and per kind of voice, and refits
as they arrive. The stats card in the admin UI shows the fit.

## The three ways a reply is spoken

The `done` frame of a live reply, and the integration's **Last synthesis
mode** sensor, say which one it was.

- **Streaming.** The model gains lead on every request — it renders faster
  than the audio plays, by enough to pay the fixed part. The first request
  goes out as soon as it holds about three seconds of speech (a bare "好了，"
  is not worth a request of its own: in 42% of production replies the first
  clause is under a second), and never more than about six. After that each
  request is the largest run of sentences whose predicted cost fits inside
  the lead the listener holds. The opening is held back only long enough to
  cover the first boundary.
- **Paced.** The whole reply was written before anything had to be sent (a
  one-line answer, or a fast writer), or the model cannot gain lead, so
  nothing could be sent safely before the reply was known. Every request is
  known, so the opening hold is computed exactly: the least wait after which
  playback never catches the renderer. For OmniVoice with a clone on a GTX
  1650 that is about a second and a half; for MOSS-TTS-Nano on a CPU, several
  seconds — still far short of waiting for the whole reply.
- **Buffered.** Nothing is released until everything is rendered. What a
  model gets while this host has not measured it (three requests are enough),
  and what **Speaking mode: buffered** asks for outright.

## Where the threshold is

A request that produces `L` seconds of audio costs `a + b·L` to render, so it
changes the lead by `(1 − b)·L − a`. Streaming needs that to clear the
planner's margin (0.5 s) on a typical batch of about six seconds:

```
b < 1 − (a + 0.5) / 6      a = 0.3 s → b < 0.87      a = 1.5 s → b < 0.67
```

That is the whole rule. A whole-render engine with a small fixed cost
(Hojo) streams under about 0.85; a cloned voice on OmniVoice carries a
1.5 s fixed cost and needs 0.67; a chunk-streaming engine (MOSS, Qwen3-TTS)
streams under about 0.9. Past the threshold a reply is paced: it never
stalls, but the opening wait grows with the reply — about
`(b − 1) × reply + fixed per request`. Driving the real planner through a
simulated session across RTF 0.5–3.0, both engine kinds, both fixed costs,
replies of 6–60 s and two writer speeds (224 runs) produced no negative lead
anywhere; at RTF 1.5 a 30 s reply waits 21–33 s, at 2.0 it waits 37–49 s.
From 1.2 upwards the answer is a faster host or model, not a setting.

## What the measurements said

Why the app sizes the requests and the integration does not. The same
six-sentence paragraph, sent back to back, with playback starting on the first
byte; the column is the least audio the listener still held when the next
request's audio landed.

| Model, host                       | Streams inside a request | Cost measured             | One request per sentence | Grouped to 9 s | Grouped to 14 s |
| --------------------------------- | ------------------------ | ------------------------- | ------------------------ | -------------- | --------------- |
| OmniVoice, GTX 1650, cloned voice | no                       | ~1.5 s fixed + 1.0× audio | −0.58 s                  | −0.25 s        | **−4.35 s**     |
| Hojo 40M, GTX 1650                | no                       | 0.35×                     | +2.8 s                   | +2.8 s         | +2.9 s          |
| Hojo 40M, HA VM CPU               | no                       | 0.7×                      | +0.9 s                   | +1.0 s         | +0.65 s         |
| MOSS-TTS-Nano, HA VM CPU, cloned  | yes                      | 1.14–1.21×                | −4.9 s                   | −4.9 s         | −4.8 s          |

Two things follow. A model that is not gaining lead cannot be streamed by any
grouping, and a fixed grouping is wrong in both directions: growing the batch
while the lead does not grow is exactly how OmniVoice fell 4.35 s behind, and
a fast model gains from any split. The batch has to follow the lead.

The same paragraph rendered as one, three and six requests changed pitch
wander by nothing measurable (f0 variation over 4-second windows: Hojo 9–11%,
OmniVoice 7–10% in every condition; MOSS drifted _more_ in one long generation,
17.6%, than in six sentences, 10.5%). Splitting a reply does not cost
consistency at this granularity, so there is no mode that trades latency for
fewer pieces.

## What a request should hold

Coalescing sentences removes the fixed cost of sending one at a time — 0.37 s
of dead air per boundary on MOSS — but a request that is too long costs more
than it saves: an autoregressive model attends over everything it has
generated, so the bill grows with the square of the request. Measured on
MOSS-TTS-Nano, the same 55-second story sent in pieces of different sizes, as
the fraction by which rendering fell behind playback:

| Per request         | Behind playback |
| ------------------- | --------------- |
| ~1 s (one sentence) | +41%            |
| ~9 s                | +4.1%           |
| ~19 s               | +5.6%           |
| ~28 s               | +6.5%           |
| ~55 s (one request) | +17.8%          |

A valley with a cliff on either side. A paced reply groups sentences to about
twelve seconds; a streamed one is bounded by the lead, which sits in the same
range on any model that is gaining it.

## The sensors that answer it

| Sensor                            | Reads                                                                    |
| --------------------------------- | ------------------------------------------------------------------------ |
| `sensor.<model>_real_time_factor` | What the last synthesis actually cost on **this** host                   |
| `sensor.<model>_playback_margin`  | The least audio the listener still held over the reply; negative ran dry |
| `sensor.<model>_delivery_mode`    | `whole`, `streaming` or `paced` — how the last reply was actually spoken |

**Margin is the one to watch.** Positive means the speaker always had something
left to play. Negative means it caught up with the renderer and waited, and the
number is how many seconds of silence that was. It measures what the app handed
Home Assistant, which is upstream of a buffer without back-pressure — it
cannot see what the speaker received, and a player that stops does not stop
the render.

## If margin is negative

The app's estimate of the model was too optimistic for that reply. It learns
from every request, so the next reply is paced from a corrected fit, and one
stutter usually corrects itself. If a model keeps losing:

- **Set that model's Speaking mode to buffered** in the integration.
- **Use a faster model**, which usually means the 40M.
- **Check the threads setting is 2.** Four was measured 70% slower than two on
  a four-core host; more is not better here.
- **Reset the model's measurements** from its card after changing the host —
  a new execution provider, a different thread count — so the fit starts
  from what the host is now rather than what it was.

## Cancelling

A listener that goes away — the pipeline cancelled, the socket closed — stops
the render at the engine's next checkpoint: a decode step, a diffusion step, a
codec chunk. Nothing an abandoned reply cost is recorded against the model.
The one thing the app cannot see is a media player that stopped: Home
Assistant's TTS cache drains the stream regardless, so that render finishes.

## Why a stream is MP3

`/api/speak/live` answers MP3: a bare sequence of
self-describing frames, with no container, no length field and no index, which
is the only honest thing to send when the length is not known yet. A WAV
stream has to declare a length before the audio exists, and a general-purpose
player given the maximal one waits for a file it believes is six hours long.
FLAC and OGG both need a size or an index written before the audio exists, so
they are refused. The bitrate — `bitrate` in the `ready` frame — is the one
measurement that exists before the first sample, and it is what turns a byte
count into a duration downstream.
