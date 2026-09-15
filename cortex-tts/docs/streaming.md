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
requests it has actually served and refits as they arrive. What a request
costs belongs to the model and the host, so a model's own voices share one
line — which is what lets one with eighteen built-in voices have a line at
all — while a clone keeps its own, because its recording rejoins the prompt on
every synthesis. How fast a voice speaks is the other question and never
pools.

What is fitted together is what shares a cost. A model's own voices pool into
one cost line — measured on MOSS, one host, the cost of a second of audio
varies 4% across them — while each clone keeps its own, because its recording
rejoins the prompt on every synthesis at 0.354 s of render per second of
reference. How fast a voice speaks is the other question and never pools: the
same 26 characters ran 6.64 s in one built-in voice and 5.12 s in another, a
30% spread, and every request the app sizes is sized in seconds of speech.
The stats card in the admin UI shows the fit.

## The three ways a reply is spoken

The `batch` frame of a live reply names the plan in force before any audio
exists; the `done` frame replaces it with what happened. The integration's
**Delivery mode** sensor is written from both.

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
  known, so the wait is not a number settled before the render starts: the
  app banks the opening audio and releases it the moment the bank covers what
  the requests still to come are predicted to lose. That question is asked
  again as each piece of audio arrives, and charged to the pace this reply is
  actually running at rather than to the host's average day — still far short
  of waiting for the whole reply.
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

That is the first of two tests. The second is the opening hold itself:
whatever of the first boundary's cost this batch's own playback does not
cover, plus the margin and the fit's spread, has to come to under four
seconds. A bank larger than that would go on growing with a reply whose
length nobody yet knows, so such a reply is paced instead. A whole-render engine with a small fixed cost
(Hojo) streams under about 0.85; a cloned voice on OmniVoice carries a
1.5 s fixed cost and needs 0.67; a chunk-streaming engine (MOSS, Qwen3-TTS)
streams under about 0.9. Past the threshold a reply is paced: it never
stalls, but the opening wait grows with the reply — about
`(b − 1) × reply + fixed per request`. Driving the real planner through a
simulated session across RTF 0.5–3.0, both engine kinds, both fixed costs,
replies of 6–60 s and two writer speeds (224 runs) produced no negative lead
anywhere; at RTF 1.5 a 30 s reply waits 21–33 s, at 2.0 it waits 37–49 s.
From 1.2 upwards the answer is a faster host or model, not a setting.

## What the measurements settled

Two results from the tables in [Models](models.md#what-batching-costs), which
is where they are measured and re-measured:

A model that is not gaining lead cannot be streamed by any grouping, and a
fixed grouping is wrong in both directions — growing the batch while the lead
does not grow is how OmniVoice fell behind, and a fast model gains from any
split. So the batch follows the lead. Splitting costs nothing in consistency:
the same paragraph as one, three and six requests changed pitch wander by
nothing measurable, so there is no mode that trades latency for fewer pieces.

And the cost of a request against its size is a valley with a cliff on either
side. Too small and every boundary repays the fixed cost; too large and an
autoregressive model's attention over what it has already generated turns the
bill quadratic. Twelve seconds is where a paced reply stops looking, because
past it the fitted line is knowingly wrong. Inside it the grouping is a search
rather than a constant: the sentences are grouped at every cut that moves a
boundary, and the plan whose first word comes soonest wins, a tie going to the
fewest boundaries. A streamed reply is bounded by the lead instead, which sits
in the same range on any model that is gaining it.

## The sensors that answer it

| Sensor                            | Reads                                                                                                                                                 |
| --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `sensor.<model>_real_time_factor` | What the last synthesis actually cost on **this** host                                                                                                |
| `sensor.<model>_playback_margin`  | The least audio the listener still held over the reply; negative ran dry                                                                              |
| `sensor.<model>_delivery_mode`    | `whole`, `streaming`, `paced` or `buffered` — how the last reply was actually spoken; `whole` whenever it fitted one request, whichever plan chose it |

**Margin is the one to watch.** Positive means the speaker always had something
left to play. Negative means it caught up with the renderer and waited, and the
number is how many seconds of silence that was. It measures what the app handed
Home Assistant, which is upstream of a buffer without back-pressure — it
cannot see what the speaker received, and a player that stops does not stop
the render.

## If margin is negative

The app's estimate of the model was too optimistic for that reply. A paced
reply corrects itself while it runs: once it has produced a second of audio
its own pace is believed over the fit's — the request still rendering
included — and the hold is taken from the dearer line. On a host made 1.4×
dearer than its fit, that turned 16 s of dry playback into under two. It also
learns from every request, so the next reply is paced from a corrected fit, and one
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
