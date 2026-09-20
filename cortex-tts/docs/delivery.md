# Keeping up

Two things can go wrong when a reply is spoken aloud: you wait too long for
the first word, or the sound stops in the middle. This page is how the app
trades one against the other, and what to do when it gets it wrong. The app
decides per reply from one number measured on your host, or the integration's
per-model **Speaking mode** — `auto`, `streaming` or `buffered` — names the
trade for it.

## Why it is a trade at all

To start speaking sooner, the app has to release the first piece of audio
before the rest is rendered. Once it does, the player runs at normal speed and
cannot be paused — Home Assistant's TTS cache drains the stream whatever the
app does — so the app's one choice is when to start. Too early and the player
runs out before the next piece arrives. The number that decides this is the
**lead**: the audio handed over so far minus the time since the first byte —
how many seconds the listener still has to play. Positive is a cushion;
negative means the sound already stopped.

## One number decides it

The **real-time factor** of the voice in use: render seconds over audio
seconds of one request, fixed cost included. The app keeps one per model and
voice — the median of that voice's last eight requests on the execution
provider now in use, shown from the first request and deciding nothing until three exist. A clone and a built-in voice
on the same model are two numbers, because the recording rejoins the prompt on
every synthesis; nothing is pooled, borrowed, or carried from another machine,
and loading a model is outside every request. The model's card in the admin
UI shows the figure, how many requests it rests on, the provider, and the
verdict.

The **verdict** is one comparison, made once per reply at `ready` and never
revised: under this host's threshold the reply streams; otherwise, or while
the voice is unmeasured, it is buffered. The threshold is **Stream under
RTF** in the app's settings (`stream_rtf`, default 0.8, 0.1 to 3.0). Raising
it brings the first word sooner on a slow voice at the price of a possible
gap on a long reply; the card shows each voice's figure against it, so the
effect of a change is visible before any reply is spoken. What a given
figure risks depends on the engine — MOSS releases audio within a sentence
and at 1.1 runs dry only on replies past about a minute, while Hojo and
OmniVoice hand each sentence over whole and can drain the bank on one long
sentence — which is why the line is a host's to move and not a fixed truth.

## Three settings, two outcomes

**Speaking mode** is per model in the integration; a caller of the API sends
it as `mode` in the `start` frame.

- `auto` takes the verdict. It is what anything serving a listener should send.
- `streaming` insists, whatever the voice measured. On a host that cannot keep
  ahead of its audio the reply stalls between sentences, audibly.
- `buffered` insists on holding the reply until it is rendered. It never
  stalls, and its wait is the longest of the three.

A reply ends up one of two ways, and the `done` frame says which.

**Streaming** — one sentence per request, rendered as the words arrive.
Nothing plays until `BANK_S` (6 s) of audio has been rendered, or the whole
reply has; from then on each sentence goes out as it finishes. The bank is the
only hold; an engine that emits audio while it renders counts its chunks
toward it. The 6 s is for a reply whose length is still unknown: once the
writer has finished, the rest is in hand and the hold shrinks to what that
rest needs at the voice's measured pace, so a short reply is not held for a
long one's sake. A single-sentence reply is never held by it at all.

**Buffered** — rendered as the words arrive: whenever the engine is free it
takes every complete sentence waiting as one request, split only at the
model's own segment limit. Nothing plays until everything has been rendered,
and the finished audio is levelled like a file before it goes. It cannot run
dry; the wait is about the longer of the writer and the renderer, plus the
last request — not their sum.

Neither outcome depends on how long the reply turns out to be, so one model
and one setting give one behaviour.

## Where 0.8 and 6 s come from

Both constants are argued in
[ADR 0001](adr/0001-rtf-threshold-pacing.md) from `scripts/replay_pacing.py`,
which replays recorded requests and real replies through the rules: the
default threshold sits just above where the median stops misclassifying any
measured host, model and voice against a verdict made by hand from the same
data, admitting the two cells at 0.78 whose buffered wait costs more than
their gap; the bank is the
smallest that leaves almost no multi-sentence production reply with a gap on a
host just under the threshold, and none on a GPU. A change to either constant
comes with the script's output, and a new host's samples go into its inputs.
A slow writer — a conversation agent producing text more slowly than it is
spoken — is outside the rule: the bank does not cover it, and a gap then lands
at a sentence boundary.

## Reading how it went

Every live reply ends in one log line from `cortex_tts.api.live`, at INFO — or
at WARNING when the reply ran dry, so filtering the app's log on WARNING lists
every stall:

```
spoke live model=hojo-40m voice=hojo_zh_f_01 setting=auto verdict=streaming rtf=0.58 threshold=0.80 samples=8 provider=cpu outcome=streaming requests=4 audio_s=17.3 first_audio_ms=5810 bank_wait_ms=412 min_lead_s=2.14 gap_at=3
```

`setting` is what was asked, `verdict` what the measurement said, `outcome`
what happened; the last two differ only when the caller insisted.
`bank_wait_ms` is how long the first audio waited between being rendered and
released; `gap_at` is the request whose audio landed at the lowest lead. The
same figures are in the `done` frame ([HTTP API](api.md#speaking-live)), and
the integration writes its sensors from that frame.

| Sensor                            | Reads                                                                                 |
| --------------------------------- | ------------------------------------------------------------------------------------- |
| `sensor.<model>_delivery_mode`    | `streaming` or `buffered` — how the last reply was actually spoken                    |
| `sensor.<model>_real_time_factor` | What the last reply cost on **this** host: render seconds over audio seconds          |
| `sensor.<model>_playback_margin`  | The least audio the listener still held over the reply; negative ran dry by that much |

**Margin is the one to watch**: the lowest the lead got over the whole reply.
Negative means the speaker caught up with the renderer, and the number is how
many seconds of silence that was; a buffered reply's margin is never negative.
It measures what the app handed to Home Assistant, upstream of a buffer with
no back-pressure — it cannot see what the speaker received.

## If margin is negative

A streaming reply ran dry: a voice close to the line loses on a busy moment,
and a reply that insisted on `streaming` loses whatever it measured. A
buffered reply cannot, so the fix is that model's setting or that model:

- **Set that model's Speaking mode to `buffered`.** The reply waits instead of
  stalling.
- **Use a faster model**, which usually means the 40M, or a faster host; what a
  card buys each model is in [Models](models.md).
- **Check the threads setting suits this model.** It is a trade, not a number
  to raise: the same four cores make MOSS slower and OmniVoice faster
  ([Models](models.md#hardware-and-running-it-elsewhere)).
- **Reset the model's measurements** from its card after changing the thread
  count, so the median describes the host as it is now. A change of execution
  provider needs no reset — samples are kept per provider. A benchmark needs
  one afterwards ([Models](models.md#what-a-benchmark-cannot-measure)).

## Cancelling

A listener that goes away — the pipeline cancelled, the socket closed — stops
the render at the engine's next checkpoint: a decode step, a diffusion step, a
codec chunk. Nothing an abandoned reply cost is recorded against the model.
The one thing the app cannot see is a media player that stopped: Home
Assistant's TTS cache drains the stream regardless, so that render finishes.
