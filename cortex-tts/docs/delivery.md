# Keeping up

Two things can go wrong when a reply is spoken aloud. You wait too long for
the first word, or the sound stops in the middle. This page is how the app
trades one against the other, and what to do when it gets it wrong.

The app decides per reply, from what it has measured on your host — or stops
deciding, if you would rather name the trade yourself. The integration gives
you one setting per model, **Speaking mode**, with four answers: `auto`,
`planned`, `unheld` and `buffered`.

## Why it is a trade at all

Rendering takes time. To start speaking sooner, the app has to release the
first piece of audio before the rest is rendered. Once it does, the player
runs at normal speed and cannot be paused — Home Assistant's TTS cache drains
the stream whatever the app does. So the app gets one choice: when to start.
Start too early and the player runs out before the next piece arrives, and the
listener hears silence.

The number that decides this is the **lead**: how many seconds of audio the
listener still has left to play. It is the audio handed over so far, minus the
time since the first byte. Positive means there is a cushion. Negative means
the sound already stopped.

## What a request costs

Every render has two parts:

- a **fixed cost** paid before any audio exists — the prompt, and a cloned
  voice's recording being fed in again
- a cost **per second of audio**, which is the real-time factor

Some engines hand a request over whole (Hojo, OmniVoice). Until one finishes,
it has produced nothing, so its entire cost is a gap the previous audio must
cover. Others emit audio while they render (MOSS-TTS-Nano, Qwen3-TTS). Those
pay only the fixed part up front, then gain or lose ground at the rate they
render.

Neither number survives a change of host, execution provider, thread count or
reference recording, so the app never ships them. It fits them from the
requests it has actually served, and refits as more arrive. The stats card in
the admin UI shows the current fit.

### Which voices share a figure

Cost belongs to the model and the host, so a model's own voices pool into one
line — measured on MOSS, one host, a second of audio varied 4% across them.
That pooling is what lets a model with eighteen built-in voices have a line at
all. A cloned voice keeps its own, because its recording rejoins the prompt on
every synthesis — measured on OmniVoice at 0.354 s of render per second of
reference, and a tenth of that on MOSS.

Speed of speech is a different question and never pools. The same 26
characters ran 6.64 s in one built-in voice and 5.12 s in another — a 30%
spread — and every request is sized in seconds of speech.

A voice with no line yet is the one exception, and only for planning: it
borrows the dearest of the model's other lines rather than being held whole.
What a clone cannot share turns out to be the fixed cost alone, not the
real-time factor — see
[Models](models.md#which-half-of-a-cost-line-belongs-to-the-voice). The card
still shows only what was actually measured.

## The four ways a reply is spoken

A live reply's `batch` frame names the plan before any audio exists; the
`done` frame says what happened. The integration's **Delivery mode** sensor is
written from both.

**Streaming** — the model renders faster than the audio plays, by enough to
cover the fixed cost each time. The first request goes out once it holds about
three seconds of speech, and never more than about six. A bare "好了，" is not
worth a request of its own: in 42% of production replies the first clause is
under a second. After that, each request is the longest run of sentences whose
predicted cost fits inside the lead the listener holds.

**Planned** — the whole reply was written before anything had to go out, or the
model cannot stay ahead. Every request is known in advance, so the opening
wait is not a fixed number. On a whole-render engine the app computes the
moment playback may start and arms a timer on the first audio byte; the bank
is the early way out, releasing sooner if it comes to cover what the
remaining requests are predicted to lose. It re-asks that question as each
piece arrives, and charges it to the pace this reply is actually running at
rather than to the host's average day. A chunk-streaming engine gets no
deadline — its bank grows with every chunk, so there is nothing to be caught
between.

**Unheld** — the same words in hand as **Planned**, cut for the soonest first
word instead of the fewest boundaries, and released with no hold at all. On a
model that cannot keep ahead it runs dry, audibly: that is the trade it
offers, a reply that starts as soon as there is audio against one that never
stalls.

Nothing held back is not the same as no wait. The first request still has to
render before there is anything to release, and on a model slower than its own
audio that alone is most of the wait: measured on OmniVoice with a clone on a
CPU, a reply whose first request ran 7.15x took 24 s to have a first word that
nothing was holding back — and 15 s more before it, making the model resident.

It is asked for, never concluded. A reply slow enough to want it is slow
enough to hear, and **Speaking mode** is where to say so — a planner that
switched per reply, on a length nobody can see, would give one model and one
setting two behaviours and leave anyone hearing the difference unable to tell
which they had.

**Buffered** — one render of the whole reply, played when it is done. Only
**Speaking mode: buffered** produces this; `auto` never arrives at it. It is
the
escape hatch, so nothing below applies to it: no cap, no cutting. Those exist
to get the first word out sooner, and a reply that releases nothing early has
no first word to bring forward.

A model this host has served nothing of is planned like any other. Its own first
request is the measurement — this host, this voice, a moment ago — and nothing
is released until that request lands. Figures are in
[Models](models.md#what-a-first-request-can-promise).

## When streaming is possible

A request producing `L` seconds of audio costs `a + b·L`, so it changes the
lead by `(1 − b)·L − a`. Streaming needs that to clear the planner's 0.5 s
margin on a typical six-second batch:

```
b < 1 − (a + 0.5) / 6      a = 0.3 s → b < 0.87      a = 1.5 s → b < 0.67
```

That is the whole test — nothing weighs the opening's own length. A model
that clears the margin above opens on a bank of 0.81–0.89 s across every line
fitted on either production host, so any ceiling loose enough to let those
through decides nothing, and one tight enough to bite refuses every model.

A planned hold is whatever never stalling takes, and nothing shortens it: the
plan was grouped so playback could not catch the renderer, and cutting the
hold leaves that plan played in a way it was not chosen for. A reply too slow
to wait out is one to set **Speaking mode: unheld** for.

The hold is the one figure that adds up a cost line's predictions over a whole
reply, so it is widened by how far that line misses what comes after it. Two
different things are measured about a line and only one of them cancels: the
scatter of requests around it, and the movement of the line itself. Across the
eight lines one host had fitted, scatter ran 2.0–9.1% of a render while the
line missed the following requests' total by up to 7.9% — and an offset that
small is seven seconds of a three-minute render. A line measured to be running
cheap therefore buys the opening that fraction more silence; one that
over-predicts buys none, having already paid for it.

In practice a whole-render engine with a small fixed cost (Hojo) streams below
about 0.85. A cloned voice on OmniVoice carries a 1.5 s fixed cost and needs
0.67. A chunk-streaming engine (MOSS, Qwen3-TTS) streams below about 0.9.

Past the threshold a reply is planned. It never stalls, but the opening wait
grows with the reply, roughly `(b − 1) × reply + fixed per request`. Driving
the real planner through a simulated session across RTF 0.5–3.0, both engine
kinds, both fixed costs, replies of 6–60 s and two writer speeds (224 runs)
produced no negative lead anywhere. At RTF 1.5 a 30 s reply waits 21–33 s; at
2.0 it waits 37–49 s. From about 1.2 upwards the fix is a faster host or a
faster model, not a setting.

## How a reply is cut up

Two results decide this, both measured in
[Models](models.md#what-batching-costs).

**A fixed batch size is wrong in both directions.** A model that is not
gaining lead cannot be streamed by any grouping, and a fast model gains from
any split. Growing the batch while the lead did not grow is how OmniVoice fell
behind. So the batch follows the lead. Splitting costs nothing in consistency
— the same paragraph as one, three and six requests changed pitch wander by
nothing measurable — so there is no mode that trades latency for fewer pieces.

**Cost against request size is a valley.** Cut too small and every boundary
repays the fixed cost. Cut too large and the model's attention over what it
has already generated makes the bill climb faster than the audio does.

Nine seconds is the default cap. That is the longest request measured still on
its own fitted line ([Models](models.md#where-the-line-breaks)). Past it the
planner would be comparing plans using arithmetic it knows to be wrong — and
wrong in the direction that hurts, because a request whose render is
under-predicted gets a hold that is too short.

It is a floor on the evidence rather than a constant. A model measured further
carries its own: MOSS-TTS-Nano is capped at fifteen, where one 63 s reply takes
six requests and no cut inside a sentence, against nine's ten and two. And
where this host's own samples held all the way to the longest it has tried, the
planner may offer one step past that, 1.25x — every sample is itself capped, so
a store left alone could otherwise only ever confirm the break it already has.

Inside the cap the grouping is a search, not a constant. The sentences are
grouped at every cut that moves a boundary, and the plan whose first word
comes soonest wins; a tie goes to the fewest boundaries. A streamed reply is
bounded by the lead instead, which lands in the same range on any model that
is keeping ahead.

Grouping can join sentences but never cut below one. So a sentence longer than
the cap is first cut at its own clause marks, and the pieces are grouped like
any others. Without that, a reply written as one long list reached the
renderer whole however it was planned. The cut is not free — every piece is
given sentence-final punctuation, so a clause mark cut at is spoken as a full
stop — which is why it is spent only where it buys an earlier first word.

## The sensors that answer it

| Sensor                            | Reads                                                                                                                                                                                                                                                             |
| --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `sensor.<model>_real_time_factor` | What the last synthesis actually cost on **this** host                                                                                                                                                                                                            |
| `sensor.<model>_playback_margin`  | The least audio the listener still held over the reply; negative ran dry                                                                                                                                                                                          |
| `sensor.<model>_delivery_mode`    | `whole`, `streaming`, `planned`, `unheld` or `buffered` — how the last reply was actually spoken; `whole` whenever it fitted one request, whichever plan chose it. Buffered is the exception: it is about releasing rather than cutting, so it survives the count |

**Margin is the one to watch.** It is the lowest the lead got over the whole
reply. Positive means the speaker always had something left to play. Negative
means it caught up with the renderer and waited, and the number is how many
seconds of silence that was.

It measures what the app handed to Home Assistant, which sits upstream of a
buffer with no back-pressure. It cannot see what the speaker actually
received, and a player that stops does not stop the render.

## If margin is negative

Usually the app's estimate of the model was too optimistic for that reply.
Not always: a sentence end is allowed to carry up to **Silence a sentence end
may carry** (1.5 s by default), because a gap there is heard as a pause
between sentences rather than as a fault, and the opening is not held against
it. Spending that allowance puts the margin slightly under zero by design. A
cut inside a sentence never earns it, and neither does a chunk-streaming
engine.

A planned reply corrects itself while it runs. Once it has produced a second of
audio, its own pace is believed over the fit's — including the request still
rendering — and the hold is taken from the dearer line. On a host made 1.4×
dearer than its fit, that turned 16 s of dry playback into under two. It also
learns from every request, so the next reply starts from a corrected fit, and
one stutter usually sorts itself out.

If a model keeps losing:

- **Name a Speaking mode for that model** instead of leaving it on `auto`:
  `unheld` for the first word as soon as one exists and stalls between
  sentences, `buffered`
  for no stalls at the cost of the longest wait.
- **Use a faster model**, which usually means the 40M.
- **Check the threads setting suits this model.** It is a trade, not a number
  to raise: four threads against two was 70% slower on MOSS and 1.4x _faster_
  on OmniVoice, both on the same four cores. The figures are per model in
  [Models](models.md).
- **Reset the model's measurements** from its card after changing the host — a
  new execution provider, a different thread count — so the fit starts from
  what the host is now rather than what it was.

## Cancelling

A listener that goes away — the pipeline cancelled, the socket closed — stops
the render at the engine's next checkpoint: a decode step, a diffusion step, a
codec chunk. Nothing an abandoned reply cost is recorded against the model.

The one thing the app cannot see is a media player that stopped. Home
Assistant's TTS cache drains the stream regardless, so that render finishes.
