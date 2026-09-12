# Keeping up

Whether a model can speak a reply while the rest is still being written, and
how to tell on your own host. The setting itself — **Speaking mode**, per
model — lives in the Home Assistant integration; what follows is the
performance model behind it.

## Three ways to speak a reply

- **Buffered**: render the whole reply, then play it. The longest wait, and it
  never stalls.
- **Sentence by sentence**: one request per finished sentence, played as each
  arrives.
- **Sentences in groups**: the same, but sentences written while a request is
  in flight are sent together, so there are fewer joins. On a model that emits
  audio while a request is still rendering (only MOSS-TTS-Nano) that removes
  the pauses outright; on one that returns each request whole it trades several
  short waits for fewer longer ones.

Both streamed modes only work if the model renders faster than the audio
plays. One that does not falls behind a little more with every sentence, and a
long reply stutters to a halt near the end. Within a request, MOSS-TTS-Nano
also emits audio before a sentence is finished (chunk streaming); the other
models send each request's audio whole. See [Models](models.md).

## Buffered until you have measured

**Every model is buffered until you say otherwise.** Streaming is not switched
on by the catalog figure, because that figure is one host's: the reference
4-core Home Assistant VM, where only the 40M outruns playback (0.67) and MOSS
(1.06) and the 80M (1.42) do not. A faster CPU runs every model two to three
times faster, a GPU helps beside a weak CPU and hurts beside a strong one —
not a constant factor, so no figure from elsewhere predicts yours. A model streamed on a host where it
cannot keep up is the exact failure the setting exists to prevent.

Use a model for a while, read `sensor.<model>_real_time_factor`, and turn
streaming on for that model if it sits comfortably under **0.5** — half of real
time rather than all of it, because 1.0 only breaks even and any host is
briefly busy with something else. Then watch `sensor.<model>_playback_margin`.

## The sensors that answer it

Each model gets diagnostic sensors in Home Assistant; two of them answer
"is it keeping up":

| Sensor                            | Reads                                                             |
| --------------------------------- | ----------------------------------------------------------------- |
| `sensor.<model>_real_time_factor` | What the last synthesis actually cost on **this** host            |
| `sensor.<model>_playback_margin`  | The least audio the listener still held when a piece arrived late |

**Margin is the one to watch.** It opens at the head start and falls every
time a piece of the reply arrives later than the audio already sent covers.
Positive means the speaker always had something left to play — it won.
Negative means the speaker caught up with the renderer and waited, which is
what stuttering is, and the number is how many seconds of silence that was.

It is measured only on a reply that arrived in pieces; one handed over whole
never had a piece that could be late, and reads unknown.

## If margin is negative

- **Shorter replies** fix it outright. The deficit grows with length.
- **Turn streaming off for that model** — its **Speaking mode** back to
  buffered, where it started. Nothing plays until it is all rendered, so the
  wait is longer, there are no gaps, and repeated text comes back from the
  cache.
- **Raise the head start** in the same place — but only on MOSS-TTS-Nano. It
  banks opening seconds before playback begins, spending wait to buy margin:
  at a rendering rate of R, a reply of L seconds needs (R − 1) × L banked, so
  two seconds covers a 44-second reply at 1.045x. It is only charged to
  replies long enough to need it, and a grouped reply whose full length is
  known before the first request is charged only what a reply that long can
  lose. The bank fills from the audio it receives, so on a model that returns
  each request whole it is all or nothing: a bank smaller than the first
  request is already full the moment that request lands, and a larger one
  waits for the second request and moves the whole reply back.
- **Use a faster model**, which usually means the 40M.
- **Check the threads setting is 2.** Four was measured 70% slower than two on
  a four-core host; more is not better here.

Nothing turns streaming on by itself, so a negative margin means someone chose
it for this model. The sensors are how you find out whether that choice still
holds — a host that has since gained another app is a slower host.

## What a request should hold

Coalescing removed the cost of sending one sentence at a time — a fresh
prefill for every sentence, 0.37 s of dead air per boundary on MOSS — but a
request that is too long costs more than it saves: the model attends over
everything it has generated so far, so the bill grows with the square of the
request. Measured on MOSS-TTS-Nano, the same 55-second story sent in pieces of
different sizes, as the fraction by which rendering fell behind playback:

| Per request         | Behind playback |
| ------------------- | --------------- |
| ~1 s (one sentence) | +41%            |
| ~9 s                | +4.1%           |
| ~19 s               | +5.6%           |
| ~28 s               | +6.5%           |
| ~55 s (one request) | +17.8%          |

A valley with a cliff on either side; the integration's grouped mode sits in
the flat part of it, at about 14 seconds of speech per request.

## What streaming costs you

Home Assistant caches TTS, but not on every path, and the difference decides
almost nothing here — which is worth saying plainly, because it is easy to
conclude otherwise.

`tts.speak` and `assist_satellite.announce` are cached by the hash of their
text, in memory and on disk, **whatever mode the model is in**. So a repeated
announcement is rendered once and replayed for free either way; streaming does
not cost you that.

The one path that loses the cache is an **Assist pipeline reply on a model with
streaming turned on**: Home Assistant stores it under a fresh id and never
writes it to disk, so it is re-synthesised every time. That matters less than
it sounds, because a spoken answer is different every time anyway — the cache
could not have helped.

So pick the mode on whether the model keeps up, not on caching:

| What you are saying                  | Text                 | Use                                                 |
| ------------------------------------ | -------------------- | --------------------------------------------------- |
| Announcements, timers, notifications | Repeats              | Either — it is cached the first time and free after |
| Assist answering questions           | Different every time | Buffered, or streaming if the model measures fast   |
| Long text read aloud                 | Different and long   | Streaming, only if measured under 0.5               |

## Why a stream is MP3

`/api/speak/stream` answers MP3: a bare sequence of self-describing frames,
with no container, no length field and no index, which is the only honest
thing to send when the length is not known yet. A WAV stream has to declare a
length before the audio exists, and a general-purpose player given the maximal
one waits for a file it believes is six hours long. FLAC and OGG both need a
size or an index written before the audio exists, so they are refused. The
`X-Cortex-Bitrate` header is the one measurement that exists before the first
sample, and it is what turns a byte count into a duration downstream.
