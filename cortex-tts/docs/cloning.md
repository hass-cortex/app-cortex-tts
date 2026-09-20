# Cloned voices

Any model marked **clones** in the [line-up](models.md) takes a reference
recording: an audio file _and_ the words spoken in it. The recording supplies
the timbre; the transcript tells the model which sounds map to which text.
Neither alone defines a voice.

## Adding one

1. Download a cloning model.
2. In the app's **Cloned voices** panel, upload a few seconds of clean speech
   that starts talking immediately and **ends on a finished sentence**, and
   type _exactly_ what is said in it.
3. The voice appears in Home Assistant within seconds, no reload needed — on
   **every** model that clones, under the same id.

Over the API this is `POST /api/references` (multipart: `audio`, `transcript`,
`name`, `language`, `gender`; see [the API](api.md)). To hear a clone before
recording your own, [`samples/`](../samples/) has a 6-second Taiwanese
Mandarin clip and the exact fields to upload it with.

### Why the end matters, and why a cut one is refused

A clip cut by a clock rather than by the speaker is refused on upload: "the
recording is still speaking when it ends". The models that clone best read the
**whole** recording as a worked example before they say anything — OmniVoice
puts its codec frames in the prompt beside the transcript and has no speaker
encoder at all — so a clip that stops mid-word teaches that _that_ is
how this speaker finishes a sentence. Nothing downstream can tell: the clone
renders with no error, and what comes out drifts, trails off and clips its own
endings. Tidying the transcript with a full stop afterwards makes it worse,
turning a fragment into a claim that the sentence completed.

The check is the last tenth of a second against the clip's own average. On
eight real uploads the seven cut at a recorder's 7.00 s limit measured between
−7.1 and **+7.5** dB — one ended louder than its own average, which is what
stopping mid-vowel looks like — and the one that ran to its own end measured
−25.3 dB. Anything above −15 dB is refused.

## The recording

Recordings shorter than 2 s or longer than 20 s are rejected. On both MOSS
Nano and OmniVoice the whole recording is the voice, so its length and
content shape the clone, paid once per reference and cached. A clean five to
ten seconds is a good start; OmniVoice asks for 3–10 s and warns past 20, the
band to aim at for both. How each model reads the clip is in
[Models](models.md).

Level matters more than length. An upload is stored exactly as it arrived; MOSS
levels a reference to −21 dBFS RMS as it reads it, and OmniVoice does not
level at all. A reference around −16 dBFS made MOSS
generate hard-clipped audio, baked into what the model produced where no
output stage could undo it. A clean, evenly loud recording is worth more than
a long one.

WAV, FLAC and OGG are accepted; the recording is stored as 16-bit PCM WAV at
its original sample rate, downmixed to mono, as `<id>.wav` beside a
`references.json` index. The Home Assistant app keeps them in
`/share/cortex-tts/references/`, which is in its backups and reachable over
the `share` Samba folder; a [standalone](standalone.md) install keeps them
under `DATA_DIR/references` unless `REFERENCES_DIR` says otherwise. Each model
keeps its encoding of the recording beside it (`<id>.<fingerprint>.<model>.*`),
so a model that was unloaded does not re-encode it when it comes back; those
files are disposable and recreated on demand.

## The transcript

The transcript is not a label. A wrong one degrades the clone **with no error
at all** — confident audio, just less like the person.

Not every cloning model is told what the recording says — which one is not is
in [Models](models.md#moss-tts-nano) — but one recording is a voice on all of
them at once, so type for the strict case. If you do not have the text, any
decent speech-to-text will do; correcting it afterwards costs nothing, because
editing a transcript (`PATCH /api/references/{id}`, or the panel's **Save
transcript**) does not re-upload the audio. The same request corrects the
display name, the gender label (`female`, `male`, `unknown` — a label for the
voice picker, nothing else reads it) and the language, which is not only a
label: changing it runs the stored transcript through the text pipeline again.

**The transcript is stored exactly as you type it**, on upload and on edit
alike. The panel asks for what the recording says, and a pass that rewrites
the answer is deciding the recording said something else — the one error a
reference cannot survive. Changing the **language** leaves it alone for the
same reason. A transcript with nothing pronounceable in it — punctuation only
— is refused before the recording is written.

**Write Chinese in Simplified.** Nothing converts it for you, and no model here
fails on Traditional glyphs — they read them as the wrong sounds, which
teaches the clone the wrong mapping from the audio; the figures (32% character
error against 4%) are in
[the text pipeline](text-pipeline.md#traditional-chinese).

This gives something up. A speaker who says the Taiwan _xì yáng_ for 夕陽 is
read by a mainland-trained model as _xī yáng_, and a transcript of 系阳 would
match the recording. The text pipeline's stand-in is not applied here because
it is a guess — a reference reading 正在為你查詢 came back as `正在维你查询`,
claiming a _wéi_ the recording does not contain — and a wrong rewrite and a
missing one are the same failure. If you want the stand-in, type it: 系阳 in
the box is stored as 系阳.

## The name and the id

The name you give it seeds the voice **id**, slugified, and the id never
changes: it is what a synthesis request names, so every pipeline and
automation that speaks in this voice holds it, and re-deriving it from a new
name would break them all without an error anywhere. A name in Latin letters
gives a readable id; one in Chinese alone folds to something like
`voice-a0a77df0`, which works but is not what you want in an automation.
Upload is the only moment the name decides anything.

The **name** itself is free and editable: type over it in the panel (it saves
when you leave the box) or send `PATCH /api/references/{id}` with a `name`.
Nothing keys on it — it is what the voice picker shows. Renaming to nothing
leaves the voice known by its id.

## What to expect

A cloned voice is a likeness, not a match; [Models](models.md) says why for
each model. Better source audio moves MOSS a little; OmniVoice, which reads the whole recording as a worked example, is where a
careless transcript shows.
