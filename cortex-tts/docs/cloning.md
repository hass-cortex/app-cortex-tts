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
3. The voice appears in Home Assistant within seconds; no reload is needed.
   The same recording is a voice on **every** model that clones, under the
   same id.

Over the API this is `POST /api/references` (multipart: `audio`, `transcript`,
`name`, `language`, `gender`); see [the API](api.md).

To hear what a clone sounds like before recording your own, the repository
ships one: [`samples/`](../samples/) has a 6-second Taiwanese Mandarin clip
and the exact fields to upload it with.

### Why the end matters, and why a cut one is refused

A clip cut by a clock rather than by the speaker is refused on upload: "the
recording is still speaking when it ends". It is not fussiness. The models
that clone best read the **whole** recording as a worked example before they
say anything — Qwen3-TTS puts its codec frames in the prompt beside its
transcript, OmniVoice has no speaker encoder at all — so a clip that stops
mid-word teaches one thing above all: _that_ is how this speaker finishes a
sentence. Nothing downstream can tell. The clone renders with no error, and
what comes out drifts, trails off and clips its own endings.

It is worse when the transcript is tidied up afterwards. Writing what you hear
and adding a full stop turns a fragment into a claim that the sentence
completed, which is the one thing the recording proves it did not.

The check is the last tenth of a second against the clip's own average. On
eight real uploads the seven cut at a recorder's 7.00 s limit measured between
−7.1 and **+7.5** dB — one ended louder than its own average, which is what
stopping mid-vowel looks like — while the one that ran to its own end measured
−25.3 dB. Anything above −15 dB is refused.

## The recording

Recordings shorter than 2 s or longer than 20 s are rejected. How much of the
clip matters depends on the model:

- **Hojo 80M**: six seconds is the number that matters. Its speaker encoder
  reads exactly that much, padding a shorter clip with silence and discarding
  the rest of a longer one. Meanwhile the _whole_ recording is encoded into
  codec tokens that join the prompt for every sentence — so audio past the
  six-second mark slows down every synthesis while contributing nothing to the
  voice. Trim to six or seven seconds.
- **MOSS Nano** and **OmniVoice**: there is no speaker encoder. The whole
  recording, as codec tokens, is the voice, so its length and content do shape
  the clone — and every second of it is paid for once per reference, then
  cached. A clean five to ten seconds is a good start. OmniVoice asks for
  3–10 s and warns past 20; that is the band to aim at for both.
- **Qwen3-TTS**: both at once, and the only model here that does. A speaker
  encoder turns the recording into one 1024-value x-vector, _and_ the whole
  recording joins the prompt as codec frames beside its transcript, as a
  worked example the model reads before the text it has to say. So the
  transcript matters more here than anywhere else, and length costs on every
  synthesis the way MOSS's does.

Level matters more than length. An upload is stored exactly as it arrived;
MOSS and Qwen3-TTS level a reference to −21 dBFS RMS as they read it, and the
80M and OmniVoice do not level at all: a reference around −16 dBFS made MOSS generate hard-clipped
audio, and the clipping was baked into what the model produced, where no output
stage could undo it. A clean, evenly loud recording is worth more than a long
one.

WAV, FLAC and OGG are accepted; the recording is stored as 16-bit PCM WAV at
its original sample rate, downmixed to mono, as `<id>.wav` beside a
`references.json` index. The Home Assistant app keeps them in
`/share/cortex-tts/references/`, which is in its backups and reachable over the
`share` Samba folder; a [standalone](standalone.md) install keeps them under
`DATA_DIR/references` unless `REFERENCES_DIR` says otherwise. Each model keeps
its encoding of the recording beside it (`<id>.<fingerprint>.<model>.*`), so a
model that was unloaded does not re-encode the reference when it comes back;
these files are disposable and are recreated on demand.

## The transcript

The transcript is not a label. A wrong one degrades the clone **with no error
at all** — it still produces confident audio, just less like the person.

Not every cloning model is told what the recording says — which one is not,
and what that changes, is in [Models](models.md#moss-tts-nano). It makes no
difference to how you fill the field: one recording is a voice on all of
them at once, so the strict case is the one to type for. If
you do not have the text, any decent speech-to-text will do; correcting it
afterwards costs nothing, because editing a transcript (`PATCH
/api/references/{id}`, or the panel's **Save transcript**) does not re-upload
the audio. The same request corrects the display name and the gender label
(`female`, `male`, `unknown`), which the panel shows as a select beside the
id; it is a label for the voice picker and nothing else reads it. It also
corrects the language — which is not only a label, because changing it runs
the stored transcript through the text pipeline again.

**The transcript is stored exactly as you type it** — on upload and on edit
alike, because a transcript means the same thing whichever way it was written.
The panel asks for what the recording says, and a pass that rewrites the
answer is deciding the recording said something else, which is the one error a
reference cannot survive. Changing the **language** leaves it alone for the
same reason: the tag says what was spoken, it is not an instruction to
re-transcribe the recording. A transcript with nothing pronounceable in it —
punctuation only — is still refused before the recording is written.

**Write Chinese in Simplified.** Nothing converts it for you any more, and
no model here fails on Traditional glyphs — they read them as the wrong
sounds, which is worse, because a transcript the model mis-reads teaches the
clone the wrong mapping from the audio. Measured over 14 sentences, Traditional
text scored 32% character error against 4% converted; the figures and the
reasoning are in [the text pipeline](text-pipeline.md#traditional-chinese).

This gives something up, and it is worth knowing which way. A reference whose
speaker says the Taiwan _xì yáng_ for 夕陽 is read by a mainland-trained model
as _xī yáng_, and writing the transcript 系阳 would have made its reading match
the recording. The [text pipeline](text-pipeline.md) can do that substitution
automatically — but it is a guess, and a measured-unreliable one: it carries
`在为` and matches characters rather than words, so a reference reading
正在為你查詢 came back as `正在维你查询`, claiming a _wéi_ the recording does
not contain. A wrong rewrite and a missing one are the same failure — the
transcript stops matching the audio — so the one that is not a guess wins. If
you want the stand-in, type it: 系阳 in the box is stored as 系阳.

## The name and the id

The name you give it seeds the voice **id**, slugified — and the id does not
change afterwards, ever. The id is what a synthesis request names, so a
pipeline or an automation that speaks in this voice holds it; re-deriving it
from a new name would break every one of them without an error anywhere. A
name in Latin letters therefore gives a readable id; one in Chinese alone
folds to something like `voice-a0a77df0`, which still works but is not what you
want to read in an automation. Pick the name with that in mind at upload,
because it is the only moment it decides anything.

The **name** itself is free, and editable: type over it in the panel (it saves
when you leave the box) or send `PATCH /api/references/{id}` with a `name`.
Nothing keys on it — it is what the voice picker shows and no more — so the
two part company at the first rename, which is intended. Renaming to nothing
leaves the voice known by its id.

## What to expect

A cloned voice is a likeness, not a match — the [model notes](models.md) say
why for each model. Better source audio moves MOSS a little and the 80M not at
all; the two models that read the whole recording as a worked example,
OmniVoice and Qwen3-TTS, are the ones where a careless transcript shows.
