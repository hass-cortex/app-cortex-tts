# Cloned voices

Any model marked **clones** in the [line-up](models.md) takes a reference
recording: an audio file _and_ the words spoken in it. The recording supplies
the timbre; the transcript tells the model which sounds map to which text.
Neither alone defines a voice.

## Adding one

1. Download a cloning model.
2. In the app's **Cloned voices** panel, upload a few seconds of clean speech
   that starts talking immediately, and type _exactly_ what is said in it.
3. The voice appears in Home Assistant within seconds; no reload is needed.
   The same recording is a voice on **every** model that clones, under the
   same id.

Over the API this is `POST /api/references` (multipart: `audio`, `transcript`,
`name`, `language`, `gender`); see [the API](api.md).

## The recording

Recordings shorter than 2 s or longer than 20 s are rejected. How much of the
clip matters depends on the model:

- **Hojo 80M**: six seconds is the number that matters. Its speaker encoder
  reads exactly that much, padding a shorter clip with silence and discarding
  the rest of a longer one. Meanwhile the _whole_ recording is encoded into
  codec tokens that join the prompt for every sentence — so audio past the
  six-second mark slows down every synthesis while contributing nothing to the
  voice. Trim to six or seven seconds.
- **MOSS Nano**: there is no speaker encoder. The whole recording, as codec
  tokens, is the voice, so its length and content do shape the clone — and
  every second of it is paid for once per reference, then cached. A clean
  five to ten seconds is a good start.

Level matters more than length. The app levels every reference to −21 dBFS
RMS on the way in: a reference around −16 dBFS made MOSS generate hard-clipped
audio, and the clipping was baked into what the model produced, where no output
stage could undo it. A clean, evenly loud recording is worth more than a long
one.

WAV, FLAC and OGG are accepted; the recording is stored as 16-bit PCM WAV at
its original sample rate, downmixed to mono.

## The transcript

The transcript is not a label. A wrong one degrades the clone **with no error
at all** — it still produces confident audio, just less like the person. If
you do not have the text, any decent speech-to-text will do; correcting it
afterwards costs nothing, because editing a transcript (`PATCH
/api/references/{id}`, or the panel's **Save transcript**) does not re-upload
the audio.

What you type is the _raw transcript_; what the model is told is the same text
after the [text pipeline](text-pipeline.md) has run over it, so it is in the
same script and normalisation as the text it will be asked to say. The panel
shows the raw one for editing and reports the other on save. A transcript with
nothing pronounceable in it — punctuation only — is refused before the
recording is written.

## The name

The name you give it becomes the voice id, slugified, and it does not change
afterwards. A name in Latin letters gives a readable id; one in Chinese alone
folds to something like `voice-a0a77df0`, which still works but is not what you
want to read in an automation.

## What to expect

A cloned voice is a likeness, not a match — the [model notes](models.md) say
why for each model. Better source audio moves MOSS a little and the 80M not at
all.
