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

Level matters more than length. The app levels every reference to −21 dBFS
RMS on the way in: a reference around −16 dBFS made MOSS generate hard-clipped
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
at all** — it still produces confident audio, just less like the person. If
you do not have the text, any decent speech-to-text will do; correcting it
afterwards costs nothing, because editing a transcript (`PATCH
/api/references/{id}`, or the panel's **Save transcript**) does not re-upload
the audio. The same request corrects the gender label (`female`, `male`,
`unknown`), which the panel shows as a select beside the id; it is a label for
the voice picker and nothing else reads it.

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
all; the two models that read the whole recording as a worked example,
OmniVoice and Qwen3-TTS, are the ones where a careless transcript shows.
