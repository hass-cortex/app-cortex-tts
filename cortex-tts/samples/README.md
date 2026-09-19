# Sample reference recording

One recording, to try cloning with before committing your own. Upload it in
the app's **Cloned voices** panel with the fields below, then pick the voice
it creates on any model that clones.

| Field | Value |
| ----- | ----- |
| File | `zh-tw-female.wav` |
| Transcript | `清晨的阳光跳上窗台，让我们带着笑脸去赴一场春天的约会吧。` |
| Language | `zh-TW` |
| Gender label | `female` |

24 kHz mono 16-bit, 6.12 s. Name it whatever you like — the id is derived
from the name at upload and is what a synthesis request then asks for.

## Why this clip

It is here to be a good example, not just a working one, so it satisfies the
rules the panel states rather than merely passing validation:

- **6.12 s**, inside the six-to-ten-second band. Shorter still uploads, and a
  clip under six seconds is what most first attempts are, which is exactly
  why the sample should not be one.
- **Ends on a finished sentence**, with the silence after it. A clip cut by a
  recorder's clock is refused, because it teaches the model that sentences
  end mid-word.
- **Transcript matches the audio exactly**, in Simplified. Nothing converts
  it for you: no model here fails on Traditional glyphs, they read them as
  the wrong sounds, and a transcript the model mis-reads teaches the clone
  the wrong mapping. See [cloning](../docs/cloning.md).

The audio carries no metadata: it has `fmt ` and `data` chunks and nothing
else. Verified, not assumed — and it holds for any recording the store
accepts, because an upload is decoded and written back out as 24 kHz mono
PCM, which discards whatever the original container carried. There is one
`sf.write` in the store and it is in `add`; editing a reference changes its
name, transcript, gender or language and never its audio, and no route
replaces the audio of an existing one. To change a recording you delete it
and upload again — through the same re-encode.
