# Hojo TTS

[![GitHub Release](https://img.shields.io/github/v/release/hass-cortex/app-hojo-tts)](https://github.com/hass-cortex/app-hojo-tts/releases)
[![HA Version](https://img.shields.io/badge/HA-2026.3.0+-green.svg)](https://www.home-assistant.io/)
[![GitHub License](https://img.shields.io/github/license/hass-cortex/app-hojo-tts)](https://github.com/hass-cortex/app-hojo-tts/blob/main/LICENSE.md)

Home Assistant app providing on-device text-to-speech on the
[Hojo TTS Light](https://github.com/HojoAI/Hojo-TTS-Light) ONNX models — the
40M with fifteen built-in voices, and the 80M that clones a voice from a few
seconds of reference audio. CPU only; no cloud, no per-character bill.

## The text pipeline is the point

The models ship no text handling, and for Chinese that is the difference
between a voice and noise. Measured on this app's own test set: Traditional
glyphs sent straight to the model score a 32% character error rate against 4%
once converted to Simplified, and unnormalised sensor text — `26.5°C`,
`14:35`, `68%` — lands between 36% and 50%.

So the app carries what the models lack:

- **Traditional → Simplified** glyph conversion (`t2s`, glyph-only, so
  Taiwanese wording survives).
- **Normalisation** of numbers, units, dates and clock literals into spoken
  Chinese: `26.5°C` becomes `攝氏二十六點五度`.

Both run before synthesis, in that order — normalisation emits Traditional
number words, so it has to precede the pass that makes them pronounceable.

Each is a request flag: the caller decides _whether_ a pass runs, and the
Home Assistant integration, which knows the language, sets the defaults. What
the app does infer is narrower — which script to spell numbers in, from the
dominant script of the text itself, because "48" has to become "forty-eight"
in an English sentence and 四十八 in a Chinese one and no flag carries that.

## Admin UI

The ingress panel doubles as the test bench. It shows the prepared text —
what the model is actually asked to say — beside a ledger of what each pass
rewrote, because that rewrite is the whole product and a voice that mangles
a temperature is the symptom you would otherwise have to guess at.

![The composer, and the prepared text beside it](images/composer.png)

It also manages the models — download, load, evict — and the cloned voices,
including the transcript each reference recording is bound to.

![The models, with what each costs](images/models.png)

## Installation

See [hojo-tts/DOCS.md](hojo-tts/DOCS.md) for install, configuration,
discovery and troubleshooting.

The companion integration lives in
[hass-cortex/hojo-tts](https://github.com/hass-cortex/hojo-tts). Install it
through HACS first — Home Assistant has no Hojo TTS platform without it — and
the running app is then discovered by itself.

## Acknowledgements

- [Hojo TTS Light](https://github.com/HojoAI/Hojo-TTS-Light) — the ONNX
  models this app serves.
- [OpenCC](https://github.com/BYVoid/OpenCC) — the Traditional/Simplified
  conversion.
- [sentence-stream](https://github.com/OHF-Voice/sentence-stream) — the
  sentence splitter Home Assistant's own streaming engines use.

## License

MIT — see [LICENSE.md](LICENSE.md).
