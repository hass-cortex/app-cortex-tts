# Cortex TTS

[![GitHub Release](https://img.shields.io/github/v/release/hass-cortex/app-cortex-tts)](https://github.com/hass-cortex/app-cortex-tts/releases)
[![HA Version](https://img.shields.io/badge/HA-2026.3.0+-green.svg)](https://www.home-assistant.io/)
[![GitHub License](https://img.shields.io/github/license/hass-cortex/app-cortex-tts)](https://github.com/hass-cortex/app-cortex-tts/blob/main/LICENSE.md)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/hass-cortex/app-cortex-tts)

Home Assistant app providing on-device text-to-speech on the
[Hojo TTS Light](https://github.com/HojoAI/Hojo-TTS-Light) ONNX models — the
40M with fifteen built-in voices, and the 80M that clones a voice from a few
seconds of reference audio. CPU only; no cloud, no per-character bill.

## Features

- **Runs on your hardware** — CPU only, no cloud, no API key, no
  per-character bill.
- **Fifteen built-in voices** on the 40M, Chinese and English, or clone one on
  the 80M from a few seconds of reference audio.
- **The text pipeline the models lack** — Traditional-to-Simplified conversion
  and number, unit, date and clock expansion, without which this model is
  unintelligible for Chinese.
- **An admin UI that shows its work** — the prepared text and a ledger of
  every rewrite, beside the composer.
- **Speaks sentence by sentence**, so a long reply starts playing after its
  first sentence rather than its last.
- **Discovered by Home Assistant** through the Supervisor, so the companion
  integration needs no address or key typed in.

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

**1. Install this app.**

[![Open this app in your Home Assistant instance.](https://my.home-assistant.io/badges/supervisor_addon.svg)](https://my.home-assistant.io/redirect/supervisor_addon/?addon=24127962_cortex_tts&repository_url=https%3A%2F%2Fgithub.com%2Fhass-cortex%2Frepository)

Start it, open its Web UI and download a model — nothing is baked into the
image.

**2. Install the companion integration.**

[![Open your Home Assistant instance and open a repository inside the Home Assistant Community Store.](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=hass-cortex&repository=cortex-tts&category=integration)

It lives in [hass-cortex/cortex-tts][integration] and Home Assistant has no Hojo
TTS platform without it: no integration, no discovery card, and no voices in
the pipeline picker. Restart Home Assistant after adding it, and the running
app is then discovered by itself.

See [cortex-tts/DOCS.md](cortex-tts/DOCS.md) for the rest — configuration,
cloned voices and troubleshooting.

[integration]: https://github.com/hass-cortex/cortex-tts

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

## Contributing

Issues and pull requests are welcome.

- [`cortex-tts/CONTRIBUTING.md`](cortex-tts/CONTRIBUTING.md) — dev setup, the four
  gates, and the commit convention.
- [`AGENTS.md`](AGENTS.md) — the module tree, the cross-module guarantees, and
  the endpoint reference.
- [`cortex-tts/CONTEXT.md`](cortex-tts/CONTEXT.md) — the domain vocabulary both of
  those use. Worth reading before naming anything: most nouns here already
  mean two things.

## Acknowledgements

- [Hojo TTS Light](https://github.com/HojoAI/Hojo-TTS-Light) — the ONNX
  models this app serves.
- [OpenCC](https://github.com/BYVoid/OpenCC) — the Traditional/Simplified
  conversion.
- [sentence-stream](https://github.com/OHF-Voice/sentence-stream) — the
  sentence splitter Home Assistant's own streaming engines use.

## License

MIT — see [LICENSE.md](LICENSE.md).
