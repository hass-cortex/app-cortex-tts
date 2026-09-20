# Cortex TTS

[![GitHub Release](https://img.shields.io/github/v/release/hass-cortex/app-cortex-tts)](https://github.com/hass-cortex/app-cortex-tts/releases)
[![HA Version](https://img.shields.io/badge/HA-2026.3.0+-green.svg)](https://www.home-assistant.io/)
[![GitHub License](https://img.shields.io/github/license/hass-cortex/app-cortex-tts)](https://github.com/hass-cortex/app-cortex-tts/blob/main/LICENSE.md)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/hass-cortex/app-cortex-tts)

Home Assistant app for on-device text-to-speech. A catalog of models — Hojo
TTS Light, MOSS-TTS-Nano and OmniVoice — runs on your own CPU or
GPU, with built-in voices, voices designed from a set of attributes, or one
cloned from a short recording. No cloud, no per-character bill. The models
ship no text front-end, so the app carries one: numbers, units, clocks and
dates are written out, and Chinese is converted from Traditional glyphs and
respelled for Taiwan readings before synthesis. The admin UI shows that
prepared text beside the composer, with a ledger of every rewrite.

![The composer, and the prepared text beside it](images/composer.png)

## Models

Three catalog entries, one per family, none baked into the image; each is
downloaded from the app's own UI. The line-up, what each costs, how each
clones and which to pick is [docs/models.md](cortex-tts/docs/models.md).

## Installation

[![Open this app in your Home Assistant instance.](https://my.home-assistant.io/badges/supervisor_addon.svg)](https://my.home-assistant.io/redirect/supervisor_addon/?addon=24127962_cortex_tts&repository_url=https%3A%2F%2Fgithub.com%2Fhass-cortex%2Frepository)

Install and start the app, then follow the App Store page,
[cortex-tts/DOCS.md](cortex-tts/DOCS.md): download a model, add the
[companion integration](https://github.com/hass-cortex/cortex-tts), pair, and
assign a voice to a pipeline.

## Documentation

| Page                                                            | What it covers                                                    |
| --------------------------------------------------------------- | ----------------------------------------------------------------- |
| [App Store page](cortex-tts/DOCS.md)                            | Install, configure, troubleshoot                                  |
| [Models](cortex-tts/docs/models.md)                             | The line-up, what each costs, how each clones, which to pick      |
| [The text pipeline](cortex-tts/docs/text-pipeline.md)           | Why Traditional Chinese and numbers are rewritten, and into what  |
| [Cloned voices](cortex-tts/docs/cloning.md)                     | The recording, the transcript, the name                           |
| [Keeping up](cortex-tts/docs/delivery.md)                       | How a reply is paced, where the RTF threshold is, and the sensors |
| [Running it elsewhere](cortex-tts/docs/standalone.md)           | A faster CPU or a GPU outside Home Assistant OS                   |
| [HTTP API](cortex-tts/docs/api.md)                              | Using the app without the integration                             |
| [Integration](https://github.com/hass-cortex/cortex-tts#readme) | `tts.speak`, voice ids, speaking mode, the sensors                |

## Acknowledgements

[Hojo TTS Light](https://github.com/HojoAI/Hojo-TTS-Light),
[MOSS-TTS-Nano](https://github.com/OpenMOSS/MOSS-TTS-Nano) and
[OmniVoice](https://github.com/k2-fsa/OmniVoice) — the models this app serves.
OmniVoice's ONNX graph is
[rhasspy/omnivoice-onnx](https://huggingface.co/rhasspy/omnivoice-onnx).

## License

MIT — see [LICENSE.md](LICENSE.md).
