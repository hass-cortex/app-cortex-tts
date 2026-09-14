# Cortex TTS

![Supports amd64 Architecture][amd64-shield] ![Supports aarch64 Architecture][aarch64-shield] ![Supports armv7 Architecture][armv7-shield]

Text-to-speech that runs on your own hardware, for Home Assistant's Assist
pipelines and `tts.speak`. A catalog of models to choose between — a small
one with fifteen built-in voices that keeps up on a Home Assistant box, and
larger ones that clone a voice from a short recording, design one from a
handful of attributes, or read ten languages — downloaded from the app's own
UI on first use.

Pair it with the [Cortex TTS integration](https://github.com/hass-cortex/cortex-tts)
from HACS, which discovers the app by itself and puts every voice in the
pipeline picker. The **Documentation** tab covers installation, settings and
which model to pick; a machine with a faster CPU or a GPU can run the app
outside Home Assistant OS, which cannot use a GPU.

[amd64-shield]: https://img.shields.io/badge/amd64-yes-green.svg
[aarch64-shield]: https://img.shields.io/badge/aarch64-no-red.svg
[armv7-shield]: https://img.shields.io/badge/armv7-no-red.svg
