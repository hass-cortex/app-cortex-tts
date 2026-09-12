# Cortex TTS

![Supports amd64 Architecture][amd64-shield] ![Supports aarch64 Architecture][aarch64-shield] ![Supports armv7 Architecture][armv7-shield]

On-device text-to-speech for Home Assistant, built on the
[Hojo TTS Light](https://github.com/HojoAI/Hojo-TTS-Light) and
[MOSS-TTS-Nano](https://github.com/OpenMOSS/MOSS-TTS-Nano) ONNX models.

Runs on CPU. Ships the Chinese text pipeline the model does not have:
Traditional-to-Simplified glyph conversion and number/unit/time normalisation.
Both are required, not cosmetic — the **Documentation** tab explains why.

[amd64-shield]: https://img.shields.io/badge/amd64-yes-green.svg
[aarch64-shield]: https://img.shields.io/badge/aarch64-no-red.svg
[armv7-shield]: https://img.shields.io/badge/armv7-no-red.svg
