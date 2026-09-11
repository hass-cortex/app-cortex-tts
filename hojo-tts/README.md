# Hojo TTS

On-device text-to-speech for Home Assistant, built on the
[Hojo TTS Light](https://github.com/HojoAI/Hojo-TTS-Light) ONNX models.

Runs on CPU. Ships the Chinese text pipeline the model does not have:
Traditional-to-Simplified glyph conversion and number/unit/time normalisation.
Both are required, not cosmetic — see `DOCS.md`.
