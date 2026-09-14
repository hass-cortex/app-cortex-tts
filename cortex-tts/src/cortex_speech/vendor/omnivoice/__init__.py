"""Vendored from `omnivoice` 0.2.1 (k2-fsa/OmniVoice), Apache-2.0 — see LICENSE.

Six modules are carried: the model itself, and the five helpers it imports.
The package on PyPI also requires gradio, accelerate, webdataset and
tensorboardx for its demo, its training loop and its data pipeline, none of
which inference touches, so depending on it would put a second web framework
in the image to render text.

Kept byte-for-byte apart from one mechanical edit: `models/omnivoice.py` and
`utils/` are flattened into this one package, so their `from omnivoice.utils.x`
imports are relative here. `modeling.py` is upstream's `models/omnivoice.py`.

Nothing here is called directly by the API layer — `cortex_speech.engine.omni`
wraps it to add the ONNX language model, reference caching and the text path.
"""
