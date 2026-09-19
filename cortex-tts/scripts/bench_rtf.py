"""Measure every model's real-time factor on one host with one text set.

This is where the figures in docs/models.md come from — and only those. A
real-time factor is a property of the host, so a card shows what that host
measured or nothing at all (`cortex_tts.stats`); what this script produces is
documentation, for comparing models with each other on one machine.

Not the same statistic as the card: this divides each sentence's whole cost,
fixed part included, while a card fits the per-second part apart from it. On
one host the card reads a little lower, and that is not a disagreement.

The reference host is written down in docs/models.md ("The reference host") and
that is the authority on it; the figure is the median over the four sentence
lengths below. Run it against any app instance:

    CORTEX_TTS_KEY=... uv run python scripts/bench_rtf.py http://host:8771 out.json

Sequential, one model at a time (the host keeps one model resident), a warm-up
run discarded so model load and reference conditioning are not counted, then
three runs per sentence; the server's own X-Cortex-* headers are what is
recorded.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
import urllib.request

HOST = sys.argv[1].rstrip("/")
KEY = os.environ.get("CORTEX_TTS_KEY", "")
RUNS = 3
SENTENCES = {
    "short": "客廳的燈已經打開了。",
    "medium": "目前室內溫度 26.5°C，濕度 68%，冷氣設定在二十四度。",
    "long": "洗衣機洗好了。今天天氣多雲，最高溫 31°C，傍晚有 40% 的降雨機率，出門記得帶傘。",
    "story": "從前有一座山，山上有一間小廟，廟裡住著一位老和尚和一位小和尚。每天早上，老和尚都會帶著小和尚到山下的溪邊打水，然後一起回到廟裡念經。日子過得很平靜，直到有一天，山下來了一位陌生的旅人。",
}
# Cheapest first, so a run that is cut short still has the comparable pairs.
# The three cloning entries name a reference recording that has to exist on the
# host being measured; the third argument is how a run skips them.
MODELS = [
    ("hojo-40m", "hojo_zh_f_01"),
    ("moss-nano", "Yuewen"),
    ("hojo-80m-clone", "ya-ping"),
    ("omnivoice", "female-young"),
    ("qwen3-tts-0.6b", "vivian"),
    ("qwen3-tts-0.6b-clone", "ya-ping"),
]
# An optional third argument narrows the run to some model ids, comma-separated.
if len(sys.argv) > 3:
    wanted = set(sys.argv[3].split(","))
    MODELS = [pair for pair in MODELS if pair[0] in wanted]


def speak(model: str, voice: str, text: str) -> dict[str, float]:
    # The OpenAI-shaped route, which is the one that answers a finished file;
    # it reports the same `X-Cortex-*` figures.
    body = json.dumps(
        {"input": text, "model": model, "voice": voice, "response_format": "wav"}
    ).encode()
    req = urllib.request.Request(
        f"{HOST}/v1/audio/speech",
        data=body,
        method="POST",
        headers={"Authorization": f"Bearer {KEY}", "content-type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:  # noqa: S310 - our own host
        r.read()
        h = r.headers
        return {
            "rtf": float(h["X-Cortex-Rtf"]),
            "inference_ms": float(h["X-Cortex-Inference-Ms"]),
            "audio_s": float(h["X-Cortex-Audio-Seconds"]),
            "wall_s": time.perf_counter() - t0,
        }


results: dict[str, dict[str, dict[str, float]]] = {}
for model, voice in MODELS:
    results[model] = {}
    warm = speak(model, voice, SENTENCES["short"])  # load + conditioning, discarded
    print(f"{model}: warm-up {warm['wall_s']:.1f}s wall", flush=True)
    for name, text in SENTENCES.items():
        runs = [speak(model, voice, text) for _ in range(RUNS)]
        rtfs = [r["rtf"] for r in runs]
        results[model][name] = {
            "chars": len(text),
            "audio_s": statistics.median(r["audio_s"] for r in runs),
            "rtf_median": statistics.median(rtfs),
            "rtf_min": min(rtfs),
            "rtf_max": max(rtfs),
            "inference_ms_median": statistics.median(r["inference_ms"] for r in runs),
        }
        print(
            f"  {name:6s} chars={len(text):3d} "
            f"audio={results[model][name]['audio_s']:.1f}s rtf={rtfs}",
            flush=True,
        )
    medians = [row["rtf_median"] for row in results[model].values()]
    print(f"  => median {statistics.median(medians):.2f}", flush=True)

with open(sys.argv[2], "w", encoding="utf-8") as out:
    json.dump(results, out, indent=2, ensure_ascii=False)
print("done")
