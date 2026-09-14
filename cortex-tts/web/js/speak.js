// Synthesis: hand the composer's text to the model and play what comes back.

import { call } from "./api.js";
import { $, msg } from "./dom.js";
import { play, showStats } from "./player.js";
import { delivery, hasVoice, refreshModels } from "./models.js";
import { switches } from "./preview.js";

let inFlight = false;

/** Speak is offered when the chosen model has a voice and nothing is in flight. */
export function syncButton() {
  if (!inFlight) $("speak").disabled = !hasVoice();
}

async function speak() {
  inFlight = true;
  $("speak").disabled = true;
  msg($("speakMsg"), "");
  try {
    const res = await call("/speak", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: $("text").value,
        model: $("model").value,
        voice: $("voice").value,
        ...switches(),
        ...delivery(),
      }),
    });
    play(URL.createObjectURL(await res.blob()), { revokable: true });

    const seconds = Number(res.headers.get("X-Cortex-Audio-Seconds") || 0);
    const ms = Number(res.headers.get("X-Cortex-Inference-Ms") || 0);
    showStats([
      ["audio", `${seconds.toFixed(2)}s`],
      ["inference", `${Math.round(ms)}ms`],
      ["RTF", res.headers.get("X-Cortex-Rtf") || "—"],
      ["segments", res.headers.get("X-Cortex-Segments") || "—"],
    ]);
    // Synthesis loads and evicts models, so the table has just gone stale.
    await refreshModels();
  } catch (err) {
    msg($("speakMsg"), err.message, "err");
  } finally {
    inFlight = false;
    syncButton();
  }
}

export function init() {
  $("speak").addEventListener("click", speak);
}
