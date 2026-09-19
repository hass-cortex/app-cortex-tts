// The one audio element on the page, and the only thing that drives it.

import { $, msg, show } from "./dom.js";

/**
 * Play a URL, replacing whatever was loaded before.
 *
 * `autoplay: false` loads it and leaves it paused — for a caller that has
 * already played the audio itself and is only offering it again. Playing it
 * there put two copies of the same reply in the room at once.
 */
export function play(src, { revokable = false, autoplay = true } = {}) {
  const player = $("player");
  // A blob URL holds its audio in memory until it is revoked.
  if (player.dataset.url) {
    URL.revokeObjectURL(player.dataset.url);
    delete player.dataset.url;
  }
  if (revokable) player.dataset.url = src;
  player.src = src;
  show($("result"));
  if (!autoplay) return;
  player.play().catch((err) => {
    if (err.name === "AbortError") return; // a newer source replaced this one
    // Autoplay refused: the audio is loaded, it just needs a press.
    msg($("playerMsg"), err.name === "NotAllowedError" ? "Press play to hear it." : err.message, "warn");
  });
}

