// The one audio element on the page, and the only thing that drives it.

import { $, esc, msg, show } from "./dom.js";

/** Play a URL, replacing whatever was loaded before. */
export function play(src, { revokable = false } = {}) {
  const player = $("player");
  // A blob URL holds its audio in memory until it is revoked.
  if (player.dataset.url) {
    URL.revokeObjectURL(player.dataset.url);
    delete player.dataset.url;
  }
  if (revokable) player.dataset.url = src;
  player.src = src;
  show($("result"));
  player.play().catch((err) => {
    if (err.name === "AbortError") return; // a newer source replaced this one
    // Autoplay refused: the audio is loaded, it just needs a press.
    msg($("speakMsg"), err.name === "NotAllowedError" ? "Press play to hear it." : err.message, "warn");
  });
}

/** Show the numbers that came back with a synthesis, or clear them. */
export function showStats(rows = []) {
  $("stats").innerHTML = rows.map(([label, value]) =>
    `<span>${esc(label)} ${esc(value)}</span>`).join("");
}
