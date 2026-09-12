// One control, three states.

import { $ } from "./dom.js";
import { THEME } from "./icons.js";

const ORDER = ["system", "light", "dark"];
const KEY = "cortex-theme";

// "system" is the absence of an override, so it stores nothing and the media
// query keeps working.
function apply(mode) {
  if (mode === "system") document.documentElement.removeAttribute("data-theme");
  else document.documentElement.setAttribute("data-theme", mode);
  $("theme").innerHTML = THEME[mode];
  $("theme").setAttribute("aria-label", `Appearance: ${mode} — click to change`);
  $("theme").dataset.mode = mode;
}

export function init() {
  let mode = "system";
  try { mode = localStorage.getItem(KEY) || "system"; } catch { /* blocked */ }
  if (!ORDER.includes(mode)) mode = "system";
  apply(mode);

  $("theme").addEventListener("click", () => {
    const next = ORDER[(ORDER.indexOf($("theme").dataset.mode) + 1) % ORDER.length];
    apply(next);
    try { localStorage.setItem(KEY, next); } catch { /* blocked */ }
  });
}
