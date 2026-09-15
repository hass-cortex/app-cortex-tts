// The lines the composer offers: Home Assistant's own shapes, so the panel
// is exercised with what it will actually be sent.

import { $, esc } from "./dom.js";

// One line each, and each a whole thought: a sample that strings unrelated
// clauses together to reach one more rewrite teaches the reader that the
// panel wants nonsense. Between them they still reach every pass — units and
// a percentage, a clock, a bare number, and a word Taiwan reads its own way.
const SAMPLES = [
  { label: "Lights and climate", text: "好的，客廳的燈已經打開了。目前室內溫度是 26.5°C，濕度 68%。" },
  { label: "Weather", text: "晚上 8:30 會下雨，降雨機率 70%，記得帶傘。" },
  { label: "Calendar", text: "明天星期三，你有 3 個行程，第一個在早上九點。" },
  { label: "English", text: "The washing machine is done. It is 26.5°C inside, humidity 68%." },
];

const clearPressed = () =>
  $("samples").querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", "false"));

export function init(onPick) {
  $("samples").innerHTML = SAMPLES.map((s, i) =>
    `<button class="chip" data-sample="${i}" aria-pressed="${i === 0}">${esc(s.label)}</button>`).join("");

  $("samples").addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-sample]");
    if (!btn) return;
    $("text").value = SAMPLES[Number(btn.dataset.sample)].text;
    clearPressed();
    btn.setAttribute("aria-pressed", "true");
    onPick();
  });

  // Typing makes the composer no longer any of the samples.
  return clearPressed;
}
