// The settings panel. These were addon options until every change cost a
// restart and every rename cost a rebuild; the app stores them itself now and
// this is where they are changed.

import { call, json } from "./api.js";
import { $, esc, fillPicker, msg, pressed } from "./dom.js";

let current = null;
let models = [];
let voices = [];
let onSaved = () => {};

/** Register what to do once a change is stored — the defaults just moved. */
export const whenSaved = (fn) => { onSaved = fn; };

const PROVIDERS = [
  ["auto", "auto — take a GPU if one answers"],
  ["cpu", "cpu"],
  ["cuda", "cuda — fail rather than fall back"],
];

/** Voices the chosen default model offers, plus the stored one if it is not among them. */
function voiceOptions() {
  const offered = voices.filter((v) => v.model_id === $("setModel").value);
  const ids = offered.map((v) => v.id);
  const options = offered.map(
    (v) => `<option value="${esc(v.id)}">${esc(v.name)}${v.language ? ` · ${esc(v.language)}` : ""}</option>`,
  );
  // A voice only exists once its model is downloaded, so the stored one may
  // name nothing this list knows. Dropping it would rewrite the setting just
  // by opening the page.
  const stored = current ? current.default_voice : "";
  if (stored && !ids.includes(stored)) {
    ids.unshift(stored);
    options.unshift(`<option value="${esc(stored)}">${esc(stored)} · not downloaded</option>`);
  }
  return {
    ids,
    markup: options.join("") || '<option value="">— first voice the model offers —</option>',
  };
}

function renderPickers() {
  fillPicker(
    $("setModel"),
    models.map((m) => `<option value="${esc(m.id)}">${esc(m.name)}</option>`).join(""),
    (id) => models.some((m) => m.id === id),
    current ? current.default_model : "",
  );
  const { ids, markup } = voiceOptions();
  fillPicker($("setVoice"), markup, (id) => ids.includes(id), current ? current.default_voice : "");
}

function render() {
  if (!current) return;
  $("setThreads").value = current.num_threads;
  $("setLoaded").value = current.max_loaded_models;
  $("setTemp").value = current.temperature;
  $("setProvider").innerHTML = PROVIDERS.map(
    ([id, label]) =>
      `<option value="${id}"${id === current.execution_provider ? " selected" : ""}>${label}</option>`,
  ).join("");
  $("setPreload").setAttribute("aria-pressed", String(current.preload));
  renderPickers();
}

/** Adopt a freshly fetched model and voice list without disturbing an edit. */
export function syncChoices(nextModels, nextVoices) {
  models = nextModels;
  voices = nextVoices;
  renderPickers();
}

/** Report which provider each resident model actually got. */
export function showProviders(health) {
  const inUse = Object.values(health.providers_in_use || {});
  const distinct = [...new Set(inUse)];
  $("providerNote").textContent = distinct.length
    ? `running on ${distinct.join(", ")}`
    : "nothing resident";
}

async function save() {
  const btn = $("setSave");
  btn.disabled = true;
  msg($("setMsg"), "Saving…");
  try {
    const res = await call("/settings", {
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        default_model: $("setModel").value,
        default_voice: $("setVoice").value,
        num_threads: Number($("setThreads").value),
        execution_provider: $("setProvider").value,
        max_loaded_models: Number($("setLoaded").value),
        temperature: Number($("setTemp").value),
        preload: pressed($("setPreload")),
      }),
    });
    const body = await res.json();
    current = body.settings;
    render();
    onSaved();
    // What came back is what is in force, which is not always what was typed:
    // a value out of range keeps its old one rather than rejecting the form.
    msg(
      $("setMsg"),
      body.reloaded
        ? "Saved. Resident models were dropped — the next reply loads them again."
        : "Saved.",
      "ok",
    );
  } catch (err) {
    msg($("setMsg"), err.message, "err");
  } finally {
    btn.disabled = false;
  }
}

/** Read what is in force and show it. */
export async function load() {
  current = await json("/settings");
  render();
}

export function init() {
  $("setModel").addEventListener("change", renderPickers);
  $("setPreload").addEventListener("click", () => {
    $("setPreload").setAttribute("aria-pressed", String(!pressed($("setPreload"))));
  });
  $("setSave").addEventListener("click", save);
}
