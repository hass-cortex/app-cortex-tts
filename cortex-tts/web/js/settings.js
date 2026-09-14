// The settings panel. These were addon options until every change cost a
// restart and every rename cost a rebuild; the app stores them itself now and
// this is where they are changed.

import { call, json } from "./api.js";
import { $, esc, fillPicker, msg, voiceOption } from "./dom.js";

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

// What each field is called on the page, for naming one the server refused.
const LABELS = {
  default_model: "Default model",
  default_voice: "Default voice",
  num_threads: "Inference threads",
  execution_provider: "Execution provider",
  max_loaded_models: "Models kept in memory",
  temperature: "Sampling temperature",
  preload: "Preload",
};

/** Voices the chosen default model offers, plus the stored one if nothing downloaded offers it. */
function voiceOptions() {
  const model = $("setModel").value;
  const offered = voices.filter((v) => v.model_id === model);
  const ids = offered.map((v) => v.id);
  const options = offered.map(voiceOption);
  // A voice only exists once its model is downloaded, so the stored one may
  // name nothing this page knows. Dropping it would rewrite the setting just
  // by opening the page — but only while the stored model is still the one
  // chosen; picking another model is picking one of its voices.
  const stored = current ? current.default_voice : "";
  const known = voices.some((v) => v.id === stored);
  if (stored && !known && model === current.default_model) {
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
  $("setPreload").checked = current.preload;
  // What is stored is the truth here — just read, or just saved — so the
  // pickers are rebuilt rather than kept on whatever they were showing.
  $("setModel").innerHTML = "";
  $("setVoice").innerHTML = "";
  renderPickers();
}

/** Adopt a freshly fetched model and voice list without disturbing an edit. */
export function syncChoices(nextModels, nextVoices) {
  models = nextModels;
  voices = nextVoices;
  // The bound counts catalog entries, so the catalog says how high it goes.
  if (models.length) $("setLoaded").max = models.length;
  renderPickers();
  showProviders(models.filter((m) => m.provider).map((m) => m.provider));
}

/** Report which providers the resident models actually got. */
export function showProviders(inUse) {
  const distinct = [...new Set(inUse)];
  $("providerNote").textContent = distinct.length
    ? `running on ${distinct.join(", ")}`
    : "nothing resident";
}

// An empty box is a field not being sent, never a zero or an empty name.
const numberOrOmit = (id) => ($(id).value.trim() === "" ? undefined : Number($(id).value));
const textOrOmit = (id) => $(id).value || undefined;

async function save() {
  const btn = $("setSave");
  btn.disabled = true;
  msg($("setMsg"), "Saving…");
  try {
    const res = await call("/settings", {
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        default_model: textOrOmit("setModel"),
        default_voice: textOrOmit("setVoice"),
        num_threads: numberOrOmit("setThreads"),
        execution_provider: $("setProvider").value,
        max_loaded_models: numberOrOmit("setLoaded"),
        temperature: numberOrOmit("setTemp"),
        preload: $("setPreload").checked,
      }),
    });
    const body = await res.json();
    current = body.settings;
    render();
    onSaved();
    // What came back is what is in force, which is not always what was typed:
    // a value out of range keeps its old one rather than rejecting the form,
    // and the server names which fields it did that to.
    const kept = (body.ignored || []).map((field) => LABELS[field] || field);
    let text = "Saved.";
    if (body.reloaded) text += " Resident models were dropped — the next reply loads them again.";
    if (kept.length) text += ` Kept previous value for: ${kept.join(", ")}.`;
    msg($("setMsg"), text, kept.length ? "warn" : "ok");
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
  $("setSave").addEventListener("click", save);
}
