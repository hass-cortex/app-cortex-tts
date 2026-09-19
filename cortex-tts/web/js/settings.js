// The settings panel. These were addon options until every change cost a
// restart and every rename cost a rebuild; the app stores them itself now and
// this is where they are changed.

import { call, json } from "./api.js";
import { $, esc, fillPicker, msg, voiceOption } from "./dom.js";

let current = null;
let models = [];
let voices = [];
let onSaved = () => {};

// The four text switches a rule can answer, and the chip name each one
// wears on the composer — so a rule reads the way the preview does.
const SWITCHES = [
  ["normalize_text", "normalise numbers"],
  ["expand_numbers", "bare numbers"],
  ["convert_script", "to Simplified"],
  ["taiwan_readings", "Taiwan readings"],
];
const THREE_WAY = [
  ["", "as decided"],
  ["true", "on"],
  ["false", "off"],
];

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
  idle_unload_seconds: "Unload when idle",
  max_sentence_pause: "Silence a sentence end may carry",
  temperature: "Sampling temperature",
  preload: "Preload",
  text_rules: "Text switch rules",
};

function threeWay(name, value) {
  const chosen = value === null || value === undefined ? "" : String(value);
  return `<select data-switch="${name}">${THREE_WAY.map(
    ([v, label]) => `<option value="${v}"${v === chosen ? " selected" : ""}>${label}</option>`,
  ).join("")}</select>`;
}

function renderRule(rule) {
  const modelOptions = [`<option value="">any model</option>`]
    .concat(models.map((m) => `<option value="${esc(m.id)}"${m.id === rule.model ? " selected" : ""}>${esc(m.name)}</option>`))
    .join("");
  return `<div class="rule">
    <div><label>Model</label><select data-rule="model">${modelOptions}</select></div>
    <div><label>Language</label><input data-rule="language" list="setLangList" placeholder="any" value="${esc(rule.language || "")}"></div>
    ${SWITCHES.map(([name, label]) => `<div><label>${esc(label)}</label>${threeWay(name, rule[name])}</div>`).join("")}
    <button class="danger" data-rule="remove" aria-label="Remove rule">Remove</button>
  </div>`;
}

function renderRules() {
  if (!current) return;
  const rules = current.text_rules || [];
  $("setRules").innerHTML = rules.length
    ? rules.map(renderRule).join("")
    : '<div class="rules-empty">No rules: every switch is the pipeline\'s call.</div>';
  // The tags the models declare, plus the Chinese ones a rule is likely to
  // want, offered as suggestions; anything else may be typed.
  const tags = new Set(["zh-TW", "zh-CN", "zh-Hant", "zh-Hans"]);
  for (const m of models) for (const l of m.languages || []) tags.add(l);
  $("setLangList").innerHTML = [...tags].sort().map((t) => `<option value="${esc(t)}">`).join("");
}

/** The rules as the rows now read, sent whole: the list replaces what was stored. */
function collectRules() {
  return [...$("setRules").querySelectorAll(".rule")].map((row) => {
    const rule = {
      model: row.querySelector('[data-rule="model"]').value || null,
      language: row.querySelector('[data-rule="language"]').value.trim() || null,
    };
    for (const sel of row.querySelectorAll("[data-switch]")) {
      rule[sel.dataset.switch] = sel.value === "" ? null : sel.value === "true";
    }
    return rule;
  });
}

function addRule() {
  // Edits in the other rows survive: the new row is appended to what the
  // rows say now, not to what was last stored.
  current = { ...current, text_rules: [...collectRules(), { model: null, language: null }] };
  renderRules();
}

function removeRule(row) {
  const rules = collectRules();
  const index = [...$("setRules").querySelectorAll(".rule")].indexOf(row);
  rules.splice(index, 1);
  current = { ...current, text_rules: rules };
  renderRules();
}

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
  $("setIdle").value = current.idle_unload_seconds;
  $("setSentencePause").value = current.max_sentence_pause;
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
  renderRules();
}

/** Adopt a freshly fetched model and voice list without disturbing an edit. */
export function syncChoices(nextModels, nextVoices) {
  models = nextModels;
  voices = nextVoices;
  // The bound counts catalog entries, so the catalog says how high it goes.
  if (models.length) $("setLoaded").max = models.length;
  renderPickers();
  renderRules();
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
        idle_unload_seconds: numberOrOmit("setIdle"),
        max_sentence_pause: numberOrOmit("setSentencePause"),
        temperature: numberOrOmit("setTemp"),
        preload: $("setPreload").checked,
        text_rules: collectRules(),
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
  $("setRuleAdd").addEventListener("click", addRule);
  $("setRules").addEventListener("click", (event) => {
    const button = event.target.closest('[data-rule="remove"]');
    if (button) removeRule(button.closest(".rule"));
  });
}
