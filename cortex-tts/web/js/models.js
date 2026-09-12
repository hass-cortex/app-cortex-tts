// The models table and the two pickers above the composer. This module owns
// the model and voice lists: everything else asks it.

import { call, json } from "./api.js";
import { $, esc, fillPicker, msg } from "./dom.js";

let models = [];
let voices = [];
// What the addon is configured to speak with. Until it answers, a picker has
// nothing to prefer and falls back to the first entry.
let defaults = { model: "", voice: "" };
let pollTimer = null;
let onVoiceChange = () => {};
let onCatalogChange = () => {};

/** Register what to do when the chosen voice may have changed. */
export const whenVoiceChanges = (fn) => { onVoiceChange = fn; };

/**
 * Register what to do when the model or voice lists have been re-read.
 *
 * A download finishing is what changes them, and it arrives on this module's
 * own poll timer rather than through anything the page clicked.
 */
export const whenCatalogChanges = (fn) => { onCatalogChange = fn; };

/** Every model in the catalog, as last fetched. */
export const catalog = () => models;

/** Every voice across downloaded models, as last fetched. */
export const knownVoices = () => voices;

/** The language declared by the chosen voice, or "" when there is none. */
export function selectedVoiceLanguage() {
  const chosen = voices.find((v) => v.id === $("voice").value);
  return (chosen && chosen.language) || "";
}

const shortName = (m) => (m.name.match(/\d+M/) || [m.name])[0];

function statePill(m) {
  if (m.download_state === "running") {
    return `<span class="pill warn">downloading ${Math.round(m.download_percent)}%</span>`;
  }
  if (m.download_state === "failed" && !m.downloaded) return '<span class="pill bad">failed</span>';
  if (m.loaded) return `<span class="pill ok">loaded${m.provider ? ` · ${esc(m.provider)}` : ""}</span>`;
  if (m.downloaded) return '<span class="pill idle">on disk</span>';
  return '<span class="pill idle">not downloaded</span>';
}

function detail(m) {
  if (m.download_state === "running") {
    return `<div class="bar"><span style="width:${Number(m.download_percent)}%"></span></div>`;
  }
  if (m.download_state === "failed" && !m.downloaded) {
    return `<p class="failed">${esc(m.download_error || "Download failed.")}</p>`;
  }
  return "";
}

function actions(m) {
  if (m.download_state === "running") return "";
  if (!m.downloaded) {
    return `<button class="sm" data-act="download" data-id="${esc(m.id)}">Download</button>`;
  }
  const load = m.loaded
    ? `<button class="sm" data-act="unload" data-id="${esc(m.id)}">Unload</button>`
    : `<button class="sm" data-act="load" data-id="${esc(m.id)}">Load</button>`;
  return `${load} <button class="sm danger" data-act="delete" data-id="${esc(m.id)}">Delete</button>`;
}

function voiceCount(m) {
  const own = voices.filter((v) => v.model_id === m.id).length;
  if (own) return String(own);
  return m.cloning ? "upload one" : "—";
}

function renderCards() {
  $("models").innerHTML = models.map((m) => `
    <div class="card${m.loaded ? " loaded" : ""}">
      <div class="card-head">
        <strong>${esc(m.name)}</strong>
        ${m.recommended ? '<span class="pill ok">recommended</span>' : ""}
        ${statePill(m)}
      </div>
      <p>${esc(m.description)}</p>
      ${detail(m)}
      <div class="grid4">
        <span class="cap">Size</span>
        <span class="cap">RTF</span>
        <span class="cap">Memory</span>
        <span class="cap">Voices</span>
        <span class="val">${Number(m.size_mb)} MB</span>
        <span class="val">${m.rtf_hint ? Number(m.rtf_hint) : "—"}</span>
        <span class="val">${m.rss_hint_mb ? `${Number(m.rss_hint_mb)} MB` : "—"}</span>
        <span class="val">${esc(voiceCount(m))}</span>
      </div>
      <div class="actions">${actions(m)}</div>
    </div>`).join("");

  const anyRunning = models.some((m) => m.download_state === "running");
  if (anyRunning && !pollTimer) pollTimer = setInterval(refreshModels, 1500);
  if (!anyRunning && pollTimer) { clearInterval(pollTimer); pollTimer = null; }

  // The pill names which models clone, rather than naming one of them.
  const cloners = models.filter((m) => m.cloning).map((m) => shortName(m));
  const pill = document.getElementById("clonesModels");
  if (pill) pill.textContent = cloners.length ? cloners.join(" · ") : "none";

  const loaded = models.filter((m) => m.loaded);
  $("status").innerHTML = loaded.length
    ? loaded.map((m) => `<span class="pill ok">${esc(shortName(m))} loaded</span>`).join("")
    : '<span class="pill idle">no model resident</span>';
}

function selectedModel() {
  return models.find((m) => m.id === $("model").value) || null;
}

function renderVoicePicker() {
  const model = selectedModel();
  const cloning = model ? model.cloning : false;
  const mine = voices.filter((v) => v.model_id === $("model").value);
  fillPicker(
    $("voice"),
    mine.length
      ? mine.map((v) => `<option value="${esc(v.id)}">${esc(v.name)}${v.language ? ` · ${esc(v.language)}` : ""}</option>`).join("")
      : `<option value="">— ${cloning ? "upload a recording below" : "no voices"} —</option>`,
    (id) => mine.some((v) => v.id === id),
    defaults.voice,
  );

  // Capabilities are independent (ADR 0001): a model may have bundled voices
  // AND clone. Reading either one as a kind mislabels the model that has both.
  const kinds = [];
  if (model && model.builtin_voices) kinds.push("built in");
  if (cloning) kinds.push("cloned");
  $("voiceLabel").textContent = kinds.length
    ? `Voice — ${kinds.join(" + ")}${mine.length ? ` (${mine.length})` : ""}`
    : "Voice";
  $("speak").disabled = mine.length === 0;
  onVoiceChange();

  $("clones").classList.toggle("idle", !cloning);
  $("clonesNote").textContent = cloning
    ? `Feeding ${model ? model.name : "the selected model"}.`
    : "Idle — the selected model does not clone.";
}

function renderModelPicker() {
  const usable = models.filter((m) => m.downloaded);
  fillPicker(
    $("model"),
    usable.length
      ? usable.map((m) => `<option value="${esc(m.id)}">${esc(m.name)}</option>`).join("")
      : '<option value="">— download a model first —</option>',
    (id) => usable.some((m) => m.id === id),
    defaults.model,
  );
  renderVoicePicker();
}

/** Read the configured defaults, so the pickers open where the addon points. */
export async function loadDefaults() {
  defaults = await json("/defaults");
}

export async function refreshModels() {
  models = await json("/models");
  renderCards();
  renderModelPicker();
  onCatalogChange();
}

export async function refreshVoices() {
  voices = await json("/voices");
  renderCards();
  renderVoicePicker();
  onCatalogChange();
}

const ENDPOINTS = {
  download: (id) => [`/models/${id}/download`, { method: "POST" }],
  load: (id) => [`/models/${id}/load`, { method: "POST" }],
  unload: (id) => [`/models/${id}/unload`, { method: "POST" }],
  delete: (id) => [`/models/${id}`, { method: "DELETE" }],
};

export function init() {
  $("model").addEventListener("change", renderVoicePicker);
  $("voice").addEventListener("change", () => onVoiceChange());

  $("models").addEventListener("click", async (e) => {
    const btn = e.target.closest("button[data-act]");
    if (!btn) return;
    const { act, id } = btn.dataset;
    if (act === "delete" && !confirm("Delete this model's files from disk?")) return;
    btn.disabled = true;
    msg($("modelMsg"), "");
    try {
      await call(...ENDPOINTS[act](id));
      await refreshModels();
      await refreshVoices();
    } catch (err) {
      msg($("modelMsg"), err.message, "err");
    } finally {
      btn.disabled = false;
    }
  });
}
