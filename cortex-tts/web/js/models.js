// The models table and the two pickers above the composer. This module owns
// the model and voice lists: everything else asks it.

import { call, json } from "./api.js";
import {
  $,
  confirmStep,
  esc,
  fillPicker,
  languageName,
  languageOptions,
  msg,
  voiceGroups,
} from "./dom.js";

let models = [];
let voices = [];
// What the addon is configured to speak with. Until it answers, a picker has
// nothing to prefer and falls back to the first entry.
let defaults = { model: "", voice: "" };
// Models with an action in flight, and those whose Delete awaits its second
// click. Both are read at render time: the download poll redraws every card,
// and would otherwise re-enable a button mid-request.
const busy = new Set();
const armed = new Set();
let onVoiceChange = () => {};
let onModelsChange = () => {};
let onCatalogChange = () => {};

/** Register what to do when the chosen voice may have changed. */
export const whenVoiceChanges = (fn) => { onVoiceChange = fn; };
/** Told when the set of models changes, which is when a download finishes. */
export const whenModelsChange = (fn) => { onModelsChange = fn; };

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

/** Whether the chosen model offers a voice to speak with. */
export const hasVoice = () => voices.some((v) => v.model_id === $("model").value);

/** The language declared by the chosen voice, or "" when there is none. */
export function selectedVoiceLanguage() {
  const chosen = voices.find((v) => v.id === $("voice").value);
  return (chosen && chosen.language) || "";
}

/**
 * The language the preview reads the text as, or null to let it sniff.
 *
 * The same fallback the server applies to a synthesis — the language field,
 * else the voice's own — so the column shows what Speak will actually send.
 */
export function previewLanguage() {
  return $("language").value || selectedVoiceLanguage() || null;
}

const shortName = (m) => (m.name.match(/\d+M/) || [m.name])[0];

function statePill(m) {
  if (m.download_state === "queued") return '<span class="pill warn">queued</span>';
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
  if (m.download_state === "running" || m.download_state === "queued") return "";
  const button = (act, label, cls = "sm") =>
    `<button class="${cls}" data-act="${act}" data-id="${esc(m.id)}"${busy.has(m.id) ? " disabled" : ""}>${label}</button>`;
  if (!m.downloaded) return button("download", "Download");
  const load = m.loaded ? button("unload", "Unload") : button("load", "Load");
  // Only offered once there is something to reset; the host's measured RTF
  // goes stale when the model moves host.
  const reset = m.rtf && m.rtf.length ? ` ${button("reset-stats", "Reset RTF")}` : "";
  return `${load} ${button("delete", armed.has(m.id) ? "Confirm delete" : "Delete", "sm danger")}${reset}`;
}

function voiceCount(m) {
  const own = voices.filter((v) => v.model_id === m.id).length;
  if (own) return String(own);
  return m.cloning ? "upload one" : "—";
}

const KIND_LABEL = { builtin: "built-in", designed: "designed", reference: "cloned" };

/** The real-time factors this host measured, one row per voice.
 *
 * The median of the voice's recent requests on the provider in use, and the
 * verdict that follows from it: how a live reply in that voice is spoken
 * under `auto`. The comparison is shown, not only its answer, so a verdict
 * reads as the settings' threshold applied and not as a judgement on the
 * voice. A clone is its own row — its recording rejoins the prompt on every
 * synthesis, so it costs differently from a built-in voice on the same
 * model.
 */
function rtf(m) {
  const measured = m.rtf || [];
  if (!measured.length) {
    // The heading already says whose host; this only has to say "none yet".
    return `<p class="qual rtf-none">Not measured yet.</p>`;
  }
  const rows = measured.map((r) => {
    const n = r.samples === 1 ? "1 request" : `${r.samples} requests`;
    const label = `${KIND_LABEL[r.kind] || r.kind} · ${r.voice}`;
    // A voice is listed from its first request; the verdict waits for three.
    const settled = r.verdict !== null;
    const against = settled
      ? `${r.verdict === "streaming" ? "<" : "≥"} ${Number(r.threshold).toFixed(2)}`
      : "";
    return `<span class="rtf-kind">${esc(label)}</span>
      <span class="val">${Number(r.rtf).toFixed(2)}</span>
      <span class="qual">${against}</span>
      <span class="qual">${n}</span>
      <span class="qual">${esc(r.provider)}</span>
      <span class="rtf-verdict ${settled ? esc(r.verdict) : "pending"}">${settled ? esc(r.verdict) : "not yet"}</span>`;
  }).join("");
  return `<div class="rtf-rows">${rows}</div>`;
}

function renderCards() {
  $("models").innerHTML = models.map((m) => `
    <div class="card${m.loaded ? " loaded" : ""}">
      <div class="card-head">
        <strong>${esc(m.name)}</strong>
        ${statePill(m)}
      </div>
      <p>${esc(m.description)}</p>
      ${detail(m)}
      <div class="grid3">
        <span class="cap">Size</span>
        <span class="cap">Memory</span>
        <span class="cap">Voices</span>
        <span class="val">${Number(m.size_mb)} MB</span>
        <span class="val">${m.rss_hint_mb ? `${Number(m.rss_hint_mb)} MB` : "—"}</span>
        <span class="val">${esc(voiceCount(m))}</span>
      </div>
      <div class="measured">
        <span class="cap">RTF on this host</span>
        ${rtf(m)}
      </div>
      <div class="actions">${actions(m)}</div>
    </div>`).join("");


  // The pill names which models clone, rather than naming one of them.
  const cloners = models.filter((m) => m.cloning).map((m) => shortName(m));
  const pill = document.getElementById("clonesModels");
  if (pill) pill.textContent = cloners.length ? cloners.join(" · ") : "none";

  // And which of them never see the transcript, read off the capability
  // rather than written into the guide: a line naming a model by hand is
  // wrong the first time the line-up changes, and this one would be wrong
  // in the direction that tells someone their typing matters when it does
  // not reach the model at all.
  const deaf = models
    .filter((m) => m.cloning && !m.reads_reference_transcript)
    .map((m) => m.name);
  const note = document.getElementById("clonesTranscriptNote");
  if (note) {
    note.textContent = deaf.length
      ? `${deaf.join(", ")} ${deaf.length > 1 ? "ignore" : "ignores"} the transcript; `
        + "type it properly anyway, the other cloning models read it."
      : "";
  }

  const loaded = models.filter((m) => m.loaded);
  $("status").innerHTML = loaded.length
    ? loaded.map((m) => `<span class="pill ok">${esc(shortName(m))} loaded</span>`).join("")
    : '<span class="pill idle">no model resident</span>';
}



function selectedModel() {
  return models.find((m) => m.id === $("model").value) || null;
}

/**
 * Every language a cloning model reads, in catalog order.
 *
 * What the upload form offers, so a recording is labelled from the same
 * vocabulary the pickers above use — it had its own hard-coded three, which
 * stopped covering the models two of them ago.
 */
/** The language the reader filtered the voices by, or "" for any. */
export const selectedLanguage = () => $("language").value;

// A recording's tag may carry the region: a Taiwanese voice labelled zh-TW
// gets Taiwan readings by default for whatever it is asked to read, which
// the bare "zh" cannot promise. Offered ahead of the base code, most common
// first.
const REGIONS = { zh: ["zh-TW", "zh-CN"] };

export function cloningLanguages() {
  const seen = [];
  for (const model of models) {
    if (!model.cloning) continue;
    for (const code of model.languages || []) {
      for (const tag of [...(REGIONS[code] || []), code]) {
        if (!seen.includes(tag)) seen.push(tag);
      }
    }
  }
  return seen;
}

/**
 * The fields a request may carry beyond text, model and voice.
 *
 * The language goes on every model: it picks the text pipeline's locale, and
 * the server tells the model too where the model takes one. The instruction
 * goes only where the model declares it: sending it elsewhere is a 400, and
 * the caller never asked for one.
 */
export function delivery() {
  const model = selectedModel();
  const out = {};
  if ($("language").value) {
    out.language = $("language").value;
  }
  if (model && model.style_instruction && $("instruct").value.trim()) {
    out.instruct = $("instruct").value.trim();
  }
  return out;
}

function renderDeliveryFields() {
  const model = selectedModel();
  const spoken = !!(model && model.language_choice);
  const instruct = !!(model && model.style_instruction);

  // Shown only where it does something. One model takes an instruction, so a
  // permanent dead field would be dead on five cards out of six.
  $("instructField").hidden = !instruct;
  $("voiceRow").classList.toggle("lone", !instruct);
  if (!instruct) $("instruct").value = "";
  $("instructHint").textContent =
    "Plain language; leave empty for the voice's normal delivery.";

  // The language field stays on every model: it narrows the voice list and
  // picks how the text is prepared whatever the model does with it, and only
  // some are additionally told which language to read.
  $("languageHint").textContent = spoken
    ? "Filters the voices, sets text preparation, and tells the model which language to read."
    : "Filters the voices and sets text preparation; this model follows its voice's language.";

  const codes = model ? model.languages || [] : [];
  fillPicker(
    $("language"),
    '<option value="">— any —</option>' + languageOptions(codes),
    (id) => id === "" || codes.includes(id),
    "",
  );
}

/**
 * The selected model's voices, narrowed to the chosen language.
 *
 * Two kinds survive any filter. A voice with no language of its own —
 * OmniVoice's designed ones — reads whatever it is given. So does a clone:
 * the language on a reference labels what was said in the recording, not
 * what the voice may be asked to say, and hiding somebody's own uploaded
 * voice behind a filter is the one case that would actually annoy.
 *
 * When nothing matches, the whole list comes back rather than an empty one:
 * OmniVoice reads ten languages with nine designed voices, so asking
 * for German names no voice and is still a sensible request. There the timbre and the
 * language are separate things.
 */
function narrowed(all) {
  const code = $("language").value;
  const base = code.split("-")[0];
  const spoken = code
    ? all.filter(
        (v) =>
          !v.language ||
          v.source === "reference" ||
          v.language.split("-")[0] === base,
      )
    : all;
  const byLanguage = spoken.length ? spoken : all;
  // Gender is a label on the voice, so the filter is strict: an empty list
  // is the true answer, where the language fallback above is not.
  const gender = $("gender").value;
  return gender ? byLanguage.filter((v) => v.gender === gender) : byLanguage;
}

function renderVoicePicker() {
  const model = selectedModel();
  const cloning = model ? model.cloning : false;
  // Before the voice list, which it shapes.
  renderDeliveryFields();
  const all = voices.filter((v) => v.model_id === $("model").value);
  const mine = narrowed(all);
  fillPicker(
    $("voice"),
    mine.length
      ? voiceGroups(mine, model ? model.languages || [] : [], languageName)
      : `<option value="">— ${cloning ? "upload a recording below" : "no voices"} —</option>`,
    (id) => mine.some((v) => v.id === id),
    defaults.voice,
  );

  // Read the kinds off the voices rather than off the capability flags. A
  // model may have more than one, the flags have been wrong before — adding
  // `designed_voices` left this saying "cloned" for a model with nine
  // designed voices — and a list cannot disagree with itself.
  const order = ["builtin", "designed", "reference"];
  const kinds = order
    .filter((kind) => mine.some((v) => v.source === kind))
    .map((kind) => KIND_LABEL[kind]);
  $("voiceLabel").textContent = kinds.length
    ? `Voice — ${kinds.join(" + ")}${mine.length ? ` (${mine.length})` : ""}`
    : "Voice";
  onVoiceChange();

  $("clones").classList.toggle("idle", !cloning);
  // Naming them matters here and nowhere else: this line is only read by
  // someone looking for cloning on a model that has none, and "does not
  // clone" on its own leaves them to guess which one does.
  const others = models.filter((m) => m.cloning).map((m) => m.name);
  $("clonesNote").textContent = cloning
    ? `Feeding ${model ? model.name : "the selected model"}.`
    : others.length
      ? `Idle — this model does not clone. These do: ${others.join(", ")}.`
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
  onModelsChange();
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
  "reset-stats": (id) => [`/models/${id}/stats`, { method: "DELETE" }],
};

export function init() {
  $("model").addEventListener("change", renderVoicePicker);
  $("language").addEventListener("change", renderVoicePicker);
  $("gender").addEventListener("change", renderVoicePicker);
  $("voice").addEventListener("change", () => onVoiceChange());

  $("models").addEventListener("click", async (e) => {
    const btn = e.target.closest("button[data-act]");
    if (!btn) return;
    const { act, id } = btn.dataset;
    if (busy.has(id)) return;
    if (act === "delete" && !confirmStep(armed, id, renderCards)) return;
    busy.add(id);
    renderCards();
    msg($("modelMsg"), "");
    try {
      await call(...ENDPOINTS[act](id));
      await refreshModels();
      await refreshVoices();
    } catch (err) {
      msg($("modelMsg"), err.message, "err");
    } finally {
      busy.delete(id);
      renderCards();
    }
  });
}
