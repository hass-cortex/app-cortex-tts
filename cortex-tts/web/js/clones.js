// Cloned voices: the recordings every cloning model speaks with, and the
// transcripts that tell it which sounds map to which text.

import { call, json } from "./api.js";
import { $, confirmStep, esc, fillPicker, languageName, languageOptions, msg } from "./dom.js";
import { cloningLanguages, refreshVoices, selectedLanguage } from "./models.js";

// A transcript must be readable in full without an inner scrollbar, so the
// box is sized to its content rather than to a fixed row count.
function autoGrow(box) {
  box.style.height = "auto";
  box.style.height = `${box.scrollHeight}px`;
}

// What each transcript looked like when it was rendered, so Save only lights
// up on a real change and a failed save can be retried.
const saved = new Map();
// Voices whose Delete awaits its second click.
const armed = new Set();

const GENDERS = ["female", "male", "unknown"];

const deleteLabel = (id) => (armed.has(id) ? "Confirm delete" : "Delete");

// The tags the cloning models read, plus the stored one if it is not among
// them: a tag typed over the API must not be silently relabelled by opening
// the page. `data-language` carries that stored tag, because the options are
// filled after the card is drawn rather than with it.
function fillLanguage(select) {
  const stored = select.dataset.language;
  const codes = cloningLanguages();
  if (stored && !codes.includes(stored)) codes.unshift(stored);
  fillPicker(select, languageOptions(codes), (c) => codes.includes(c), stored);
}

// Every card's language select, refilled from the current vocabulary.
//
// A card is drawn from `/references`, which answers on its own fetch and
// normally before `/models` — so at the moment the markup is built there is
// no line-up to read the languages off, and a select filled there and never
// again would offer the one tag it already has and could not be changed.
function fillRowLanguages() {
  for (const select of $("refs").querySelectorAll("select[data-ref-language]")) {
    fillLanguage(select);
  }
}

const row = (r) => `
  <div class="ref">
    <div class="ref-head">
      <input class="ref-name" data-ref-name="${esc(r.id)}" value="${esc(r.name)}"
        aria-label="Name of ${esc(r.id)}" spellcheck="false">
      <span class="ref-id" title="The id a request names; renaming does not change it.">${esc(r.id)}</span>
      <select class="ref-gender" data-ref-gender="${esc(r.id)}" aria-label="Gender label of ${esc(r.id)}">
        ${GENDERS.map((g) => `<option value="${g}"${g === r.gender ? " selected" : ""}>${g}</option>`).join("")}
      </select>
      <select class="ref-gender" data-ref-language="${esc(r.id)}" data-language="${esc(r.language)}"
        aria-label="Language of ${esc(r.id)}"></select>
      <span class="ref-meta">${Number(r.seconds).toFixed(1)}s</span>
    </div>
    <textarea class="tr-edit" data-ref-tr="${esc(r.id)}" rows="1"
      aria-label="Transcript of ${esc(r.id)}" spellcheck="false">${esc(r.raw_transcript)}</textarea>
    <div class="tr-hint" data-base="tr-hint" data-ref-hint="${esc(r.id)}" aria-live="polite"></div>
    <audio class="ref-audio" data-ref-audio="${esc(r.id)}" controls hidden></audio>
    <div class="actions">
      <button class="sm" data-ref-play="${esc(r.id)}">Play</button>
      <button class="sm" data-ref-save="${esc(r.id)}" disabled>Save transcript</button>
      <button class="sm danger" data-ref-del="${esc(r.id)}">${deleteLabel(r.id)}</button>
    </div>
  </div>`;

// Set once the reader picks a language for an upload themselves, after which
// the field stops following the one above it: they have said something more
// specific than the filter could.
let refLangChosen = false;

/**
 * Point every language control in this panel at the cloning line-up.
 *
 * Which languages can be cloned into changes when the line-up does, so the
 * vocabulary is read off the models rather than hard-coded — and the models
 * arrive on a fetch of their own, after the cards are drawn. Whoever tells us
 * the line-up changed also un-freezes the cards.
 *
 * The upload form has a second link to the Language control above: its value
 * follows the filter until the reader overrides it here, because someone who
 * has just filtered the voices to Japanese is usually about to upload a
 * Japanese recording.
 */
export function syncLanguages() {
  const codes = cloningLanguages();
  const picker = $("refLang");
  // Read the wanted value before replacing the options: assigning innerHTML
  // drops the old `<option>`s and resets `value` to the first of the new
  // ones, so asking afterwards returns what we are about to overwrite.
  const wanted = refLangChosen ? picker.value : selectedLanguage();
  picker.innerHTML = languageOptions(codes);
  picker.value = codes.includes(wanted) ? wanted : codes[0] || "";
  fillRowLanguages();
}

export async function refresh() {
  const refs = await json("/references");
  $("refs").innerHTML = refs.length
    ? refs.map(row).join("")
    : '<div class="empty">No cloned voices yet.</div>';
  $("refs").querySelectorAll("textarea.tr-edit").forEach(autoGrow);
  fillRowLanguages();
  saved.clear();
  for (const r of refs) saved.set(r.id, r.raw_transcript);
}

const hintFor = (id) => $("refs").querySelector(`[data-ref-hint="${id}"]`);
const boxFor = (id) => $("refs").querySelector(`textarea[data-ref-tr="${id}"]`);
const deleteFor = (id) => $("refs").querySelector(`button[data-ref-del="${id}"]`);

async function saveTranscript(button) {
  const id = button.dataset.refSave;
  const box = boxFor(id);
  button.disabled = true;
  try {
    const res = await call(`/references/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ transcript: box.value }),
    });
    const body = await res.json();
    saved.set(id, body.raw_transcript);
    box.value = body.raw_transcript;
    autoGrow(box);
    box.classList.remove("dirty");
    // Nothing to quote back: the box is what the model is told, verbatim.
    msg(hintFor(id), "Saved. The model is told exactly this.", "ok");
  } catch (err) {
    msg(hintFor(id), err.message, "err");
    button.disabled = false;
  }
}

async function saveLanguage(select) {
  const id = select.dataset.refLanguage;
  select.disabled = true;
  try {
    const res = await call(`/references/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ language: select.value }),
    });
    const body = await res.json();
    select.dataset.language = body.language;
    select.value = body.language;
    // The server may have tidied the tag into one the list does not carry.
    fillLanguage(select);
    msg(hintFor(id), `Speaks ${languageName(body.language)}.`, "ok");
  } catch (err) {
    msg(hintFor(id), err.message, "err");
  } finally {
    select.disabled = false;
  }
}

async function saveName(input) {
  const id = input.dataset.refName;
  input.disabled = true;
  try {
    const res = await call(`/references/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: input.value }),
    });
    const body = await res.json();
    // An empty name is not a failure: the voice falls back to its id, which
    // is the same rule the upload form has.
    input.value = body.name;
    msg(hintFor(id), `Now called ${body.name}.`, "ok");
    await refreshVoices();
  } catch (err) {
    msg(hintFor(id), err.message, "err");
  } finally {
    input.disabled = false;
  }
}

async function saveGender(select) {
  const id = select.dataset.refGender;
  select.disabled = true;
  try {
    const res = await call(`/references/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ gender: select.value }),
    });
    const body = await res.json();
    select.value = body.gender;
    msg(hintFor(id), `Labelled ${body.gender}.`, "ok");
    await refreshVoices();
  } catch (err) {
    msg(hintFor(id), err.message, "err");
  } finally {
    select.disabled = false;
  }
}

async function remove(button) {
  const id = button.dataset.refDel;
  const relabel = () => { const b = deleteFor(id); if (b) b.textContent = deleteLabel(id); };
  if (!confirmStep(armed, id, relabel)) return;
  relabel();
  button.disabled = true;
  try {
    await call(`/references/${id}`, { method: "DELETE" });
    await refresh();
    await refreshVoices();
  } catch (err) {
    msg($("refMsg"), err.message, "err");
    button.disabled = false;
  }
}

async function add() {
  const file = $("refFile").files[0];
  if (!file) { msg($("refMsg"), "Choose a recording first.", "err"); return; }
  const form = new FormData();
  form.append("name", $("refName").value || file.name);
  form.append("transcript", $("refText").value);
  form.append("language", $("refLang").value);
  form.append("gender", $("refGender").value);
  form.append("audio", file);

  $("refAdd").disabled = true;
  try {
    await call("/references", { method: "POST", body: form });
    msg($("refMsg"), "Voice added.", "ok");
    $("refName").value = "";
    $("refText").value = "";
    $("refFile").value = "";
    await refresh();
    await refreshVoices();
  } catch (err) {
    msg($("refMsg"), err.message, "err");
  } finally {
    $("refAdd").disabled = false;
  }
}

export function init() {
  $("refLang").addEventListener("change", () => { refLangChosen = true; });
  $("refs").addEventListener("input", (e) => {
    const box = e.target.closest("textarea[data-ref-tr]");
    if (!box) return;
    const id = box.dataset.refTr;
    autoGrow(box);
    const dirty = box.value !== saved.get(id);
    box.classList.toggle("dirty", dirty);
    const save = $("refs").querySelector(`button[data-ref-save="${id}"]`);
    if (save) save.disabled = !dirty || !box.value.trim();
    msg(hintFor(id), "");
  });

  // A short field commits on `change` — blur or Enter — the way the two
  // selects beside it do. Only the transcript earns a button of its own: it
  // is long enough that leaving the box is not a decision to save it.
  $("refs").addEventListener("change", (e) => {
    const name = e.target.closest("input[data-ref-name]");
    if (name) saveName(name);
    const gender = e.target.closest("select[data-ref-gender]");
    if (gender) saveGender(gender);
    const language = e.target.closest("select[data-ref-language]");
    if (language) saveLanguage(language);
  });
  $("refs").addEventListener("click", (e) => {
    const playBtn = e.target.closest("button[data-ref-play]");
    if (playBtn) {
      // The card's own player, not the composer's: hearing a recording must
      // not replace the utterance someone just rendered up there. Fetched
      // rather than handed to <audio> as a URL, so the key travels.
      const id = playBtn.dataset.refPlay;
      const audio = $("refs").querySelector(`audio[data-ref-audio="${CSS.escape(id)}"]`);
      call(`/references/${id}/audio`)
        .then((res) => res.blob())
        .then((blob) => {
          if (audio.dataset.url) URL.revokeObjectURL(audio.dataset.url);
          audio.dataset.url = URL.createObjectURL(blob);
          audio.src = audio.dataset.url;
          audio.hidden = false;
          return audio.play();
        })
        .catch((err) => {
          if (err.name !== "NotAllowedError") msg($("refMsg"), err.message, "err");
        });
      $("stats").textContent = "";
      return;
    }

    const save = e.target.closest("button[data-ref-save]");
    if (save) return saveTranscript(save);

    const del = e.target.closest("button[data-ref-del]");
    if (del && !del.disabled) return remove(del);
  });

  $("refAdd").addEventListener("click", add);
}
