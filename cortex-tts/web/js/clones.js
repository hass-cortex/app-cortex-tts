// Cloned voices: the recordings every cloning model speaks with, and the
// transcripts that tell it which sounds map to which text.

import { call, json } from "./api.js";
import { $, confirmStep, esc, msg } from "./dom.js";
import { play } from "./player.js";
import { refreshVoices } from "./models.js";

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

const deleteLabel = (id) => (armed.has(id) ? "Confirm delete" : "Delete");

const row = (r) => `
  <div class="ref">
    <div class="ref-head">
      <span class="ref-name">${esc(r.name)}</span>
      <span class="ref-id">${esc(r.id)}${r.gender === "unknown" ? "" : ` · ${esc(r.gender)}`}</span>
      <span class="ref-meta">${Number(r.seconds).toFixed(1)}s · ${esc(r.language)}</span>
    </div>
    <textarea class="tr-edit" data-ref-tr="${esc(r.id)}" rows="1"
      aria-label="Transcript of ${esc(r.name)}" spellcheck="false">${esc(r.raw_transcript)}</textarea>
    <div class="tr-hint" data-base="tr-hint" data-ref-hint="${esc(r.id)}" aria-live="polite"></div>
    <div class="actions">
      <button class="sm" data-ref-play="${esc(r.id)}">Play</button>
      <button class="sm" data-ref-save="${esc(r.id)}" disabled>Save transcript</button>
      <button class="sm danger" data-ref-del="${esc(r.id)}">${deleteLabel(r.id)}</button>
    </div>
  </div>`;

export async function refresh() {
  const refs = await json("/references");
  $("refs").innerHTML = refs.length
    ? refs.map(row).join("")
    : '<div class="empty">No cloned voices yet.</div>';
  $("refs").querySelectorAll("textarea.tr-edit").forEach(autoGrow);
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
    // Show what the model will actually be told, not just what was typed.
    msg(hintFor(id), `Saved. The model is told: ${body.transcript}`, "ok");
  } catch (err) {
    msg(hintFor(id), err.message, "err");
    button.disabled = false;
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

  $("refs").addEventListener("click", (e) => {
    const playBtn = e.target.closest("button[data-ref-play]");
    if (playBtn) {
      // Fetched rather than handed to <audio> as a URL, so the key travels.
      call(`/references/${playBtn.dataset.refPlay}/audio`)
        .then((res) => res.blob())
        .then((blob) => play(URL.createObjectURL(blob), { revokable: true }))
        .catch((err) => msg($("refMsg"), err.message, "err"));
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
