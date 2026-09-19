// The vocabulary every panel shares: how an element is found, how text is
// made safe, and the one shape a message takes.

export const $ = (id) => document.getElementById(id);

export const show = (el, on = true) => { el.hidden = !on; };

/** Escape text for interpolation into markup. */
export function esc(value) {
  return String(value).replace(/[&<>"]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
}

export const pressed = (el) => el.getAttribute("aria-pressed") === "true";

/**
 * Pause every player on the page but the one that just started.
 *
 * Two panels own audio elements — the composer's, and one per cloned voice
 * so that hearing a recording does not discard the utterance rendered above
 * — and they play into the same room. Two at once is never what the press
 * meant, and the second is heard as the first having gone wrong.
 *
 * Bound to `play` in the capture phase, so it holds however playback began:
 * a panel's own button, the element's native controls, or a script. `play`
 * does not bubble, which is why it is captured rather than listened for at
 * the target. Neither panel calls it, and neither has to know the other
 * exists.
 */
export function soloAudio(keep) {
  for (const el of document.querySelectorAll("audio")) {
    if (el !== keep) el.pause();
  }
}

// What a language code is called. The catalog ships codes; a list of bare
// codes is not a thing anyone can choose from. Shared rather than per-panel
// so the voice picker, the language picker and the upload form cannot end up
// offering three different vocabularies for the same thing.
export const LANGUAGE_NAMES = {
  zh: "Chinese", "zh-TW": "Chinese (Taiwan)", "zh-CN": "Chinese (China)",
  en: "English", ja: "Japanese", ko: "Korean", de: "German",
  fr: "French", it: "Italian", pt: "Portuguese", ru: "Russian", es: "Spanish",
};

export const languageName = (code) =>
  code ? LANGUAGE_NAMES[code] || code : "Any language";

/** `<option>`s for language codes, in the order given. */
export const languageOptions = (codes) =>
  codes
    .map((c) => `<option value="${esc(c)}">${esc(languageName(c))}</option>`)
    .join("");

/** The one way a voice is written into a picker, whichever panel asks. */
export const voiceOption = (v, withLanguage = true) =>
  `<option value="${esc(v.id)}">${esc(v.name)}` +
  `${withLanguage && v.language ? ` · ${esc(v.language)}` : ""}</option>`;

/**
 * Voices as language groups, in the order given.
 *
 * MOSS-TTS-Nano offers twenty-six in one list across three languages and two
 * kinds; grouping is what makes that a thing to pick from rather than scroll.
 * Language stays a control of its own — these groups say what a voice reads
 * by default, not what it may be asked to read.
 *
 * Voices that declare no language lead: they are not a leftover but the
 * language-agnostic kind, OmniVoice's designed voices, which read whatever
 * the text is.
 */
export function voiceGroups(voices, order, label) {
  const byLanguage = new Map();
  for (const voice of voices) {
    const key = voice.language || "";
    if (!byLanguage.has(key)) byLanguage.set(key, []);
    byLanguage.get(key).push(voice);
  }
  const keys = [
    ...(byLanguage.has("") ? [""] : []),
    ...order.filter((code) => byLanguage.has(code)),
    // Anything the catalog did not list still has to appear.
    ...[...byLanguage.keys()].filter((k) => k && !order.includes(k)),
  ];
  // One group is no grouping: a flat list reads better than a lone heading —
  // and with one language left there is nothing for the suffix to tell apart.
  if (keys.length < 2) return voices.map((v) => voiceOption(v, false)).join("");
  return keys
    .map(
      (key) =>
        `<optgroup label="${esc(label(key))}">` +
        // No ` · zh` inside a group headed "Chinese".
        byLanguage.get(key).map((v) => voiceOption(v, false)).join("") +
        "</optgroup>",
    )
    .join("");
}

/**
 * Put a message in a slot. Every failure in this UI lands in one of these,
 * next to the control that failed — never in a dialog the panel cannot style
 * and the reader cannot copy from.
 *
 * `kind` is "err", "ok" or "warn"; an empty text clears the slot.
 */
export function msg(el, text, kind = "") {
  const base = el.dataset.base ?? "msg";
  el.className = kind ? `${base} ${kind}` : base;
  el.textContent = text;
  // A slot with its own base class reserves space in its layout, so it stays
  // put when it has nothing to say.
  el.hidden = !text && base === "msg";
}

/**
 * The two-step delete, for the same reason a message is never a dialog: the
 * first click arms the button, a second within a few seconds confirms, and
 * nothing happens if it never comes. `armed` is the caller's set of armed
 * ids and `render` redraws from it, so a re-render in between loses nothing.
 */
export function confirmStep(armed, id, render, ms = 4000) {
  if (armed.delete(id)) return true;
  armed.add(id);
  render();
  setTimeout(() => { if (armed.delete(id)) render(); }, ms);
  return false;
}

/**
 * Replace a picker's options, keeping the selection: a re-render with
 * identical options must not reset the choice under whoever is using it.
 *
 * Precedence: what a person chose, then `preferred` if the list offers it,
 * then the first entry — which is the browser's own and needs no code.
 */
export function fillPicker(select, options, has, preferred) {
  if (select.innerHTML === options) return;
  const chosen = select.value;
  select.innerHTML = options;
  for (const candidate of [chosen, preferred]) {
    if (candidate && has(candidate)) {
      select.value = candidate;
      return;
    }
  }
}
