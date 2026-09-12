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

/** The one way a voice is written into a picker, whichever panel asks. */
export const voiceOption = (v) =>
  `<option value="${esc(v.id)}">${esc(v.name)}${v.language ? ` · ${esc(v.language)}` : ""}</option>`;

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
