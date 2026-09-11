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
