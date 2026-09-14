// The right-hand column: what the model is actually asked to say, and the
// ledger of every rewrite that got it there.

import { call } from "./api.js";
import { $, esc, pressed } from "./dom.js";
import { diffSpans, joinSpans } from "./diff.js";
import { ARROW } from "./icons.js";
import { previewLanguage } from "./models.js";

// Han only: the warning is about Traditional glyphs, which kana never are.
const HAS_HAN = /[㐀-䶿一-鿿]/;

// The switches the server decides a default for, and the chip each one is
// shown as: bare numbers by the model (on for one that cannot say a digit),
// the two rewrites by the language. A language without a rewrite has no
// chip for it: the server lists only the passes the language has.
const DECIDED = { expand_numbers: "num", convert_script: "conv", taiwan_readings: "tw" };

// What the reader has set by hand, per switch: null means "as decided",
// which the server answers in `passes`. Reset when the language changes,
// because a choice made for Chinese says nothing about German.
const UNSET = { expand_numbers: null, convert_script: null, taiwan_readings: null };
let overrides = { ...UNSET };
let lastLanguage = null;
// The server's answer for the current text, so a click can invert it.
let passes = { normalize_text: true };

/** Flip a switch the reader clicked, against what the server said it is. */
export function toggle(name) {
  overrides[name] = !passes[name];
  refresh();
}

/** The switch fields a request carries: explicit where set, absent otherwise. */
export function switches() {
  const out = { normalize_text: pressed($("norm")) };
  for (const name of Object.keys(DECIDED)) {
    if (overrides[name] !== null) out[name] = overrides[name];
  }
  return out;
}

// One sentence per pass, so the hint claims exactly what ran: saying numerals
// are rewritten while the number pass is off is the lie the panel exists to stop.
function hint(active) {
  const parts = [];
  parts.push(
    active.normalize_text
      ? "Units, times and dates are written out in words."
      : "Numbers stay as digits.",
  );
  if (active.normalize_text) {
    parts.push(
      active.expand_numbers
        ? "A bare number is read as a quantity too."
        : "A bare number stays as digits: it may be a room, a phone number or a model.",
    );
  }
  if ("convert_script" in active) {
    parts.push(
      active.convert_script
        ? "Traditional glyphs are converted to Simplified."
        : "Traditional glyphs are left as written; the model will misread them.",
    );
  }
  if (active.taiwan_readings) {
    parts.push("Words Taiwan reads differently are respelled with homophones.");
  }
  return parts.join(" ");
}

function syncChips(active) {
  for (const [name, id] of Object.entries(DECIDED)) {
    const has = name in active;
    $(id).hidden = !has;
    if (has) $(id).setAttribute("aria-pressed", String(active[name]));
  }
  $("preparedHint").textContent = hint(active);
}

// The ledger mirrors the passes, and the pipeline runs them in this order:
// normalise first, then convert what it produced, then respell what that
// produced. Measuring each pass against its own input keeps a number
// expansion and a glyph swap from landing in the same row. The respellings
// are not diffed at all: a stand-in is one glyph for one glyph, so two
// adjacent ones would read as one rewrite. The server lists them instead.

// The appended sentence-final stop is a real change worth showing, but as an
// insertion it has nothing on the left. Name it rather than render "text — → .".
const STOP_ONLY = /^[。．.!?！？；;…\s]+$/;

function kindOf(span) {
  if (!span.from.trim() && STOP_ONLY.test(span.to)) return "stop";
  return /\d/.test(span.from) ? "number" : "text";
}

function renderLedger(parts) {
  const rows = [];
  if (parts.numberPair) {
    // Rows are read, so they are joined back into whole rewrites. The glyph
    // count below is counted, so it is not.
    for (const s of joinSpans(diffSpans(parts.numberPair[0], parts.numberPair[1]))) {
      // `prepare` strips, so trailing whitespace arrives as an empty rewrite.
      if (s.from.trim() || s.to.trim()) {
        rows.push({ kind: kindOf(s), from: s.from, to: s.to, note: "" });
      }
    }
  }
  if (parts.glyphPair) {
    const spans = diffSpans(parts.glyphPair[0], parts.glyphPair[1]);
    const count = spans.reduce((n, s) => n + s.from.trim().length, 0);
    if (count) {
      rows.push({
        kind: "script",
        from: "繁體",
        to: "簡體",
        note: `${count} glyph${count === 1 ? "" : "s"}`,
      });
    }
  }
  for (const r of parts.readings || []) {
    rows.push({ kind: "reading", from: r.word, to: r.standin, note: "" });
  }

  if (!rows.length) {
    // Only dangerous for Chinese; elsewhere it is the normal state, and a
    // warning that fires when nothing is wrong is one people learn to ignore.
    const risky = parts.conversionOff && HAS_HAN.test($("text").value);
    $("ledger").innerHTML = risky
      ? '<div class="msg warn">Script conversion is off — the model will be handed Traditional glyphs it cannot pronounce.</div>'
      : '<div class="cap ledger-empty">Nothing to rewrite</div>';
    return;
  }
  $("ledger").innerHTML = '<div class="cap">Rewrites</div>' + rows.map((r) => `
    <div class="led">
      <span class="pill idle kind">${esc(r.kind)}</span>
      <span class="from">${esc(r.from) || "—"}</span>
      ${ARROW}
      <span class="to">${esc(r.to) || "—"}</span>
      <span class="note-num">${esc(r.note)}</span>
    </div>`).join("");
}

function prepared(fields) {
  return call("/preview", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      text: $("text").value,
      model: $("model").value || null,
      language: previewLanguage(),
      ...fields,
    }),
  }).then((res) => res.json());
}

function fill(text, empty) {
  $("prepared").textContent = text;
  $("prepared").classList.toggle("empty", empty);
}

let sequence = 0;

/** Recompute the column for whatever is in the composer right now. */
export async function refresh() {
  // Bumped before the empty case too, so a reply still in flight cannot
  // refill a column that was just cleared.
  const seq = (sequence += 1);
  if (!$("text").value.trim()) {
    fill("…", true);
    $("ledger").innerHTML = "";
    syncChips(passes);
    return;
  }
  try {
    const full = await prepared(switches());
    if (seq !== sequence) return; // a newer keystroke already won
    if (full.language !== lastLanguage) {
      // The language moved under the reader's choices: drop them, and ask
      // again only if there were any — with none, the answer is this one.
      lastLanguage = full.language;
      const handSet = Object.values(overrides).some((v) => v !== null);
      overrides = { ...UNSET };
      if (handSet) return refresh();
    }
    passes = full.passes;
    syncChips(passes);

    // The diffed rows need the text as each pass saw it: with the number
    // and glyph passes both on, the converter never saw the raw text, so ask
    // for the stage between them; with the respelling on, the stage before it.
    const glyphs = passes.convert_script === true;
    const [between, respelled] = await Promise.all([
      passes.normalize_text && glyphs
        ? prepared({ ...switches(), convert_script: false, taiwan_readings: false })
        : null,
      passes.taiwan_readings ? prepared({ ...switches(), taiwan_readings: false }) : null,
    ]);
    if (seq !== sequence) return;
    const mid = between ? between.prepared : null;
    const converted = respelled ? respelled.prepared : full.prepared;
    fill(full.prepared, false);
    renderLedger({
      numberPair: passes.normalize_text ? [full.original, mid ?? converted] : null,
      glyphPair: glyphs ? [mid ?? full.original, converted] : null,
      readings: full.readings,
      conversionOff: passes.convert_script === false,
    });
  } catch (err) {
    if (seq !== sequence) return;
    // The prepared text is the failure's own subject, so it carries the error
    // rather than a banner somewhere else on the page.
    fill(err.message, true);
    $("ledger").innerHTML = "";
  }
}

// The prepared text is the product, so it is never more than a keystroke
// behind: the panel refreshes itself once typing settles.
let timer = null;

export function schedule() {
  clearTimeout(timer);
  timer = setTimeout(refresh, 350);
}
