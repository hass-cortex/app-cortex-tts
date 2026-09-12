// The right-hand column: what the model is actually asked to say, and the
// ledger of every rewrite that got it there.

import { call } from "./api.js";
import { $, esc, pressed } from "./dom.js";
import { diffSpans, joinSpans } from "./diff.js";
import { ARROW } from "./icons.js";

// Han only: the warning is about Traditional glyphs, which kana never are.
const HAS_HAN = /[㐀-䶿一-鿿]/;

// One line per combination: claiming numerals are rewritten while the number
// pass is off is the same lie the panel was built to stop telling.
const HINTS = {
  "1,1": "This model cannot pronounce Traditional glyphs or Arabic numerals. "
    + "Everything below is rewritten before a single sample is generated.",
  "0,1": "Traditional glyphs are converted below; numbers stay as digits, which "
    + "this model cannot pronounce.",
  "1,0": "Numbers, units and times are written out in the script of the text; "
    + "nothing else below is rewritten.",
  "0,0": "Both passes are off, so the model is given the text below exactly "
    + "as it is written.",
};

export function syncHint() {
  const key = `${pressed($("norm")) ? 1 : 0},${pressed($("conv")) ? 1 : 0}`;
  $("preparedHint").textContent = HINTS[key];
}

let lastLanguage = null;

/**
 * Re-default the passes to a voice's language.
 *
 * Only the script conversion follows it: number expansion follows the script
 * of the text, so it stays on for every voice. The voice is the only place a
 * language is declared — the model takes none, its prompt being the text plus
 * a speaker slot — so picking the voice is picking the language.
 *
 * Re-defaulting only on a change is what lets a deliberate toggle survive the
 * picker being rebuilt.
 */
export function syncPassesToVoice(language) {
  if (!language || language === lastLanguage) return;
  lastLanguage = language;
  $("conv").setAttribute("aria-pressed", String(language.toLowerCase().startsWith("zh")));
  syncHint();
  // The switches moved, so the column was computed under the old ones.
  refresh();
}

// The ledger mirrors the two passes the toggles control, and the pipeline
// runs them in this order: normalise first, then convert what it produced.
// Measuring each pass against its own input keeps a number expansion and a
// glyph swap from landing in the same row.

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

  if (!rows.length) {
    // Only dangerous for Chinese; on English it is the normal state, and a
    // warning that fires when nothing is wrong is one people learn to ignore.
    const risky = parts.bothOff && HAS_HAN.test($("text").value);
    $("ledger").innerHTML = risky
      ? '<div class="msg warn">Both passes are off — the model will be handed Traditional glyphs it cannot pronounce.</div>'
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

function prepared(normalize, convert) {
  return call("/preview", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      text: $("text").value,
      normalize_text: normalize,
      convert_script: convert,
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
    return;
  }
  const normalize = pressed($("norm"));
  const convert = pressed($("conv"));
  try {
    syncHint();
    // With both passes on, the converter never saw the raw text — ask for the
    // stage between them as well, so each row is measured against its own input.
    const [full, between] = await Promise.all([
      prepared(normalize, convert),
      normalize && convert ? prepared(true, false) : null,
    ]);
    if (seq !== sequence) return; // a newer keystroke already won
    const mid = between ? between.prepared : null;
    fill(full.prepared, false);
    renderLedger({
      numberPair: normalize ? [full.original, mid ?? full.prepared] : null,
      glyphPair: convert ? [mid ?? full.original, full.prepared] : null,
      bothOff: !normalize && !convert,
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
