// The chart: one reply record in, the panel's HTML out.
//
// Nothing here reaches for a socket, an audio clock or a button — every figure
// it draws is already in the record it is handed, which is what lets the panel
// be read (and this file be exercised) without a reply in flight. What moves
// those figures is `live.js`; what they look like is here.

import { esc } from "./dom.js";

/** The right edge of the chart: what has happened, or is happening. */
export function span(run, batches, clock) {
  const ends = [clock, ...batches.map((b) => b.renderedAt ?? clock),
    run.bufferedTo ?? 0,
    run.released === null ? 0
      : run.released + batches.reduce((n, b) => n + (b.audioS || 0), 0)];
  return Math.max(1, ...ends) * 1.02;
}
export const pct = (value, total) => `${(100 * value / total).toFixed(3)}%`;
/**
 * One bar on the shared axis, with its label wherever it fits.
 *
 * A request is a tenth of a long reply, so a label set inside the bar is
 * clipped to a few characters and the figure it carries is the one thing the
 * row is for. Narrow bars hand it to the space on their right.
 */
function bar(from, to, total, cls, label = "") {
  const width = Math.max(to - from, 0);
  // A bar too narrow to hold its own label hands it to the track beside it,
  // and which side is decided by where the bar sits rather than by a width in
  // pixels: the label's length in characters is known here, the track's width
  // is not — it follows the window, and behind ingress the panel is narrower
  // still. Whichever half the bar is in, the other half is the empty one, and
  // half a track is more room than any label needs.
  const outside = width / total < 0.13;
  const flip = outside && (from + width / 2) / total > 0.5;
  const side = outside ? (flip ? " out-left" : " out") : "";
  return `<span class="tl-bar ${cls}${side}" ` +
    `style="left:${pct(from, total)};width:${pct(width, total)}">` +
    `<i>${esc(label)}</i></span>`;
}
/**
 * The listener's own bar: held back, then played, then in hand and not yet
 * played.
 *
 * One bar rather than two lanes, read the way a video player's is. On a wall
 * clock the width of that last piece *is* the lead — the seconds of audio
 * between the listener and silence — so where it narrows to nothing is where
 * playback caught the renderer. Two lanes can show when each thing happened
 * and never show the distance between them, which is the quantity the
 * pacing verdict is about.
 */
function listenerRow(run, batches, total) {
  const audio = run.audio;
  const played = audio.ctx && !audio.stalled;
  // With no audio clock there is no playback to report, only arrivals. Said
  // plainly rather than left to read as a playback stuck at zero.
  const arrived = batches.reduce((n, b) => n + (b.audioS || 0), 0);
  const head = played ? (run.headAt ?? run.released) : run.released;
  const buffered = played
    ? (run.bufferedTo ?? run.released)
    : run.released + arrived;
  const lead = Math.max(0, buffered - head);
  const stopped = run.error ? " — the reply stopped here" : "";
  const caption = played
    ? `held back ${run.released.toFixed(2)}s, then played `
      + `${(run.heardS || 0).toFixed(1)}s of ${audio.scheduledS.toFixed(1)}s — `
      + `${lead.toFixed(1)}s in hand`
      + (run.gaps.length ? `, ${run.gaps.length} gap(s)` : "")
      + stopped
    : `held back ${run.released.toFixed(2)}s, then ${arrived.toFixed(1)}s of `
      + "audio arrived — no audio clock, nothing was played";
  // A gap too short to be noticed is counted but not drawn.
  const gaps = run.gaps.map((g) => {
    const cls = gapClass(g.to - g.from);
    return cls ? bar(g.from, g.to, total, cls, `${(g.to - g.from).toFixed(2)}s`) : "";
  });
  return `
      <div class="tl-item heard">
        <div class="tl-said"><b>▶</b><span>${esc(caption)}</span></div>
        <div class="tl-track">
          ${bar(0, run.released, total, "wait")}
          ${played ? bar(run.released, head, total, "play") : ""}
          ${bar(head, buffered, total, "buffer",
            lead > 0.5 ? `${lead.toFixed(1)}s` : "")}
          ${gaps.join("")}
        </div>
      </div>`;
}
/**
 * A gap is classed by its size alone: under 0.3 s passes unnoticed, up to a
 * second is heard as a pause, longer is the reply run dry.
 */
function gapClass(lengthS) {
  if (lengthS > 1) return "dry";
  if (lengthS >= 0.3) return "warn";
  return "";
}
export function axis(total) {
  // A tick every 1, 2, 5, 10… seconds, whichever gives about six of them.
  const raw = total / 6;
  const power = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * power).find((s) => s >= raw) || power * 10;
  const ticks = [];
  for (let at = 0; at <= total; at += step) {
    ticks.push(`<span style="left:${pct(at, total)}">${at.toFixed(step < 1 ? 1 : 0)}s</span>`);
  }
  return ticks.join("");
}
export function summary(run) {
  const done = run.done;
  const rows = [];
  if (run.ready) {
    rows.push(["model", `${run.ready.model} · ${run.ready.voice}`]);
    // What the verdict rests on: the voice's median RTF on this host, or
    // nothing yet, in which case `auto` streams on trust.
    const { rtf, samples } = run.ready;
    rows.push(["measured", rtf === null || rtf === undefined
      ? "not yet" : `RTF ${rtf.toFixed(2)} over ${samples}`]);
  }
  // `ready` carries the verdict; `done` says what happened. The last one there is.
  const asked = run.asked;
  const actual = done ? done.mode : (run.ready ? run.ready.mode : "…");
  // Under `auto` the verdict is the answer; an insistence is shown beside
  // the outcome when the two differ rather than silently diverging.
  rows.push(["mode", asked === "auto" || asked === actual
    ? actual : `${actual} (asked ${asked})`]);
  if (run.released !== null) {
    rows.push(["first word", `${run.released.toFixed(2)}s`]);
    const whole = run.audio.ctx && !run.audio.stalled ? run.audio.scheduledS : 0;
    if (whole) {
      rows.push(["played", `${(run.heardS || 0).toFixed(1)}s / ${whole.toFixed(1)}s`]);
    }
  }
  // Only claimed where there was an audio clock to hear them on.
  if (run.audio.ctx && !run.audio.stalled) {
    if (run.gaps.length) {
      const worst = Math.max(...run.gaps.map((g) => g.to - g.from));
      rows.push(["gaps heard", `${run.gaps.length}, worst ${worst.toFixed(2)}s`]);
    } else if (run.done) {
      rows.push(["gaps heard", "none"]);
    }
  }
  if (done) {
    rows.push(["requests", String(done.batches)]);
    rows.push(["audio", `${done.audio_seconds.toFixed(2)}s`]);
    rows.push(["render", `${(done.render_ms / 1000).toFixed(2)}s`]);
    // What the first audio waited between being rendered and being released.
    if (done.bank_wait_ms !== null && done.bank_wait_ms !== undefined) {
      rows.push(["bank wait", `${Math.round(done.bank_wait_ms)}ms`]);
    }
    if (done.min_lead_s !== null && done.min_lead_s !== undefined) {
      const at = done.gap_at !== null && done.gap_at !== undefined
        ? ` at request ${done.gap_at}` : "";
      rows.push(["min lead",
        `${done.min_lead_s > 0 ? "+" : ""}${done.min_lead_s.toFixed(2)}s${at}`]);
    }
  }
  return rows.map(([label, value]) =>
    `<span>${esc(label)} ${esc(value)}</span>`).join("");
}

/** Every row of the chart, in reading order; `""` when there is nothing yet. */
export function rows(run, clock, total) {
  const batches = run.batches;
  const rows = [];

  // Making the model resident is paid before anything is rendered, and on a
  // cold start it is most of the wait. Left unsaid it reads as dead air.
  if (run.readyAt !== null && run.readyAt > 0.3) {
    rows.push(`
      <div class="tl-item">
        <div class="tl-said"><b>·</b><span>making the model resident</span></div>
        <div class="tl-track">${
          bar(0, run.readyAt, total, "load", `${run.readyAt.toFixed(2)}s`)}</div>
      </div>`);
  }

  for (const batch of batches) {
    const to = batch.renderedAt ?? clock ?? batch.sentAt;
    // A request with no cost and no reply left is one that did not finish.
    const failed = batch.renderedAt === null && run.error !== null;
    // The bar's width is the render, so the audio beside it is a second
    // quantity on an axis it does not belong to — which reads as a range.
    // The factor is what the pair was being read for anyway, and it says
    // which way round they go.
    const renderS = batch.renderMs / 1000;
    const factor = batch.audioS > 0
      ? ` (${(renderS / batch.audioS).toFixed(2)}×)`
      : "";
    const cost = batch.renderMs
      ? `${renderS.toFixed(2)}s → ${batch.audioS.toFixed(2)}s${factor}`
      : failed ? "did not finish" : "rendering…";
    rows.push(`
      <div class="tl-item">
        <div class="tl-said">
          <b>${batch.index}</b>
          <span>${esc(batch.text.replace(/\s+/g, " ").trim())}</span>
        </div>
        <div class="tl-track">${
          bar(batch.sentAt, to, total, failed ? "dry" : "render", cost)}</div>
      </div>`);
  }

  if (run.released !== null) rows.push(listenerRow(run, batches, total));
  return rows.join("");
}
