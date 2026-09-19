// The live panel: speak a reply over the socket, play it as it arrives, and
// draw when everything happened.
//
// Every figure on the chart is observed here — from frames, from byte
// arrivals, and from what the audio clock actually did. Nothing is a
// prediction the server made. That is the point: the planner's arithmetic is
// already written down, and what this answers is whether the delivery matched
// it, which a chart drawn from the same arithmetic could never say.
//
// The reply plays here while it arrives, and the playback is scheduled
// sample-accurately rather than handed to an `<audio>` element, because the
// thing worth hearing is the timing: the opening silence, the pause between
// two sentences, the moment it runs dry. An element decides those for itself
// — it buffers, it stalls, it catches up — and a media-source pipeline would
// still put a decoder's own latency between the socket and the speaker. Raw
// PCM on the Web Audio clock does not: a buffer is heard exactly when it is
// scheduled, and a buffer that arrives after its slot is a gap the listener
// hears and this file can measure.

import { api, storedKey } from "./api.js";
import { $, esc, msg, show } from "./dom.js";
import { delivery, hasVoice, refreshModels, setSpeaking } from "./models.js";
import { play } from "./player.js";
import { axis, pct, rows, span, summary } from "./timeline.js";
import { switches } from "./preview.js";

let socket = null;
let run = null;
let ticker = null;

const now = () => performance.now() / 1000;

/** Speaking is offered when the chosen model has a voice. */
export function syncButton() {
  if (!socket) $("live").disabled = !hasVoice();
}

/**
 * Audio still on the clock: the reply is over but the listener is not.
 *
 * A reply released early finishes rendering long before it finishes playing —
 * the panel says so itself, and keeps redrawing for it — so the socket closing
 * is not the end of anything the listener can hear.
 */
function stillPlaying() {
  const audio = run && run.audio;
  return Boolean(
    audio
      && audio.ctx
      && !audio.stalled
      && audio.offset !== null
      && audio.ctx.currentTime < audio.playAt,
  );
}

/** Stop while there is anything to stop, which is not the same as connected. */
function syncStopButton() {
  const busy = Boolean(socket) || stillPlaying();
  const button = $("live");
  button.textContent = busy ? "Stop" : "Speak";
  button.classList.toggle("danger", busy);
  button.classList.toggle("primary", !busy);
  if (!busy) syncButton();
}

// -- the run ----------------------------------------------------------------

function start() {
  stopAudio();
  // The table below is about to go stale: this reply evicts whatever is
  // resident and loads what it asked for, and neither shows up in a page that
  // only looks again when the reply is over.
  setSpeaking(true);
  const t0 = now();
  const key = storedKey();
  const target = new URL(api("/speak/live"));
  target.protocol = target.protocol === "https:" ? "wss:" : "ws:";
  // A WebSocket constructor cannot set a header, so behind a published port
  // the key travels as a subprotocol. Behind ingress there is none to send
  // and the Supervisor has already said who this is.
  socket = key
    ? new WebSocket(target.href, ["cortex-tts", key])
    : new WebSocket(target.href);
  socket.binaryType = "arraybuffer";

  run = {
    t0,
    // What was asked for, read once: the picker is free to change while the
    // reply is still running, and the summary is about the reply.
    asked: $("liveMode").value,
    total: 1,
    heardS: 0,
    endedAt: null,
    // Where playback is, and how far the audio in hand reaches — both on the
    // wall clock, so the distance between them is the lead in seconds.
    headAt: null,
    bufferedTo: null,
    ready: null,
    readyAt: null,
    batches: [],
    pcm: [],
    released: null,
    done: null,
    error: null,
    // The silences the audio clock actually left; where playback is comes
    // from the clock itself rather than from a list of what was scheduled.
    gaps: [],
    audio: newAudio(t0),
  };
  // Left enabled on purpose: while a reply is in flight this button is Stop.
  syncStopButton();
  msg($("liveMsg"), run.audio.ctx ? "" : "no audio clock here — showing the times only", "warn");
  show($("timeline"));
  draw();
  ticker = setInterval(draw, 100);

  socket.addEventListener("open", () => {
    socket.send(JSON.stringify({
      type: "start",
      model: $("model").value,
      voice: $("voice").value,
      // Raw PCM, because this panel schedules samples itself. MP3 would put
      // a decode step between the socket and the speaker, and the whole
      // question here is where the time went.
      format: "wav",
      // What this run should be, not what the server would choose: the panel
      // exists to compare them on one reply.
      mode: run.asked,
      ...switches(),
      ...delivery(),
    }));
    socket.send(JSON.stringify({ type: "text", text: $("text").value }));
    socket.send(JSON.stringify({ type: "end" }));
  });

  socket.addEventListener("message", (event) => {
    if (typeof event.data !== "string") return receiveAudio(event.data);
    receiveFrame(JSON.parse(event.data));
  });

  socket.addEventListener("error", () => {
    // A browser reports a refused handshake as a bare error with no reason.
    if (!run.done && !run.error) run.error = "the socket would not open";
  });
  socket.addEventListener("close", finish);
}

function receiveFrame(frame) {
  const at = now() - run.t0;
  if (frame.type === "ready") {
    run.ready = frame;
    run.readyAt = at;
  } else if (frame.type === "batch") {
    run.batches.push({
      index: frame.index,
      mode: frame.mode,
      text: frame.text || "",
      endsSentence: !!frame.ends_sentence,
      sentAt: at,
      renderedAt: null,
      audioS: null,
    });
  } else if (frame.type === "rendered") {
    const batch = run.batches.find((b) => b.index === frame.index);
    if (batch) {
      batch.renderedAt = at;
      batch.audioS = frame.audio_s;
      batch.renderMs = frame.render_ms;
    }
  } else if (frame.type === "done") {
    run.done = frame;
  } else if (frame.type === "error") {
    run.error = frame.message || "the server refused the reply";
    // Where the reply stopped. Without it the request still in flight goes on
    // drawing itself as rendering, and its bar grows for as long as the page
    // is left open — measured on a render that ran the card out of memory.
    run.endedAt = at;
  }
}

function finish() {
  clearInterval(ticker);
  ticker = null;
  socket = null;
  // Not necessarily back to Speak: the render is what ended, and audio
  // released early is still being heard.
  syncStopButton();
  if (run.error) msg($("liveMsg"), run.error, "err");
  // The page's own player, for hearing it again without the timing — loaded
  // paused, because this panel has already played it once and starting it
  // here would play the reply over itself.
  if (run.pcm.length) {
    play(URL.createObjectURL(wavBlob()), { revokable: true, autoplay: false });
  }
  draw();
  // Playback outlasts the render on a reply that was released early, so the
  // panel keeps redrawing until the last sample has been heard.
  if (run.audio.ctx && !run.audio.stalled && run.audio.offset !== null) {
    ticker = setInterval(draw, 100);
    requestAnimationFrame(() => movePlayhead(true));
  }
  // A reply loads and evicts models, and it has just added a measurement to
  // the model's own line. One last look, then stop looking.
  setSpeaking(false);
  refreshModels().catch(() => { /* the table says its own errors */ });
}

// -- hearing it -------------------------------------------------------------

function newAudio(t0) {
  if (!window.AudioContext) return { ctx: null };
  // Built inside the click that started the run: a context created anywhere
  // else starts suspended and the first buffer is heard whenever the browser
  // decides, which is the one thing this panel must not let it decide.
  const ctx = new AudioContext();
  ctx.resume().catch(() => { /* already running */ });
  return {
    ctx,
    sources: [],
    // Where the next buffer goes on the audio clock; 0 until the first one.
    playAt: 0,
    scheduledS: 0,
    headerLeft: 44,
    odd: null,
    // What the audio clock reads when the reply clock reads zero. Fixed at
    // the first buffer rather than here: a context the browser has not let
    // start yet leaves `currentTime` at zero however long it is held, and
    // anchoring on that put the first word of a five-second wait at 0.00 s.
    offset: null,
  };
}

/**
 * Where playback is.
 *
 * Called both from the frame clock, for a line that moves smoothly, and from
 * the redraw, because a frame callback does not fire in a background tab and
 * the figure it keeps is the one thing here that must not quietly stop.
 */
function movePlayhead(fromFrame = false) {
  const audio = run && run.audio;
  const head = $("tlHead");
  if (!audio || !audio.ctx || audio.stalled || audio.offset === null
      || !run.total || run.released === null) {
    head.hidden = true;
    return;
  }
  // Where the audio scheduled so far runs out; gaps push it later than the
  // audio is long, which is exactly why one is not the other.
  const ends = audio.playAt + audio.offset;
  const at = Math.min(audio.ctx.currentTime + audio.offset, ends);
  run.headAt = at;
  run.bufferedTo = ends;
  head.hidden = false;
  head.style.left = pct(at, run.total);
  // Audio heard, not time passed: a gap moves the head and plays nothing.
  run.heardS = Math.max(0, audio.scheduledS - (ends - at));
  // A host with no audio device reports a running context whose clock never
  // advances — measured in headless Chrome, `state: "running"` and
  // `currentTime` still 0 a second later. Left alone the panel counts a
  // playback that is not happening and never stops redrawing, so the moment
  // it is clear nothing is moving the claim is withdrawn.
  if (audio.scheduledS > 1 && at <= run.released + 0.05
      && (now() - run.t0) - run.released > 2) {
    audio.stalled = true;
    head.hidden = true;
    msg($("liveMsg"),
      "this browser has no audio clock, so nothing was played — the times "
      + "below are when the audio arrived", "warn");
  }
  // Only once there is nothing left to draw: a stalled clock stops the
  // playhead, not the reply, whose later requests still have to reach the
  // chart. One more pass first, so what was just decided reaches the rows.
  if (run.done && (audio.stalled || at >= ends) && ticker) {
    setTimeout(() => { clearInterval(ticker); ticker = null; }, 150);
  }
  if (fromFrame && at < ends) requestAnimationFrame(() => movePlayhead(true));
}

function stopAudio() {
  if (!run || !run.audio || !run.audio.ctx) return;
  for (const source of run.audio.sources) {
    try { source.stop(); } catch { /* already finished */ }
  }
  run.audio.ctx.close().catch(() => { /* already closed */ });
  run.audio.ctx = null;
}

function receiveAudio(buffer) {
  const bytes = new Uint8Array(buffer);
  run.pcm.push(bytes);
  const audio = run.audio;
  if (!audio.ctx) {
    if (run.released === null) run.released = now() - run.t0;
    return;
  }
  // A WAV stream opens with a 44-byte header, which may be cut anywhere.
  let from = 0;
  if (audio.headerLeft > 0) {
    from = Math.min(audio.headerLeft, bytes.length);
    audio.headerLeft -= from;
    if (from === bytes.length) return;
  }
  schedule(samples(audio, bytes.subarray(from)));
}

/** 16-bit little-endian PCM as floats, carrying any half sample over. */
function samples(audio, bytes) {
  let data = bytes;
  if (audio.odd !== null) {
    data = new Uint8Array(bytes.length + 1);
    data[0] = audio.odd;
    data.set(bytes, 1);
    audio.odd = null;
  }
  if (data.length % 2) {
    audio.odd = data[data.length - 1];
    data = data.subarray(0, data.length - 1);
  }
  const view = new DataView(data.buffer, data.byteOffset, data.length);
  const out = new Float32Array(data.length / 2);
  for (let i = 0; i < out.length; i++) out[i] = view.getInt16(i * 2, true) / 32768;
  return out;
}

/**
 * Put these samples on the audio clock, immediately after what is already on
 * it — or immediately, when what is already on it has finished playing.
 *
 * The second case is the reply running dry. The silence it leaves is what a
 * listener hears, so it is recorded rather than smoothed over: catching up by
 * dropping audio would hide exactly the fault this panel exists to show.
 */
function schedule(pcm) {
  const audio = run.audio;
  if (!pcm.length) return;
  const rate = (run.ready && run.ready.sample_rate) || audio.ctx.sampleRate;
  const buffer = audio.ctx.createBuffer(1, pcm.length, rate);
  buffer.copyToChannel(pcm, 0);
  const source = audio.ctx.createBufferSource();
  source.buffer = buffer;
  source.connect(audio.ctx.destination);

  const clock = audio.ctx.currentTime;
  if (audio.offset === null) {
    audio.offset = (now() - run.t0) - clock;
    if (audio.ctx.state !== "running") {
      msg($("liveMsg"),
        "the browser is holding the audio back — click the page to let it play",
        "warn");
    }
  }
  if (audio.playAt < clock) {
    if (audio.playAt > 0) {
      run.gaps.push({
        from: audio.playAt + audio.offset,
        to: clock + audio.offset,
        afterS: audio.scheduledS,
      });
    }
    audio.playAt = clock;
  }
  source.start(audio.playAt);
  audio.sources.push(source);
  const at = audio.playAt + audio.offset;
  if (run.released === null) {
    run.released = at;
    requestAnimationFrame(() => movePlayhead(true));
  }
  audio.playAt += buffer.duration;
  audio.scheduledS += buffer.duration;
}

/** Everything heard so far, with a header that says how long it really is. */
function wavBlob() {
  const rate = (run.ready && run.ready.sample_rate) || 24000;
  const pcm = run.pcm.reduce((n, part) => n + part.length, 0) - 44;
  const header = new DataView(new ArrayBuffer(44));
  const ascii = (at, text) => {
    for (let i = 0; i < text.length; i++) header.setUint8(at + i, text.charCodeAt(i));
  };
  ascii(0, "RIFF"); header.setUint32(4, 36 + pcm, true); ascii(8, "WAVEfmt ");
  header.setUint32(16, 16, true); header.setUint16(20, 1, true);
  header.setUint16(22, 1, true); header.setUint32(24, rate, true);
  header.setUint32(28, rate * 2, true); header.setUint16(32, 2, true);
  header.setUint16(34, 16, true); ascii(36, "data");
  header.setUint32(40, pcm, true);
  // The stream's own header declared a length nobody knew yet; drop it.
  const body = run.pcm.slice();
  body[0] = body[0].subarray(44);
  return new Blob([header.buffer, ...body], { type: "audio/wav" });
}

// -- the chart --------------------------------------------------------------




function draw() {
  // First, so the rows below are built from where playback actually is. It
  // also decides whether there is any playback to report, and it stops the
  // redraw when there is not — a flag raised after the rows were built would
  // never reach a row, because the redraw that would carry it is the one
  // being cancelled.
  movePlayhead();
  const clock = run.done ? null : (run.endedAt ?? now() - run.t0);
  const total = span(run, run.batches, clock ?? 0);
  run.total = total;
  // Nothing to draw yet, and which of the two reasons matters: the socket
  // opens in milliseconds, while the server resolves the model and makes it
  // resident before it sends `ready` — 15 s on a cold start here. The load bar
  // exists to say where that went, but it cannot be drawn until `readyAt`
  // arrives with the frame that ends the wait — so while it is being paid
  // there is nothing to draw, and the wait has to be named rather than left
  // to read as a connection that will not open.
  $("tlRows").innerHTML = rows(run, clock, total) || `<p class="hint">${
    socket && socket.readyState === WebSocket.OPEN
      ? "Connected — the server is getting the model ready. A model that is "
        + "not resident is loaded first, which on a cold start is most of the wait."
      : "Connecting…"}</p>`;
  $("tlAxis").innerHTML = axis(total);
  $("tlStats").innerHTML = summary(run);
  // Playback ends without anything being sent or received, so the redraw is
  // the only thing that notices. Asked here rather than at one end-of-audio
  // branch, because an errored reply has no such branch and still leaves
  // audio on the clock.
  syncStopButton();
}





// -- wiring -----------------------------------------------------------------

/**
 * Stop a run in flight. `cancel` says the listener went on purpose, which the
 * server answers by stopping the render at the engine's next checkpoint;
 * closing alone would do it but reads as a connection that dropped. Playback
 * already scheduled is silenced here, because the sources are queued ahead of
 * the clock and would otherwise play on after the render has gone.
 */
function cancel() {
  // Two things can be stopped and they end at different times: the render,
  // which the server is told about, and the playback, which is this page's
  // alone. Past the socket's close only the second is left.
  if (socket && socket.readyState === WebSocket.OPEN) {
    try {
      socket.send(JSON.stringify({ type: "cancel" }));
    } catch { /* going anyway */ }
  }
  stopAudio();
  msg($("liveMsg"), "stopped", "warn");
  if (socket) {
    // `close` is wired to `finish`, which tidies the ticker and the button.
    socket.close();
    return;
  }
  clearInterval(ticker);
  ticker = null;
  syncStopButton();
  draw();
}

export function init() {
  $("live").addEventListener("click", () => {
    if (socket || stillPlaying()) cancel();
    else start();
  });
}
