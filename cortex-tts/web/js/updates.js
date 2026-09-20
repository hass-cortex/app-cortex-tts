// One socket that says which read went stale, so nothing here polls.
//
// The server sends `{type}` for models, voices, references or settings; the
// page re-reads that endpoint. While the socket is down — the app restarting,
// a proxy dropping it — the page falls back to one refresh whenever it comes
// back into view, and reconnects with a growing wait.

import { api, storedKey } from "./api.js";
import * as clones from "./clones.js";
import * as models from "./models.js";
import * as settings from "./settings.js";

const REFRESH = {
  models: () => models.refreshModels(),
  voices: () => models.refreshVoices(),
  references: () => clones.refresh(),
  settings: () => settings.load(),
};

let socket = null;
let wait = 1000;
let connected = false;

/** Re-read every section; each one reports its own errors. */
const refreshAll = () => Object.values(REFRESH).forEach((fn) => fn().catch(() => {}));

function connect() {
  const key = storedKey();
  const target = new URL(api("/events"));
  target.protocol = target.protocol === "https:" ? "wss:" : "ws:";
  socket = key ? new WebSocket(target.href, ["cortex-tts", key]) : new WebSocket(target.href);
  socket.onopen = () => {
    wait = 1000;
    if (!connected) {
      connected = true;
      refreshAll(); // whatever changed while the socket was down
    }
  };
  socket.onmessage = (event) => {
    const { type } = JSON.parse(event.data);
    const refresh = REFRESH[type];
    if (refresh) refresh().catch(() => { /* the section says its own errors */ });
  };
  socket.onclose = () => {
    connected = false;
    socket = null;
    setTimeout(connect, wait);
    wait = Math.min(wait * 2, 30000);
  };
  socket.onerror = () => socket && socket.close();
}

/** Open the socket now, and again after the key changes. */
export function init() {
  if (socket) socket.close();
  connect();
  // A key just entered: the socket that was refused, or never opened, is
  // replaced by one that carries it.
  document.getElementById("keyForm")?.addEventListener("submit", () => {
    wait = 100;
    if (socket) socket.close();
    else connect();
  });
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible" && !connected) refreshAll();
  });
}

export const isConnected = () => connected;
