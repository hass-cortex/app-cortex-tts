// How this page talks to the addon. Ingress serves it under a generated
// prefix, so every URL is resolved against the <base> the document sets from
// its own location — nothing here may assume it lives at the server root.

/** Resolve a path against the ingress prefix the page was served under. */
export const url = (path) => new URL(path.replace(/^\//, ""), document.baseURI).href;

/** Resolve a path under the JSON API. */
export const api = (path) => url(`api${path}`);

/**
 * Fetch from the API, raising the server's own message on failure.
 *
 * Every caller surfaces what this throws, so the message has to be the one
 * worth reading: the API's `message` field when there is one, the status
 * line when there is not.
 */
export async function call(path, options = {}, retried = false) {
  const headers = new Headers(options.headers || {});
  const key = storedKey();
  if (key) headers.set("X-API-Key", key);
  const res = await fetch(api(path), { ...options, headers });
  if (res.ok) {
    // A call that worked is proof no key is needed, so any prompt on screen
    // is moot — including one still waiting for an answer. Without this the
    // form is shown once and never taken back: only the submit handler hides
    // it, it lives in the header rather than in anything that re-renders, and
    // the call parked behind it resolves only when someone types. Nobody
    // types, because nothing is wrong.
    dismissKeyForm();
    return res;
  }

  // The body is read once, here, because a 401 has to be identified before it
  // can be acted on and the stream cannot be read twice.
  let code = "";
  let detail = `${res.status} ${res.statusText}`;
  try {
    const body = await res.json();
    if (body && body.message) detail = body.message;
    if (body && body.code) code = body.code;
  } catch { /* non-JSON error body */ }

  // Only the app's own 401 means a key is missing. Behind ingress the
  // Supervisor answers 401 of its own when the ingress session has lapsed,
  // and that one carries no `AUTH_REQUIRED` — asking for a key there offers a
  // cure for something the key cannot touch, and the page then waits on an
  // answer that can never work. Reloading is what fixes that one.
  if (res.status === 401 && code === "AUTH_REQUIRED" && !retried) {
    // On a published port the page asks once and keeps the answer in this
    // browser.
    await askForKey();
    return call(path, options, true);
  }
  throw new Error(detail);
}

const KEY_SLOT = "cortex-tts-api-key";

function storedKey() {
  try { return localStorage.getItem(KEY_SLOT) || ""; } catch { return ""; }
}

let pendingKey = null;
let answerKey = null;

/**
 * Take the key form down, releasing anything parked behind it.
 *
 * Safe to call when it is already down. Releasing matters: the parked call
 * retries, and since the retry is what proved a key unnecessary it succeeds.
 */
function dismissKeyForm() {
  const form = document.getElementById("keyForm");
  if (form) form.hidden = true;
  const release = answerKey;
  pendingKey = null;
  answerKey = null;
  if (release) release();
}

/** Show the key form in the header and resolve once a key is entered. */
function askForKey() {
  if (pendingKey) return pendingKey;
  const form = document.getElementById("keyForm");
  const input = document.getElementById("keyInput");
  form.hidden = false;
  input.focus();
  pendingKey = new Promise((resolve) => {
    answerKey = resolve;
    form.addEventListener("submit", (e) => {
      e.preventDefault();
      try { localStorage.setItem(KEY_SLOT, input.value.trim()); } catch { /* private window */ }
      dismissKeyForm();
    }, { once: true });
  });
  return pendingKey;
}

/** Fetch a JSON collection from the API. */
export const json = async (path) => (await call(path)).json();
