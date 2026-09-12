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
  if (res.status === 401 && !retried) {
    // Behind ingress no key is ever needed; on a published port the page
    // asks once and keeps the answer in this browser.
    await askForKey();
    return call(path, options, true);
  }
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body && body.message) detail = body.message;
    } catch { /* non-JSON error body */ }
    throw new Error(detail);
  }
  return res;
}

const KEY_SLOT = "cortex-tts-api-key";

function storedKey() {
  try { return localStorage.getItem(KEY_SLOT) || ""; } catch { return ""; }
}

let pendingKey = null;

/** Show the key form in the header and resolve once a key is entered. */
function askForKey() {
  if (pendingKey) return pendingKey;
  const form = document.getElementById("keyForm");
  const input = document.getElementById("keyInput");
  form.hidden = false;
  input.focus();
  pendingKey = new Promise((resolve) => {
    form.addEventListener("submit", (e) => {
      e.preventDefault();
      try { localStorage.setItem(KEY_SLOT, input.value.trim()); } catch { /* private window */ }
      form.hidden = true;
      pendingKey = null;
      resolve();
    }, { once: true });
  });
  return pendingKey;
}

/** Fetch a JSON collection from the API. */
export const json = async (path) => (await call(path)).json();
