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
export async function call(path, options = {}) {
  const res = await fetch(api(path), options);
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

/** Fetch a JSON collection from the API. */
export const json = async (path) => (await call(path)).json();
