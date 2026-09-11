// The smallest set of (from -> to) spans that turns one string into another.

// The strings are one utterance, so a quadratic LCS is fine once the shared
// head and tail are stripped; beyond the cap the caller gets a single
// whole-text span rather than a frozen tab.
const CELLS = 1_200_000;

// Latin diffs by word; CJK has no word gaps, so each character is its own
// token. A character-level LCS shreds a Latin rewrite into fragments.
const TOKENS = /[A-Za-z0-9]+|[\s\S]/g;
const tokenise = (text) => text.match(TOKENS) ?? [];

export function diffSpans(a, b) {
  const left = tokenise(a);
  const right = tokenise(b);
  let head = 0;
  while (head < left.length && head < right.length && left[head] === right[head]) head += 1;
  let tail = 0;
  while (tail < left.length - head && tail < right.length - head
         && left[left.length - 1 - tail] === right[right.length - 1 - tail]) tail += 1;

  const A = left.slice(head, left.length - tail);
  const B = right.slice(head, right.length - tail);
  if (!A.length && !B.length) return [];
  if (A.length * B.length > CELLS) return [{ from: A.join(""), to: B.join(""), gap: "" }];

  const n = A.length;
  const m = B.length;
  const width = m + 1;
  const lcs = new Int32Array((n + 1) * width);
  for (let i = n - 1; i >= 0; i -= 1) {
    for (let j = m - 1; j >= 0; j -= 1) {
      lcs[i * width + j] = A[i] === B[j]
        ? lcs[(i + 1) * width + j + 1] + 1
        : Math.max(lcs[(i + 1) * width + j], lcs[i * width + j + 1]);
    }
  }

  const spans = [];
  let from = "";
  let to = "";
  // What was left alone since the previous span ended. The caller needs it to
  // tell one rewrite interrupted by a space from two separate rewrites.
  let gap = "";
  let i = 0;
  let j = 0;
  const flush = () => {
    if (from || to) {
      spans.push({ from, to, gap });
      gap = "";
    }
    from = "";
    to = "";
  };
  while (i < n && j < m) {
    if (A[i] === B[j]) { flush(); gap += A[i]; i += 1; j += 1; }
    else if (lcs[(i + 1) * width + j] >= lcs[i * width + j + 1]) { from += A[i]; i += 1; }
    else { to += B[j]; j += 1; }
  }
  from += A.slice(i).join("");
  to += B.slice(j).join("");
  flush();
  return spans;
}

/**
 * Join spans that a trivial unchanged run separated, so one rewrite reads as
 * one row. Not for counting: a joined span reports the kept run as changed.
 */
export function joinSpans(spans, limit = 2) {
  const joined = [];
  for (const span of spans) {
    const last = joined[joined.length - 1];
    if (last && span.gap.length <= limit) {
      last.from += span.gap + span.from;
      last.to += span.gap + span.to;
    } else {
      joined.push({ ...span });
    }
  }
  return joined;
}
