# ADR 0007 — A real-time factor belongs to a host, not to a model

Status: accepted. How the figure is used to pace a reply is ADR 0001.

## Context

A real-time factor measured on the project's reference machine read 3x out
on another. A number a reader can find in two places is a number that will
disagree with itself: one MOSS-on-a-GPU measurement was carried by four pages
in three versions, and the page that had drifted furthest used its stale
pair to draw a conclusion its own table contradicted.

## Decision

- **The catalog carries no figure of its own.** `cortex_tts/stats.py` keeps
  what this host measured and a card shows that or says "not measured".
  `scripts/bench_rtf.py` produces documentation — the tables in
  `docs/models.md` — never a figure the app repeats back to someone else's
  machine.
- **One primitive, recorded once per request**: a `RenderSample` of raw
  audio seconds, wall seconds and the execution provider, by every
  transport, with the model made resident before the clock starts. A load
  is not what a request costs, and a sample that folds one in describes a
  moment no later request reproduces.
- **One figure per model and voice**, the median of the newest eight
  same-provider samples, absent until three exist. Voices are kept apart
  because the cost is: a clone runs about twice what a designed voice does
  on the same model, since the reference's codec frames rejoin the prompt on
  every synthesis. Nothing is pooled and nothing is borrowed. A sample from
  another provider is another machine's; a file written without providers
  is not read.
- **Every measured figure lives in `docs/models.md` and only there**; every
  other page points at its tables. The test is whether a re-measurement can
  be applied by editing one file.
- **A benchmark pollutes the store it runs against.** A run of one length
  leaves the median describing that length and nothing else — a clone's
  short sentences cost more than its average — and 24 samples of it evict
  the traffic that was there. `bench_rtf.py` goes through `/v1/audio/speech`
  and records like any other request, so `DELETE /api/models/{id}/stats`
  afterwards on any host whose figures are in use.

## Consequences

- `tests/test_stats.py` pins that the figure is one voice's own, one
  provider's, and absent until three requests exist.
- The admin UI's card and the live reply's verdict read the same number, so
  there is no second series to keep honest.
