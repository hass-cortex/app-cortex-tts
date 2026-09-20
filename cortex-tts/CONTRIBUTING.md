# Contributing to Cortex TTS

Thank you for your interest in contributing!

## Development Setup

### Prerequisites

- Python 3.12 or 3.13 (see `pyproject.toml`)
- [uv](https://docs.astral.sh/uv/) — this project does not use pip

### Build and run

```bash
cd cortex-tts
uv sync --frozen

# Run it against a scratch data directory, with auth off.
API_KEY= PORT=8799 \
  DATA_DIR=/tmp/cortex-data STATIC_DIR="$PWD/web" \
  uv run python -m cortex_tts
```

The environment carries only what must be settled before the process starts
(`HOST`, `PORT`, `DATA_DIR`, `STATIC_DIR`, `API_KEY`, `LOG_LEVEL`, and
`REFERENCES_DIR`, which defaults to `<DATA_DIR>/references`). Everything else
is a stored setting in `<DATA_DIR>/settings.json`; `preload` defaults to
`true`, so a fresh data directory downloads the 40M bundle on first start. To
work without weights, write `{"preload": false}` into that file before
starting (or turn it off in the UI once).

The admin UI has **no build step**: `web/` is plain ES modules and CSS served
by `StaticFiles`; edit a file and reload the page.

### Code quality

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run --with pyright pyright
uv run pytest -q
```

All four run in CI and at the release gate. `pre-commit install` wires the
first two plus yamllint, hadolint, shellcheck and prettier; pyright and pytest
run at pre-push.

## Commit Convention

[Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`,
`docs:`, `chore:`, `refactor:`, `test:` — as in
`fix: expand a number that ends a sentence`. The reasoning behind a change —
what was measured, what was tried, why the other option was rejected — belongs
in the commit message; a comment in the source explains only what the code
cannot say about itself (a constraint, an ordering requirement, a unit,
something that looks wrong but is deliberate). Never write the same
explanation in both places; only the copy in the source will rot.

## Pull Request Process

1. Fork and branch from `main`.
2. Make the change.
3. Ensure the four gates under **Code quality** pass.
4. Write the PR description against the template — **Issue**, **Approach**,
   **Verification**, **Not verified**, **Blast radius**: link the issue rather
   than restating it, say which approach you took and what you rejected, say
   how you verified it, and say plainly what you did **not** verify.
5. Request review.

## Testing

`uv run pytest -q` (one file: `uv run pytest -q tests/test_english.py`; one
area: `uv run pytest -q -k welded -v`) needs no model weights, in CI or
locally — the text path, `/api/preview` and the whole suite run on an empty
data directory. A change to the text path deserves a test, because its failure
mode is silent: a wrong reading produces confident, fluent audio rather than
an error, and the honest check for a changed reading is to diff the two
outputs over a corpus and confirm every difference is intended. Synthesis
itself is exercised by hand against real bundles; no fixture runs the ONNX
sessions.

## Architecture

[`AGENTS.md`](../AGENTS.md#architecture) has the module tree and the rules the
code keeps; [`CONTEXT.md`](CONTEXT.md) defines the vocabulary both use.

## Adding a Model

[`AGENTS.md`](../AGENTS.md#how-to) › **Add an engine**.

## Questions?

Open an [issue](https://github.com/hass-cortex/app-cortex-tts/issues/new/choose). The
templates cover a bug report and a feature request; anything else is welcome as
a blank issue.
