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
(`HOST`, `PORT`, `DATA_DIR`, `STATIC_DIR`, `API_KEY`). Everything else is a
stored setting in `<DATA_DIR>/settings.json`, and one of them matters here:
`preload` defaults to `true`, so a fresh data directory downloads the 241 MB
40M bundle on first start. To work without weights, write
`{"preload": false}` into that file before starting (or turn it off in the
UI once).

The admin UI has **no build step**. `web/` is plain ES modules and CSS served
by `StaticFiles`; edit a file and reload the page.

Nothing needs model weights except synthesis itself. The text path,
`/api/preview` and the whole test suite run on an empty data directory, which
is what makes the interesting part of this app cheap to work on.

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

This project uses [Conventional Commits](https://www.conventionalcommits.org/):

| Prefix      | Use Case                                 |
| ----------- | ---------------------------------------- |
| `feat:`     | New feature                              |
| `fix:`      | Bug fix                                  |
| `docs:`     | Documentation only                       |
| `chore:`    | Maintenance / tooling                    |
| `refactor:` | Code restructure without behavior change |
| `test:`     | Adding or updating tests                 |

Examples:

```
feat: expand ordinals in English text
fix: expand a number that ends a sentence
docs: describe what the two text switches actually do
chore: wire pyright into the release gate
refactor: build both normalisers from one pass table
test: cover the welded-digit readings
```

### What goes in the message, not the code

The reasoning behind a change — what was measured, what was tried, why the
other option was rejected — belongs in the commit message. A comment in the
source explains what the code cannot say about itself: a constraint, an
ordering requirement, a unit, something that looks wrong but is deliberate.
Never write the same explanation in both places; only the copy in the source
will rot.

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

No model files are needed, in CI or locally.

```bash
uv run pytest -q
uv run pytest -q tests/test_english.py
uv run pytest -q -k welded -v
```

A change to the text path deserves a test, because its failure mode is silent:
a wrong reading produces confident, fluent audio rather than an error. If you
are changing a reading, the honest check is to compare the two outputs over a
corpus and confirm that every difference is one you intended — a handful of
assertions will not catch a pass that quietly claimed someone else's digits.

Synthesis itself is exercised by hand against real bundles; there is no fixture
that runs the ONNX sessions.

## Architecture

[`AGENTS.md`](../AGENTS.md) has the module tree and the cross-module
guarantees; [`CONTEXT.md`](CONTEXT.md) defines the vocabulary both use — read
it before naming anything new, because most nouns in this codebase already
mean two things. The user-facing reference pages are under [`docs/`](docs/).

Two packages, one dependency direction — `cortex_tts` imports `cortex_speech`
through its facade and never the reverse; `tests/test_architecture.py` fails
the build otherwise.

## Adding a Model

The catalog is hand-written (`src/cortex_speech/catalog.py`): three models,
each one or more Hugging Face repos plus the list of files in its bundle, and
the capabilities it actually has. A new model is a `ModelSpec` naming an
existing backend, or a new backend registered with `backends.register` — the
steps are in [`AGENTS.md`](../AGENTS.md) under **Add an engine**, and nothing
in the registry, the API layer or the integration needs to change for either.

Mind the cost figures on the entry (`size_mb`, `rtf_hint`, `rss_hint_mb`): the
admin UI and the documentation show them so a reader can compare models with
each other. Nothing decides with `rtf_hint` any more — the integration defaults
every model to buffered and leaves streaming to someone who has measured their
own host — but a figure from another machine still misleads whoever is
choosing. `rtf_hint` is therefore measured, never copied from upstream: run
`scripts/bench_rtf.py` against the reference host (a 4-core Home Assistant OS
VM at two threads) with the new model added to its list, and record the median
it prints, then update the tables in `docs/models.md` and `DOCS.md`.

For a quantised model, name the CPU too. Full-precision weights compute the
same thing anywhere; a dynamically quantised one picks a kernel per host, and
the kernels do not always agree — an INT8 export tried here stopped correctly
on a machine with AVX-512 VNNI and never emitted its stop token on one without,
which is a wrong answer rather than a slow one. So a single machine says less
about a quantised model than about any other.

## Questions?

Open an [issue](https://github.com/hass-cortex/app-cortex-tts/issues/new/choose). The
templates cover a bug report and a feature request; anything else is welcome as
a blank issue.
