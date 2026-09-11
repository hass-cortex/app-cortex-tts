# Contributing to Hojo TTS

Thank you for your interest in contributing!

## Development Setup

### Prerequisites

- Python 3.12 or 3.13 (see `pyproject.toml`)
- [uv](https://docs.astral.sh/uv/) — this project does not use pip

### Build and run

```bash
cd hojo-tts
uv sync --frozen

# Run it against a scratch data directory, with auth off.
API_KEY= PRELOAD=false PORT=8799 \
  DATA_DIR=/tmp/hojo-data STATIC_DIR="$PWD/web" \
  uv run python -m hojo_tts
```

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
3. Ensure the gates pass:
   ```bash
   uv run ruff check src tests
   uv run ruff format --check src tests
   uv run --with pyright pyright
   uv run pytest -q
   ```
4. Write the PR description against the template: link the issue rather than
   restating it, say which approach you took and what you rejected, say how you
   verified it — and say plainly what you did **not** verify.
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

See [`AGENTS.md`](../AGENTS.md) for the module tree, the cross-module
guarantees and the endpoint reference. [`CONTEXT.md`](CONTEXT.md) defines the
vocabulary both use — read it before naming anything new, because most nouns in
this codebase already mean two things.

Top-level modules:

- `src/hojo_tts/text/` — the text path: normalise, convert, split into segments
- `src/hojo_tts/engine/` — the `Engine` protocol, the two implementations, and
  the registry that bounds how many stay in memory
- `src/hojo_tts/catalog.py` + `download.py` — what can be run, and getting it
  onto disk
- `src/hojo_tts/refs.py` — reference recordings, which are the 80M's voices
- `src/hojo_tts/api/` — FastAPI routes; thin shells over the modules above
- `web/` — the ingress admin UI, one ES module per concern

## Adding a Model

The catalog is hand-written (`src/hojo_tts/catalog.py`): two models, each a
Hugging Face repo id plus the list of files in its bundle. A new model needs an
entry there, and an `Engine` implementation unless it fits one of the two that
exist — `PresetEngine` for fixed voices, `CloneEngine` for reference-derived
ones.

Mind the cost figures on the entry (`size_mb`, `rtf_hint`, `rss_hint_mb`): the
admin UI shows them, and the integration reads `rtf_hint` to decide whether a
model can outrun playback and is therefore safe to stream sentence by sentence.
A figure invented rather than measured will make that decision wrongly.

## Questions?

Open a [Discussion](https://github.com/hass-cortex/app-hojo-tts/discussions).
