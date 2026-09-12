## Issue

<!-- Link it; do not restate it. "Closes #123" or "No issue: <one line>". -->

## Approach

<!-- What this change does about it, and why this way. Name the alternatives
you tried or considered and what rejected them, with the evidence. -->

## Verification

- [ ] `uv run ruff check src tests` clean
- [ ] `uv run ruff format --check src tests` clean
- [ ] `uv run --with pyright pyright` clean
- [ ] `uv run pytest -q` passes

<!-- Anything beyond the gates: which model, which host, what you listened to
or measured. Numbers beat adjectives. -->

## Not verified

<!-- Say plainly what you did not run or could not measure — a model you do
not have, a host you do not own, a path only ingress exercises. -->

## Blast radius

<!-- Shared code touched, behaviour that changes for existing users, the wire
(`api_version`), anything a maintainer must judge. "None" is an answer. -->
