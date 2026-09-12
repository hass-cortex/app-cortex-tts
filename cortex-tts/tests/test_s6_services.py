"""The s6 service tree, where a filename is a contract.

s6-rc expresses dependencies as empty files *named* after the service they
point at, so renaming a service means renaming files that contain nothing.
A rename that rewrites file contents will not touch them, and nothing fails
until `s6-rc-compile` runs inside a Docker build — minutes later, on a host,
with the error arriving as a failed addon install:

    s6-rc-compile: fatal: during dependency resolution for service
    cortex-tts: undefined service name init-hojo-tts

These tests are the same check, in milliseconds, on a laptop.
"""

from __future__ import annotations

from pathlib import Path

import pytest

S6 = Path(__file__).resolve().parent.parent / "rootfs/etc/s6-overlay/s6-rc.d"


def _services() -> set[str]:
    """Every service directory, which is every legal dependency name."""
    return {p.name for p in S6.iterdir() if p.is_dir() and p.name != "user"}


def test_there_are_services_to_check() -> None:
    """A moved or renamed rootfs would make every other test vacuously pass."""
    assert _services(), f"no service directories under {S6}"


def test_every_declared_dependency_names_a_real_service() -> None:
    """The failure that motivated this file."""
    known = _services()
    missing: list[str] = []
    for service in S6.iterdir():
        deps = service / "dependencies.d"
        if not deps.is_dir():
            continue
        for dep in deps.iterdir():
            if dep.name not in known:
                missing.append(f"{service.name} depends on {dep.name}")
    assert not missing, "dependencies naming nothing: " + ", ".join(missing)


def test_every_enabled_service_exists() -> None:
    """`user/contents.d` enables services the same way: by filename."""
    known = _services()
    contents = S6 / "user/contents.d"
    enabled = {p.name for p in contents.iterdir()} if contents.is_dir() else set()
    assert enabled, "no services enabled — the app would start nothing"
    assert not (enabled - known), f"enabled but absent: {sorted(enabled - known)}"


@pytest.mark.parametrize("required", ["run", "type"])
def test_every_service_is_runnable(required: str) -> None:
    """A service directory without these is not a service."""
    incomplete = [s for s in _services() if not (S6 / s / required).is_file()]
    assert not incomplete, f"services with no {required}: {incomplete}"


def test_no_file_under_the_service_tree_names_a_stale_service() -> None:
    """Catches the other half: a path *inside* a script pointing at an old name.

    `init-cortex-tts/up` is a one-line path to its own `run`, so a rename that
    moved the directory but not that line would also only fail at build time.
    """
    known = _services()
    stale: list[str] = []
    for path in S6.rglob("*"):
        if not path.is_file() or path.stat().st_size == 0:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for name in ("hojo-tts", "init-hojo-tts", "hojo_tts"):
            if name in text and name not in known:
                stale.append(f"{path.relative_to(S6)} mentions {name}")
    assert not stale, "stale service names: " + ", ".join(stale)
