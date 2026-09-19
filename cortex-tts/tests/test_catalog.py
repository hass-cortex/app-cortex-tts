"""The catalog's capability model and the backend table.

`EngineKind` used to answer three questions with one enum member, and a model
that had both bundled voices and cloning could not be spelled. These tests pin
the replacement: capabilities are independent, and a backend is looked up.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

from cortex_speech import BY_ID, CATALOG, ModelSpec, catalog
from cortex_speech.catalog import BundleSource
from cortex_speech.engine import backends
from cortex_speech.pacing.model import PRIOR_CJK_PER_S
from cortex_tts.preferences import Preferences


def _spec(
    *,
    builtin_voices: bool = False,
    cloning: bool = False,
    chunk_streaming: bool = False,
) -> ModelSpec:
    """A minimal spec; only the capability flags are worth varying here."""
    return ModelSpec(
        id="test-model",
        name="Test",
        description="",
        sources=(BundleSource(repo_id="acme/test", files=("a.onnx",)),),
        backend="hojo-preset",
        size_mb=1,
        languages=("zh",),
        builtin_voices=builtin_voices,
        cloning=cloning,
        chunk_streaming=chunk_streaming,
    )


class TestCapabilities:
    def test_a_model_can_have_builtin_voices_and_cloning(self) -> None:
        """The shape the old enum could not express, and the reason it went."""
        spec = _spec(builtin_voices=True, cloning=True)
        assert spec.builtin_voices and spec.cloning

    def test_capabilities_default_to_absent(self) -> None:
        """A new entry claims nothing it did not ask for."""
        spec = _spec()
        assert not spec.builtin_voices
        assert not spec.cloning
        assert not spec.chunk_streaming

    # Which engine module implements each cloning backend. A flag that says
    # what the model is told has to be checked against the code that tells
    # it, or the two drift and the disagreement is silent: the UI promises
    # the transcript matters, the model never sees it.
    CLONING_BACKENDS = {
        "hojo-clone": "clone.py",
        "moss": "moss.py",
        "qwen3-tts": "qwen3.py",
        "omnivoice": "omni.py",
    }

    def test_the_transcript_flag_matches_what_the_engine_passes(self) -> None:
        """`reads_reference_transcript` against the code that would read it.

        MOSS conditions on codec frames and its synthesis call takes no
        prompt text, so the transcript reaches it nowhere; the other three
        hand it to the model. Declared wrong in the permissive direction,
        the panel tells someone their transcript matters when nothing will
        ever read it.
        """
        source_dir = Path(__file__).resolve().parent.parent / "src/cortex_speech/engine"
        for spec in CATALOG:
            if not spec.cloning:
                continue
            module = self.CLONING_BACKENDS[spec.backend]
            source = (source_dir / module).read_text(encoding="utf-8")
            passes_it = ".transcript" in source
            assert passes_it == spec.reads_reference_transcript, (
                f"{spec.id}: declares reads_reference_transcript="
                f"{spec.reads_reference_transcript} but {module} "
                f"{'does' if passes_it else 'does not'} reference it"
            )

    def test_only_a_cloning_model_claims_to_read_a_transcript(self) -> None:
        """The flag is meaningless without a reference to have one."""
        for spec in CATALOG:
            if spec.reads_reference_transcript:
                assert spec.cloning, f"{spec.id} reads a transcript it can never get"

    def test_shipped_models_declare_what_they_actually_do(self) -> None:
        """The 40M has voices, the 80M clones; neither streams sub-sentence."""
        preset = BY_ID["hojo-40m"]
        clone = BY_ID["hojo-80m-clone"]
        assert (preset.builtin_voices, preset.cloning) == (True, False)
        assert (clone.builtin_voices, clone.cloning) == (False, True)
        assert not preset.chunk_streaming and not clone.chunk_streaming

    def test_every_entry_can_produce_a_voice_somehow(self) -> None:
        """A model with neither capability would be unselectable in the UI."""
        mute = [s.id for s in CATALOG if not (s.builtin_voices or s.cloning)]
        assert not mute, f"models that can never produce a voice: {mute}"

    def test_no_model_has_both_kinds_of_its_own_voices(self) -> None:
        """`routes.voice_kind` decides "builtin" or "designed" from the spec
        alone, without looking the voice up — it can do that only while no
        model declares both. An entry that did would mislabel every
        measurement it made, silently, so the build refuses it here."""
        both = [s.id for s in CATALOG if s.builtin_voices and s.designed_voices]
        assert both == []


class TestBackendTable:
    def test_every_catalog_backend_is_registered(self) -> None:
        """A typo in a spec's backend would otherwise surface at first load."""
        unknown = [s.id for s in CATALOG if s.backend not in backends.registered()]
        assert not unknown, f"catalog names unregistered backends: {unknown}"

    def test_an_unknown_backend_is_a_clear_error(self) -> None:
        """The message names what was asked for and what exists."""
        context = backends.BuildContext(
            directory=Path("/nowhere"),
            references=None,  # type: ignore[arg-type]
            num_threads=1,
            temperature=0.0,
        )
        with pytest.raises(backends.UnknownBackendError) as err:
            backends.build("not-a-backend", context)
        assert "not-a-backend" in str(err.value)
        assert "hojo-preset" in str(err.value)

    def test_registering_a_duplicate_is_refused(self) -> None:
        """Two modules claiming one key means one of them silently loses."""
        with pytest.raises(ValueError):
            backends.register("hojo-preset", lambda _: None)  # type: ignore[arg-type,return-value]


class TestListingVoicesLoadsNothing:
    """Naming a model's voices must not put the model in memory.

    `EngineRegistry.voices` used to `acquire()` the engine to call `voices()`
    on it. Both lists it needs are static files in the bundle, so that spent a
    multi-second load — and, at the default of one resident model, evicted
    whatever was speaking — to read a list of names. Asking for every model's
    voices in turn, which is exactly what the integration does at startup,
    therefore evicted a model per model.
    """

    def test_every_model_with_builtin_voices_can_list_them_off_disk(self) -> None:
        """Without a reader the registry would have to fall back to loading."""
        missing = [
            s.id
            for s in CATALOG
            if s.builtin_voices and s.backend not in backends.reads_voices()
        ]
        assert not missing, f"models whose voices need a load: {missing}"

    def test_a_backend_without_a_reader_says_so(self) -> None:
        """The 80M has no built-in voices, so it registers none."""
        with pytest.raises(backends.UnknownBackendError) as err:
            backends.own_voices("hojo-clone", Path("/nowhere"))
        assert "hojo-clone" in str(err.value)

    def test_a_reader_is_optional_at_registration(self) -> None:
        """A cloning-only backend must not be forced to write a stub."""
        backends.register("test-clone-only", lambda _: None)  # type: ignore[arg-type,return-value]
        try:
            assert "test-clone-only" in backends.registered()
            assert "test-clone-only" not in backends.reads_voices()
        finally:
            backends._BUILDERS.pop("test-clone-only")


class TestDefaultsMatchTheCatalog:
    """The settings the UI writes name models, and nothing links the two.

    These used to live in `config.yaml` as a `list(...)` Supervisor needed
    static, which made the catalog written down twice; the first time they
    disagreed the only symptom was a model missing from a dropdown. Storing
    them removes the duplicated list but not the coupling — `preferences`
    still validates against the catalog and still ships a default that has to
    exist.
    """

    def test_the_shipped_default_is_a_real_model(self) -> None:
        assert Preferences().default_model in BY_ID

    def test_every_model_can_be_chosen_as_the_default(self) -> None:
        chosen = [
            Preferences().merged({"default_model": spec.id}).default_model
            for spec in CATALOG
        ]
        assert chosen == [spec.id for spec in CATALOG]

    def test_models_may_be_kept_in_memory_up_to_the_catalog_size(self) -> None:
        """A ceiling below the model count makes a model unreachable together."""
        ceiling = Preferences().merged({"max_loaded_models": len(CATALOG)})
        assert ceiling.max_loaded_models == len(CATALOG)


class TestAddonOptions:
    """What is left in `config.yaml` is what a restart is the only way to change.

    Anything else there is a setting that costs a restart to change and a
    rebuild to rename, for no reason — which is what moving them out fixed.
    """

    def test_only_startup_settings_remain(self) -> None:
        config = (Path(__file__).resolve().parent.parent / "config.yaml").read_text()
        schema = yaml.safe_load(config)["schema"]
        assert set(schema) == {"log_level", "discovery_api_key"}


ENGINE_DIR = Path(__file__).resolve().parent.parent / "src" / "cortex_speech" / "engine"


def _registered_engine_classes() -> dict[str, tuple[str, str]]:
    """Map each backend key to the module and class its builder constructs.

    Derived from `backends.py` rather than written down, so a backend that is
    registered is automatically covered by everything below. A literal table
    here would fail open: the engine someone forgot to add is exactly the one
    whose contract was never checked.
    """
    tree = ast.parse((ENGINE_DIR / "backends.py").read_text(encoding="utf-8"))

    builders: dict[str, tuple[str, str]] = {}
    keys: dict[str, str] = {}
    for node in ast.walk(tree):
        # `def hojo_preset(context): from .preset import PresetEngine; return PresetEngine(...)`
        if isinstance(node, ast.FunctionDef):
            module = next(
                (n.module for n in ast.walk(node) if isinstance(n, ast.ImportFrom)),
                None,
            )
            built = next(
                (
                    n.func.id
                    for n in ast.walk(node)
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                ),
                None,
            )
            if module and built and built.endswith("Engine"):
                builders[node.name] = (f"{module}.py", built)
        # `register("moss", moss, voices=moss_voices)`
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "register"
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[1], ast.Name)
        ):
            keys[node.args[1].id] = node.args[0].value

    found = {keys[name]: where for name, where in builders.items() if name in keys}
    assert found, "could not read any engine class out of backends.py"
    return found


def _engine_init(module: str, class_name: str) -> ast.FunctionDef:
    """Return an engine's ``__init__`` without importing it.

    Loading one needs real weights and, for two of the three, torch.
    """
    for node in ast.walk(ast.parse((ENGINE_DIR / module).read_text(encoding="utf-8"))):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    return item
    raise AssertionError(f"no {class_name}.__init__ in {module}")


def _defines(module: str, class_name: str, method: str) -> bool:
    """Whether an engine class defines a method, read off disk."""
    for node in ast.walk(ast.parse((ENGINE_DIR / module).read_text(encoding="utf-8"))):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return any(
                isinstance(item, ast.FunctionDef) and item.name == method
                for item in node.body
            )
    raise AssertionError(f"no class {class_name} in {module}")


ENGINE_CLASSES = _registered_engine_classes()


class TestEveryEngineAnswersForItsProvider:
    """An engine takes the provider choice and reports what it actually got.

    Worth pinning because the first version of the GPU support shipped with
    two engines that were handed `execution_provider` by the backend table
    without accepting it, and logged a `self.provider` they never set — a
    `TypeError` and an `AttributeError` that nothing here could have caught,
    because nothing here loads a model.
    """

    @pytest.mark.parametrize("backend", sorted(ENGINE_CLASSES))
    def test_it_accepts_the_choice(self, backend: str) -> None:
        """`backends.py` passes this to every builder; refusing it is a crash."""
        init = _engine_init(*ENGINE_CLASSES[backend])
        names = {arg.arg for arg in init.args.args + init.args.kwonlyargs}
        assert "execution_provider" in names

    @pytest.mark.parametrize("backend", sorted(ENGINE_CLASSES))
    def test_it_reports_what_arrived(self, backend: str) -> None:
        """`registry.providers_in_use` reads this, and `/health` reports it."""
        init = _engine_init(*ENGINE_CLASSES[backend])
        assigned = {
            target.attr
            for node in ast.walk(init)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Attribute)
        }
        assert "provider" in assigned


class TestChunkStreamingIsOneFact:
    """`chunk_streaming` and `synthesize_stream` must agree.

    They are declared in different places for good reasons — the capability is
    catalog metadata a caller can read before anything is loaded, the method is
    what the registry finds with `isinstance`. Nothing tied them together, and
    the failure is silent in the direction that matters: a spec claiming the
    capability without the method makes `/api/speak/live` report
    `chunk_streaming: true` while the registry renders whole utterances.
    """

    @pytest.mark.parametrize("spec", CATALOG, ids=lambda spec: spec.id)
    def test_the_capability_matches_the_engine(self, spec: ModelSpec) -> None:
        module, class_name = ENGINE_CLASSES[spec.backend]
        assert spec.chunk_streaming == _defines(
            module, class_name, "synthesize_stream"
        ), (
            f"{spec.id} declares chunk_streaming={spec.chunk_streaming} but "
            f"{class_name} "
            f"{'defines' if not spec.chunk_streaming else 'does not define'} "
            "synthesize_stream"
        )

    def test_every_backend_a_model_names_is_registered(self) -> None:
        """A spec naming a backend nothing builds fails only on first use."""
        assert {spec.backend for spec in CATALOG} <= set(ENGINE_CLASSES)


class TestOrphanedBundles:
    """Weights outlive the catalog entry that named them.

    `inspect` only looks where a known id says to look, so a model dropped or
    renamed in a release leaves its bundle on disk — 2 GB, in MOSS's case —
    and nothing ever mentions it again.
    """

    def test_a_known_model_is_not_an_orphan(self, tmp_path: Path) -> None:
        (tmp_path / "models" / CATALOG[0].id).mkdir(parents=True)
        assert catalog.orphaned_bundles(tmp_path) == {}

    def test_an_unknown_directory_is_reported_with_its_size(
        self, tmp_path: Path
    ) -> None:
        bundle = tmp_path / "models" / "retired-model"
        (bundle / "nested").mkdir(parents=True)
        (bundle / "weights.onnx").write_bytes(b"x" * 900)
        (bundle / "nested" / "codec.onnx").write_bytes(b"y" * 100)
        assert catalog.orphaned_bundles(tmp_path) == {"retired-model": 1000}

    def test_a_fresh_install_has_nothing_to_report(self, tmp_path: Path) -> None:
        """No `models/` yet — startup must not fail on the reporting."""
        assert catalog.orphaned_bundles(tmp_path) == {}

    def test_a_stray_file_is_not_a_bundle(self, tmp_path: Path) -> None:
        (tmp_path / "models").mkdir(parents=True)
        (tmp_path / "models" / "notes.txt").write_text("x")
        assert catalog.orphaned_bundles(tmp_path) == {}


class TestTheAudioCeiling:
    """What one call may produce is the model's; what it costs is the host's.

    A generator ceiling truncates: past it the call returns the audio it had
    and the rest of the text is never spoken. That figure belongs to the model
    and is declared. How long a request should be *here* is a different
    question, and `batch_cap_s` answers it from this host's own samples — so a
    model with no ceiling of its own is given no number on its behalf.
    """

    ZH = "有一天，臥室裡的立扇決定離家出走。" * 40
    EN = "One day the fan in the bedroom decided to leave home. " * 40

    def test_a_model_with_no_established_ceiling_is_not_given_one(self) -> None:
        spec = BY_ID["omnivoice"]
        assert spec.max_audio_s is None
        assert spec.max_chars_per_segment is None
        assert spec.segment_limit(self.ZH) is None
        assert spec.segment_limit(self.EN) is None

    @pytest.mark.parametrize(
        ("model", "seconds"),
        [
            ("hojo-40m", 40.96),
            ("hojo-80m-clone", 40.96),
            ("moss-nano", 30.0),
            ("qwen3-tts-0.6b", 163.84),
            ("qwen3-tts-0.6b-clone", 163.84),
        ],
    )
    def test_a_declared_ceiling_is_what_the_model_can_produce(
        self, model: str, seconds: float
    ) -> None:
        assert BY_ID[model].max_audio_s == seconds

    def test_the_limit_is_in_characters_and_so_differs_by_script(self) -> None:
        """One ceiling, two answers: CJK speaks about 4.1 characters a second
        against Latin's 14.7, so the same seconds are very different texts."""
        spec = BY_ID["moss-nano"]
        assert spec.segment_limit(self.ZH) == int(30.0 * PRIOR_CJK_PER_S)
        assert spec.segment_limit(self.EN) > 3 * spec.segment_limit(self.ZH)

    def test_no_model_can_be_handed_more_audio_than_it_declares(self) -> None:
        """The invariant the ceiling exists to keep, over the whole catalog."""
        for spec in CATALOG:
            if spec.max_audio_s is None:
                continue
            for text, rate in ((self.ZH, PRIOR_CJK_PER_S), (self.EN, 14.7)):
                limit = spec.segment_limit(text)
                assert limit is not None
                assert limit / rate <= spec.max_audio_s + 1e-6

    def test_an_empty_text_falls_back_rather_than_dividing_by_nothing(self) -> None:
        spec = BY_ID["moss-nano"]
        assert spec.segment_limit("   ") == spec.max_chars_per_segment
