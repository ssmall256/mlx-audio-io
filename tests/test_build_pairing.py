"""MLX and nanobind must be chosen together, at build time and checked at import.

mlx-audio-io publishes an sdist, so the extension is compiled on the installing
machine inside pip's build isolation. pip resolves `mlx` and `nanobind` from
`[build-system] requires` independently of each other and independently of the
environment the wheel is installed into, so a plain `pip install` can produce:

  * a nanobind MLX was never built with -> TypeError on the first call, with an
    error that never mentions nanobind
  * an MLX minor the user does not run -> dlopen symbol error at import

Both failure modes produce a binary that builds and links cleanly.
"""

from __future__ import annotations

import pytest

import build_backend
from mlx_audio_io import _native_loader


class TestPairedBuildRequirements:
    """The hook validates; it deliberately returns nothing.

    pip installs `[build-system] requires` into the isolated build environment
    before calling the hook, and rejects any exact pin returned from it that
    differs from what it already resolved ("build dependencies conflict with
    the backend dependencies"). Verified against a real TestPyPI sdist install.
    So the backend cannot correct the combination it was handed -- it can only
    refuse to build a binary that is known to be broken.
    """

    def test_a_consistent_environment_adds_no_requirements(self, monkeypatch):
        monkeypatch.delenv(build_backend._BUILD_MLX_ENV, raising=False)
        monkeypatch.setattr(
            build_backend,
            "_installed_version",
            lambda dist: {"mlx": "0.32.2", "nanobind": "2.15.0"}[dist],
        )
        assert build_backend.paired_build_requirements() == []

    def test_a_mismatched_pair_refuses_to_build(self, monkeypatch):
        monkeypatch.delenv(build_backend._BUILD_MLX_ENV, raising=False)
        monkeypatch.setattr(
            build_backend,
            "_installed_version",
            lambda dist: {"mlx": "0.32.2", "nanobind": "2.12.0"}[dist],
        )
        with pytest.raises(RuntimeError) as excinfo:
            build_backend.paired_build_requirements()
        text = str(excinfo.value)
        assert "nanobind/MLX mismatch" in text
        # The user will search for the error the broken binary would produce.
        assert "Unable to convert function return value" in text
        assert "--no-build-isolation" in text

    def test_the_env_var_refuses_when_it_cannot_be_honoured(self, monkeypatch):
        """It cannot change pip's isolated resolution, so it must not pretend."""
        monkeypatch.setenv(build_backend._BUILD_MLX_ENV, "0.31.2")
        monkeypatch.setattr(
            build_backend,
            "_installed_version",
            lambda dist: {"mlx": "0.32.2", "nanobind": "2.15.0"}[dist],
        )
        with pytest.raises(RuntimeError) as excinfo:
            build_backend.paired_build_requirements()
        text = str(excinfo.value)
        assert "0.31.2" in text and "0.32.2" in text
        assert "--no-build-isolation" in text
        # The recipe must name the nanobind for the MLX that was asked for.
        assert "nanobind==2.12.0" in text

    def test_the_env_var_is_satisfied_when_it_already_matches(self, monkeypatch):
        """The --no-build-isolation case: the caller's env is what gets used."""
        monkeypatch.setenv(build_backend._BUILD_MLX_ENV, "0.31.2")
        monkeypatch.setattr(
            build_backend,
            "_installed_version",
            lambda dist: {"mlx": "0.31.2", "nanobind": "2.12.0"}[dist],
        )
        assert build_backend.paired_build_requirements() == []

    def test_an_unknown_mlx_minor_is_not_second_guessed(self, monkeypatch):
        monkeypatch.delenv(build_backend._BUILD_MLX_ENV, raising=False)
        monkeypatch.setattr(
            build_backend,
            "_installed_version",
            lambda dist: {"mlx": "0.99.0", "nanobind": "2.12.0"}[dist],
        )
        assert build_backend.paired_build_requirements() == []

    def test_no_mlx_at_all_is_left_to_the_build_to_report(self, monkeypatch):
        monkeypatch.delenv(build_backend._BUILD_MLX_ENV, raising=False)
        monkeypatch.setattr(build_backend, "_installed_version", lambda dist: None)
        assert build_backend.paired_build_requirements() == []

    def test_the_hook_runs_the_check_and_returns_the_backend_list(self, monkeypatch):
        monkeypatch.delenv(build_backend._BUILD_MLX_ENV, raising=False)
        monkeypatch.setattr(
            build_backend,
            "_installed_version",
            lambda dist: {"mlx": "0.32.2", "nanobind": "2.12.0"}[dist],
        )
        monkeypatch.setattr(build_backend, "_call", lambda *a, **k: ["cmake"])
        with pytest.raises(RuntimeError, match="nanobind/MLX mismatch"):
            build_backend.get_requires_for_build_wheel()


class TestVerifyNanobindPairing:
    def test_a_matching_pair_is_accepted(self):
        _native_loader.verify_nanobind_pairing(
            {"build_mlx_version": "0.31.2", "build_nanobind_version": "2.12.0"}
        )

    def test_a_mismatch_is_rejected_at_import(self):
        with pytest.raises(RuntimeError, match="nanobind/MLX mismatch"):
            _native_loader.verify_nanobind_pairing(
                {"build_mlx_version": "0.32.2", "build_nanobind_version": "2.12.0"}
            )

    def test_the_message_names_both_versions_and_the_real_symptom(self):
        with pytest.raises(RuntimeError) as excinfo:
            _native_loader.verify_nanobind_pairing(
                {"build_mlx_version": "0.32.2", "build_nanobind_version": "2.12.0"}
            )
        text = str(excinfo.value)
        assert "0.32.2" in text and "2.12.0" in text and "2.15" in text
        # The user will search for the error they actually saw.
        assert "Unable to convert function return value" in text

    def test_the_override_downgrades_it_to_a_warning(self, monkeypatch):
        monkeypatch.setenv(_native_loader._ALLOW_MISMATCH_ENV, "1")
        with pytest.warns(RuntimeWarning, match="nanobind/MLX mismatch"):
            _native_loader.verify_nanobind_pairing(
                {"build_mlx_version": "0.32.2", "build_nanobind_version": "2.12.0"}
            )

    @pytest.mark.parametrize(
        "build",
        [
            {},
            {"build_mlx_version": "0.31.2"},
            {"build_nanobind_version": "2.12.0"},
            {"build_mlx_version": "0.31.2", "build_nanobind_version": "unknown"},
            {"build_mlx_version": "unknown", "build_nanobind_version": "2.12.0"},
            # An MLX minor with no recorded pairing must not be guessed at.
            {"build_mlx_version": "0.99.0", "build_nanobind_version": "2.12.0"},
        ],
    )
    def test_incomplete_or_unknown_metadata_skips_the_check(self, build):
        """Editable checkouts and future MLX releases must not hard-fail."""
        _native_loader.verify_nanobind_pairing(build)


def test_build_info_schema_keeps_the_nanobind_field():
    """`load_build_info` filters through a fixed key set, so a new field that is
    not listed there is silently dropped -- which has happened before."""
    from mlx_audio_io._build_info import load_build_info

    assert "build_nanobind_version" in load_build_info()


def test_the_rebuild_command_installs_its_own_build_tools():
    """`--no-build-isolation` means pip installs nothing for the build.

    scikit-build-core declares cmake and ninja dynamically, and turning
    isolation off skips that declaration -- so the command we print has to name
    them. A user following it verbatim on a machine without CMake got
    "No CMAKE_CXX_COMPILER could be found" instead of a working binary
    (ssmall256/mlx-audio-separator#4).
    """
    from mlx_audio_io import _native_loader

    texts = [
        _native_loader._REBUILD_REMEDIATION,
        build_backend.paired_build_requirements.__doc__ or "",
    ]
    hint = _native_loader._rebuild_hint("0.32.2", "0.31.2")
    texts.append(hint)

    for name, text in (("_REBUILD_REMEDIATION", texts[0]), ("_rebuild_hint", hint)):
        assert "--no-build-isolation" in text, name
        for tool in ("cmake", "ninja"):
            assert tool in text, f"{name} must tell the user to install {tool}"


def test_the_readme_rebuild_recipe_matches():
    """The README is where most people will find this, so keep it in step."""
    import pathlib

    readme = pathlib.Path(__file__).resolve().parents[1] / "README.md"
    if not readme.is_file():          # not shipped in the wheel
        pytest.skip("README.md is not available in this install")
    text = readme.read_text()
    recipe = [
        line for line in text.splitlines()
        if "--no-build-isolation" in line or "scikit-build-core" in line
    ]
    assert recipe, "README lost its rebuild recipe"
    joined = "\n".join(recipe)
    for tool in ("cmake", "ninja"):
        assert tool in joined, f"README rebuild recipe must install {tool}"
