"""MLX and nanobind must be chosen together, at build time and checked at import.

mlx-audio-io publishes an sdist, so the extension is compiled on the installing
machine inside pip's build isolation. The build hook selects a verified default
MLX/nanobind pair, or selects nanobind for MLX supplied by a uv consumer.
pip still resolves its isolated build independently of the target runtime.

A plain pip install can therefore build for an MLX version the user does not
run. The loader rejects that mismatch at import.
"""

from __future__ import annotations

import pytest

import build_backend
from mlx_audio_io import _native_loader


class TestPairedBuildRequirements:
    """The hook pins nanobind to the MLX in the build environment.

    Both MLX and nanobind are absent from static build requirements, so uv can
    supply runtime MLX before the hook, and the hook can request the exact
    compatible nanobind without a resolver conflict.
    """

    def test_a_consistent_environment_adds_no_requirements(self, monkeypatch):
        monkeypatch.delenv(build_backend._BUILD_MLX_ENV, raising=False)
        monkeypatch.setattr(
            build_backend,
            "_installed_version",
            lambda dist: {"mlx": "0.32.2", "nanobind": "2.15.0"}[dist],
        )
        assert build_backend.paired_build_requirements() == []

    def test_mlx_0323_requires_nanobind_3(self, monkeypatch):
        monkeypatch.delenv(build_backend._BUILD_MLX_ENV, raising=False)
        monkeypatch.setattr(
            build_backend,
            "_installed_version",
            lambda dist: {"mlx": "0.32.3", "nanobind": "3.0.1"}[dist],
        )
        assert build_backend.paired_build_requirements() == []

        monkeypatch.setattr(
            build_backend,
            "_installed_version",
            lambda dist: {"mlx": "0.32.3", "nanobind": "2.15.0"}[dist],
        )
        with pytest.raises(RuntimeError, match="nanobind/MLX mismatch"):
            build_backend.paired_build_requirements()

    @pytest.mark.parametrize(
        ("mlx_version", "nanobind_version"),
        [("0.31.2", "2.12.0"), ("0.32.2", "2.15.0"), ("0.32.3", "3.0.1")],
    )
    def test_missing_nanobind_is_requested_dynamically(
        self, monkeypatch, mlx_version, nanobind_version
    ):
        monkeypatch.delenv(build_backend._BUILD_MLX_ENV, raising=False)
        monkeypatch.setattr(
            build_backend,
            "_installed_version",
            lambda dist: mlx_version if dist == "mlx" else None,
        )
        monkeypatch.setattr(build_backend, "_call", lambda *args, **kwargs: [])
        expected = [f"nanobind=={nanobind_version}"]
        assert build_backend.get_requires_for_build_wheel() == expected
        assert build_backend.get_requires_for_build_editable() == expected

    def test_direct_build_requires_preinstalled_nanobind(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            build_backend,
            "_installed_version",
            lambda dist: "0.32.3" if dist == "mlx" else None,
        )
        with pytest.raises(RuntimeError, match="nanobind==3.0.1"):
            build_backend.build_wheel(str(tmp_path))

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

    @pytest.mark.parametrize("mlx_version", ["0.32.4", "0.99.0"])
    def test_an_unknown_mlx_version_is_rejected(self, monkeypatch, mlx_version):
        monkeypatch.delenv(build_backend._BUILD_MLX_ENV, raising=False)
        monkeypatch.setattr(
            build_backend,
            "_installed_version",
            lambda dist: {"mlx": mlx_version, "nanobind": "2.12.0"}[dist],
        )
        with pytest.raises(RuntimeError, match="No verified nanobind pairing"):
            build_backend.paired_build_requirements()

    def test_no_mlx_requests_a_verified_default_pair(self, monkeypatch):
        monkeypatch.delenv(build_backend._BUILD_MLX_ENV, raising=False)
        monkeypatch.setattr(build_backend.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(build_backend, "_installed_version", lambda dist: None)
        assert build_backend.paired_build_requirements() == [
            "mlx==0.32.3", "nanobind==3.0.1"
        ]

    def test_explicit_build_mlx_selects_a_paired_version(self, monkeypatch):
        monkeypatch.setenv(build_backend._BUILD_MLX_ENV, "0.31.2")
        monkeypatch.setattr(build_backend.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(build_backend, "_installed_version", lambda dist: None)
        assert build_backend.paired_build_requirements() == [
            "mlx==0.31.2", "nanobind==2.12.0"
        ]

    def test_linux_default_build_requests_cpu_extra(self, monkeypatch):
        monkeypatch.delenv(build_backend._BUILD_MLX_ENV, raising=False)
        monkeypatch.setattr(build_backend.platform, "system", lambda: "Linux")
        monkeypatch.setattr(build_backend, "_installed_version", lambda dist: None)
        assert build_backend.paired_build_requirements() == [
            "mlx[cpu]==0.32.3", "nanobind==3.0.1"
        ]

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

    def test_mlx_0323_pair_is_accepted_and_old_pair_rejected(self):
        _native_loader.verify_nanobind_pairing(
            {"build_mlx_version": "0.32.3", "build_nanobind_version": "3.0.1"}
        )
        with pytest.raises(RuntimeError, match="nanobind/MLX mismatch"):
            _native_loader.verify_nanobind_pairing(
                {"build_mlx_version": "0.32.3", "build_nanobind_version": "2.15.0"}
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
