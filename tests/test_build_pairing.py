"""Build requirements and import-time checks for the native extension.

Since 1.3.23 the extension neither builds against MLX's C++ headers nor links
``libmlx``: audio crosses into MLX through DLPack and the buffer protocol. The
build therefore must not pull MLX or a pinned nanobind into pip's isolated
environment. The loader still understands the MLX/nanobind fields that older
builds recorded, and skips the check for builds that record none.
"""

from __future__ import annotations

import pathlib
import platform
import subprocess

import pytest

import build_backend
from mlx_audio_io import _native_loader

ROOT = pathlib.Path(__file__).resolve().parents[1]


class TestBuildRequirements:
    def test_the_hooks_add_nothing_to_the_backend_requirements(self, monkeypatch):
        monkeypatch.setattr(build_backend, "_call", lambda name, *a, **k: ["scikit-build-core-own"])
        assert build_backend.get_requires_for_build_wheel() == ["scikit-build-core-own"]
        assert build_backend.get_requires_for_build_editable() == ["scikit-build-core-own"]

    def test_static_build_requirements_do_not_include_mlx(self):
        tomllib = pytest.importorskip("tomllib")
        requires = tomllib.loads((ROOT / "pyproject.toml").read_text())["build-system"]["requires"]
        assert not [r for r in requires if r.split(";")[0].strip().lower().startswith("mlx")]
        assert any(r.startswith("nanobind") for r in requires)

    @pytest.mark.skipif(platform.system() != "Darwin", reason="otool is macOS-only")
    def test_the_extension_does_not_link_libmlx(self):
        import mlx_audio_io

        native = next(pathlib.Path(mlx_audio_io.__file__).parent.glob("_core*.so"), None)
        if native is None:
            pytest.skip("native extension not found next to the package")
        linked = subprocess.run(["otool", "-L", str(native)], capture_output=True, text=True, check=True).stdout
        assert "libmlx" not in linked


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
