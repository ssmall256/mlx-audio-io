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
    def test_nanobind_is_pinned_to_match_the_target_mlx(self, monkeypatch):
        monkeypatch.setenv(build_backend._BUILD_MLX_ENV, "0.31.2")
        assert build_backend.paired_build_requirements() == [
            "mlx==0.31.2",
            "nanobind==2.12.0",
        ]

    def test_a_different_mlx_minor_selects_a_different_nanobind(self, monkeypatch):
        monkeypatch.setenv(build_backend._BUILD_MLX_ENV, "0.32.2")
        assert build_backend.paired_build_requirements() == [
            "mlx==0.32.2",
            "nanobind==2.15.0",
        ]

    def test_without_the_env_var_it_pairs_off_the_build_environment(self, monkeypatch):
        """The common case: pip resolved some MLX, so pin the nanobind for it."""
        monkeypatch.delenv(build_backend._BUILD_MLX_ENV, raising=False)
        monkeypatch.setattr(
            build_backend, "_installed_version", lambda dist: "0.32.2"
        )
        assert build_backend.paired_build_requirements() == ["nanobind==2.15.0"]

    def test_an_unknown_mlx_minor_constrains_nothing(self, monkeypatch):
        """A future MLX must not be pinned to a nanobind nobody has verified."""
        monkeypatch.delenv(build_backend._BUILD_MLX_ENV, raising=False)
        monkeypatch.setattr(
            build_backend, "_installed_version", lambda dist: "0.99.0"
        )
        assert build_backend.paired_build_requirements() == []

    def test_no_mlx_at_all_constrains_nothing(self, monkeypatch):
        monkeypatch.delenv(build_backend._BUILD_MLX_ENV, raising=False)
        monkeypatch.setattr(build_backend, "_installed_version", lambda dist: None)
        assert build_backend.paired_build_requirements() == []

    def test_the_hook_appends_to_what_the_backend_asks_for(self, monkeypatch):
        monkeypatch.setenv(build_backend._BUILD_MLX_ENV, "0.31.2")
        monkeypatch.setattr(build_backend, "_call", lambda *a, **k: ["cmake"])
        assert build_backend.get_requires_for_build_wheel() == [
            "cmake",
            "mlx==0.31.2",
            "nanobind==2.12.0",
        ]


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
