from __future__ import annotations

import tomllib
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
_PYPROJECT = _ROOT / "pyproject.toml"
_CMAKE = _ROOT / "CMakeLists.txt"
_DARWIN_MLX = "mlx>=0.31.2,<0.32; platform_system == 'Darwin'"
_LINUX_MLX = "mlx[cpu]>=0.31.2,<0.32; platform_system == 'Linux'"
_NANOBIND = "nanobind==2.12.0"


def _load_pyproject() -> dict:
    return tomllib.loads(_PYPROJECT.read_text())


def test_runtime_dependencies_bound_mlx_and_exclude_pytest():
    project = _load_pyproject()["project"]
    dependencies = project["dependencies"]

    assert _DARWIN_MLX in dependencies
    assert _LINUX_MLX in dependencies
    assert not any(dep.startswith("pytest") for dep in dependencies)


def test_build_requires_match_runtime_mlx_pins():
    data = _load_pyproject()
    build_requires = data["build-system"]["requires"]
    runtime_dependencies = data["project"]["dependencies"]

    assert _NANOBIND in build_requires
    assert _DARWIN_MLX in build_requires
    assert _LINUX_MLX in build_requires
    assert _DARWIN_MLX in runtime_dependencies
    assert _LINUX_MLX in runtime_dependencies


def test_dev_extra_keeps_pytest_out_of_runtime_dependencies():
    project = _load_pyproject()["project"]
    dev_dependencies = project["optional-dependencies"]["dev"]

    assert any(dep.startswith("pytest") for dep in dev_dependencies)


def test_cmake_uses_cxx20_for_mlx_headers():
    cmake = _CMAKE.read_text()

    assert "set(CMAKE_CXX_STANDARD 20)" in cmake


def test_mlx_requirement_stays_within_one_minor():
    """MLX has no stable C++ ABI, so a single wheel must not span minors.

    `StreamOrDevice` gained a variant alternative in MLX 0.32.0, remangling
    every op that takes a stream, so a binary built against 0.31.x cannot load
    against 0.32.x. Widening this range past a minor would let pip resolve an
    MLX the extension cannot load.
    """
    data = _load_pyproject()
    for group in (data["build-system"]["requires"], data["project"]["dependencies"]):
        for dep in group:
            if not dep.startswith("mlx"):
                continue
            # Only the requirement half matters; the environment marker after
            # ";" legitimately contains "==" (platform_system == 'Darwin').
            requirement = dep.split(";", 1)[0]
            assert "<0.32" in requirement, f"unbounded MLX requirement: {dep}"
            assert "==" not in requirement, f"exact pin reintroduced: {dep}"


def test_cmake_declares_compatible_mlx_versions():
    cmake = _CMAKE.read_text()

    assert "MLX_AUDIO_IO_COMPATIBLE_MLX_VERSIONS" in cmake
