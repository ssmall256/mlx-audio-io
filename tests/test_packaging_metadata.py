from __future__ import annotations

import tomllib
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
_PYPROJECT = _ROOT / "pyproject.toml"
_CMAKE = _ROOT / "CMakeLists.txt"
_DARWIN_MLX = "mlx>=0.31.2,<0.33; platform_system == 'Darwin'"
_LINUX_MLX = "mlx[cpu]>=0.31.2,<0.33; platform_system == 'Linux'"


def _load_pyproject() -> dict:
    return tomllib.loads(_PYPROJECT.read_text())


def test_runtime_dependencies_bound_mlx_and_exclude_pytest():
    project = _load_pyproject()["project"]
    dependencies = project["dependencies"]

    assert _DARWIN_MLX in dependencies
    assert _LINUX_MLX in dependencies
    assert not any(dep.startswith("pytest") for dep in dependencies)


def test_build_selects_mlx_and_nanobind_after_the_build_hook_runs():
    data = _load_pyproject()
    build_requires = data["build-system"]["requires"]
    runtime_dependencies = data["project"]["dependencies"]

    assert not any(dep.startswith("nanobind") for dep in build_requires)
    assert not any(dep.startswith("mlx") for dep in build_requires)
    assert _DARWIN_MLX in runtime_dependencies
    assert _LINUX_MLX in runtime_dependencies


def test_dev_extra_keeps_pytest_out_of_runtime_dependencies():
    project = _load_pyproject()["project"]
    dev_dependencies = project["optional-dependencies"]["dev"]

    assert any(dep.startswith("pytest") for dep in dev_dependencies)


def test_cmake_uses_cxx20_for_mlx_headers():
    cmake = _CMAKE.read_text()

    assert "set(CMAKE_CXX_STANDARD 20)" in cmake


def test_mlx_requirement_is_bounded_but_not_an_exact_pin():
    """The range must allow every MLX the tree can be built against, and no more.

    MLX has no stable C++ ABI: `StreamOrDevice` gained a variant alternative in
    0.32.0, which remangles every operation taking a stream, so a *built*
    extension only works with the MLX it was compiled against. That is enforced
    at runtime by the build-recorded compatible-version list in
    `_native_loader`, which fails loudly and names the versions it supports.

    The metadata range is therefore about what this source tree can build
    against, not what any one wheel can load. It must stay bounded -- an
    unbounded `mlx>=` would let a future ABI break resolve silently -- and it
    must never go back to an exact pin, which is what stopped downstream
    projects testing other MLX releases in the first place.
    """
    data = _load_pyproject()
    for group in (data["build-system"]["requires"], data["project"]["dependencies"]):
        for dep in group:
            if not dep.startswith("mlx"):
                continue
            # Only the requirement half matters; the environment marker after
            # ";" legitimately contains "==" (platform_system == 'Darwin').
            requirement = dep.split(";", 1)[0]
            assert "<" in requirement, f"unbounded MLX requirement: {dep}"
            assert "==" not in requirement, f"exact pin reintroduced: {dep}"


def test_cmake_declares_compatible_mlx_versions():
    cmake = _CMAKE.read_text()

    assert "MLX_AUDIO_IO_COMPATIBLE_MLX_VERSIONS" in cmake


def test_nanobind_is_selected_after_build_mlx_is_installed():
    """nanobind must match the version MLX itself was built with.

    The extension shares nanobind's type registry with mlx.core through
    NB_DOMAIN, so `mlx::core::array` crosses the boundary using *MLX's*
    registered caster. If the two are built with different nanobind versions
    the registry does not match and every call fails at runtime with
    "Unable to convert function return value to a Python type", even though
    the extension compiled and linked cleanly.

    MLX 0.31.2 builds with nanobind 2.12.0; MLX 0.32.2 uses 2.15.0,
    and MLX 0.32.3 uses 3.0.1. A static requirement would choose a single
    latest nanobind for all three.
    """
    data = _load_pyproject()
    build_requires = data["build-system"]["requires"]
    assert not any(dep.startswith("nanobind") for dep in build_requires)
    backend = (_ROOT / "build_backend.py").read_text()
    assert 'f"nanobind=={expected_nanobind}"' in backend


def test_uv_editable_build_matches_locked_runtime_mlx():
    uv_config = _load_pyproject()["tool"]["uv"]["extra-build-dependencies"]
    assert uv_config["mlx-audio-io"] == [
        {"requirement": "mlx[cpu]", "match-runtime": True}
    ]
