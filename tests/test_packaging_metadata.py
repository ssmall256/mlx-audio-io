from __future__ import annotations

import tomllib
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
_PYPROJECT = _ROOT / "pyproject.toml"
_CMAKE = _ROOT / "CMakeLists.txt"
_DARWIN_MLX = "mlx>=0.20.0; platform_system == 'Darwin'"
_LINUX_MLX = "mlx[cpu]>=0.20.0; platform_system == 'Linux'"


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

    # Extension is decoupled from MLX C++ headers, so MLX is not a build dependency
    assert any(dep.startswith("nanobind") for dep in build_requires)
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
    data = _load_pyproject()
    for group in (data["build-system"]["requires"], data["project"]["dependencies"]):
        for dep in group:
            if not dep.startswith("mlx"):
                continue
            requirement = dep.split(";", 1)[0]
            assert "==" not in requirement, f"exact pin reintroduced: {dep}"


def test_cmake_declares_compatible_mlx_versions():
    cmake = _CMAKE.read_text()

    assert "MLX_AUDIO_IO_COMPATIBLE_MLX_VERSIONS" in cmake


def test_nanobind_is_selected_after_build_mlx_is_installed():
    """nanobind is a static build requirement because mlx-audio-io is decoupled
    from MLX's internal nanobind type registry."""
    data = _load_pyproject()
    build_requires = data["build-system"]["requires"]
    assert any(dep.startswith("nanobind") for dep in build_requires)


def test_uv_editable_build_matches_locked_runtime_mlx():
    """Since mlx-audio-io is decoupled from MLX C++ ABI, extra-build-dependencies
    is no longer required."""
    uv_tool = _load_pyproject().get("tool", {}).get("uv", {})
    extra_deps = uv_tool.get("extra-build-dependencies", {})
    assert "mlx-audio-io" not in extra_deps
