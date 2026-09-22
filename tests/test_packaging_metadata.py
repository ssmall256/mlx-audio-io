from __future__ import annotations

import tomllib
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
_PYPROJECT = _ROOT / "pyproject.toml"
_CMAKE = _ROOT / "CMakeLists.txt"
_DARWIN_MLX = "mlx>=0.31.2,<0.33; platform_system == 'Darwin'"
_LINUX_MLX = "mlx[cpu]>=0.31.2,<0.33; platform_system == 'Linux'"
_NANOBIND = "nanobind>=2.12.0,<2.16"


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


def test_nanobind_range_can_match_the_mlx_being_built_against():
    """nanobind must match the version MLX itself was built with.

    The extension shares nanobind's type registry with mlx.core through
    NB_DOMAIN, so `mlx::core::array` crosses the boundary using *MLX's*
    registered caster. If the two are built with different nanobind versions
    the registry does not match and every call fails at runtime with
    "Unable to convert function return value to a Python type", even though
    the extension compiled and linked cleanly.

    MLX 0.31.2 builds with nanobind 2.12.0; MLX 0.32.2 moved to 2.15.0
    (its CMakeLists FetchContent GIT_TAG). The range has to span both for a
    single source tree to build against either.
    """
    data = _load_pyproject()
    build_requires = data["build-system"]["requires"]
    nb = [dep for dep in build_requires if dep.startswith("nanobind")]
    assert len(nb) == 1, f"expected exactly one nanobind requirement, got {nb}"
    spec = nb[0]
    assert "==" not in spec, (
        f"nanobind is pinned exactly ({spec}); that forbids building against an "
        "MLX release that uses a different nanobind"
    )
    assert "2.12" in spec and "2.16" in spec, (
        f"nanobind range {spec} must cover both 2.12.0 (MLX 0.31.x) and "
        "2.15.0 (MLX 0.32.x)"
    )
