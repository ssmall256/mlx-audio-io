#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


SYSTEM_ABS_PREFIXES_MACOS = ("/System/Library/", "/usr/lib/")
DELOCATE_INTERNAL_PREFIXES_MACOS = ("/DLC/",)
SYSTEM_ABS_PREFIXES_LINUX = ("/lib/", "/usr/lib/", "/lib64/", "/usr/lib64/")
REQUIRED_NOTICE_FILES = (
    "mlx_audio_io/THIRD_PARTY_NOTICES.md",
    "mlx_audio_io/licenses/libsoxr/LICENCE",
    "mlx_audio_io/licenses/libsoxr/COPYING.LGPL",
    "mlx_audio_io/licenses/libsoxr/AUTHORS",
)


def _run(cmd: list[str], *, cwd: Path | None = None) -> str:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        detail = "\n".join(x for x in (proc.stdout.strip(), proc.stderr.strip()) if x)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{detail}")
    return proc.stdout


def _iter_binaries(root: Path) -> list[Path]:
    out: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix in {".so", ".dylib"}:
            out.append(p)
            continue
        if ".so." in p.name:
            out.append(p)
    return sorted(out)


def _check_macos_binary(binary: Path) -> list[str]:
    errors: list[str] = []
    out = _run(["otool", "-L", str(binary)])
    deps = []
    for line in out.splitlines()[1:]:
        line = line.strip()
        if not line:
            continue
        deps.append(line.split(" (", 1)[0])

    for dep in deps:
        if dep.startswith("@"):
            continue
        if dep.startswith("/"):
            if dep.startswith(DELOCATE_INTERNAL_PREFIXES_MACOS):
                continue
            if dep.startswith(SYSTEM_ABS_PREFIXES_MACOS):
                continue
            errors.append(
                f"{binary}: disallowed absolute dependency path: {dep}"
            )
    return errors


def _extract_readelf_tags(readelf_out: str, tag: str) -> list[str]:
    pattern = re.compile(rf"\({re.escape(tag)}\).*?\[(.*?)\]")
    return pattern.findall(readelf_out)


def _check_linux_binary(binary: Path) -> list[str]:
    errors: list[str] = []
    out = _run(["readelf", "-d", str(binary)])
    needed = _extract_readelf_tags(out, "NEEDED")
    runpaths = _extract_readelf_tags(out, "RUNPATH") + _extract_readelf_tags(out, "RPATH")

    for dep in needed:
        if dep.startswith("/"):
            if dep.startswith(SYSTEM_ABS_PREFIXES_LINUX):
                continue
            errors.append(f"{binary}: disallowed absolute NEEDED path: {dep}")

    for entry in runpaths:
        for token in entry.split(":"):
            token = token.strip()
            if not token or token.startswith("$ORIGIN"):
                continue
            if token.startswith("/") and not token.startswith(SYSTEM_ABS_PREFIXES_LINUX):
                errors.append(f"{binary}: disallowed absolute RUNPATH/RPATH entry: {token}")
    return errors


def _wheel_contains_soxr(zip_names: list[str]) -> bool:
    for name in zip_names:
        base = name.rsplit("/", 1)[-1]
        if base.startswith("libsoxr") and (base.endswith(".dylib") or ".so" in base):
            return True
    return False


def check_wheel(wheel_path: Path, *, require_third_party_notices: bool) -> None:
    if not wheel_path.exists():
        raise FileNotFoundError(f"wheel not found: {wheel_path}")

    with zipfile.ZipFile(wheel_path) as zf:
        names = zf.namelist()
        if require_third_party_notices and _wheel_contains_soxr(names):
            missing = [name for name in REQUIRED_NOTICE_FILES if name not in names]
            if missing:
                raise RuntimeError(
                    "Wheel bundles libsoxr but is missing required notice files: "
                    + ", ".join(missing)
                )

        with tempfile.TemporaryDirectory(prefix="mlx_audio_io_wheel_check_") as tmpdir:
            tmp = Path(tmpdir)
            zf.extractall(tmp)
            binaries = _iter_binaries(tmp)
            if not binaries:
                raise RuntimeError(f"No shared libraries found in wheel: {wheel_path}")

            errors: list[str] = []
            for binary in binaries:
                if sys.platform == "darwin":
                    errors.extend(_check_macos_binary(binary))
                elif sys.platform.startswith("linux"):
                    errors.extend(_check_linux_binary(binary))
                else:
                    raise RuntimeError(f"Unsupported platform for linkage check: {sys.platform}")

            if errors:
                raise RuntimeError(
                    "Wheel linkage check failed:\n" + "\n".join(f"- {e}" for e in errors)
                )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify wheel linkage does not embed absolute host library paths."
    )
    parser.add_argument("wheels", nargs="+", type=Path, help="Wheel file(s) to inspect")
    parser.add_argument(
        "--no-require-third-party-notices",
        action="store_true",
        help="Do not enforce third-party notice files when libsoxr is bundled",
    )
    args = parser.parse_args()

    require_third_party_notices = not args.no_require_third_party_notices
    for wheel in args.wheels:
        check_wheel(wheel, require_third_party_notices=require_third_party_notices)
        print(f"[ok] {wheel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
