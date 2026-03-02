from __future__ import annotations

import importlib.resources as resources


def test_third_party_notice_files_present():
    pkg = resources.files("mlx_audio_io")
    required = [
        pkg / "THIRD_PARTY_NOTICES.md",
        pkg / "licenses" / "libsoxr" / "LICENCE",
        pkg / "licenses" / "libsoxr" / "COPYING.LGPL",
        pkg / "licenses" / "libsoxr" / "AUTHORS",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    assert not missing, f"Missing bundled notice files: {missing}"
