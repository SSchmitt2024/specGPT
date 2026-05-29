"""
Unit tests for per-spec path/metadata resolution (src/spec_env.py).

Pure-Python (os/pathlib only) — no Supabase/Voyage/Anthropic. Run from root:

    venv/bin/python3 -m pytest tests/test_spec_env.py

Or without pytest:

    venv/bin/python3 tests/test_spec_env.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make `src` importable when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import spec_env


_SPEC_VARS = (
    "NVME_SPEC", "SPEC_DATA_DIR", "SPEC_PDF_PATH",
    "SPEC_PAGE_OFFSET", "SPEC_DOCUMENT", "SPEC_VERSION",
)


def _clear_env():
    for k in _SPEC_VARS:
        os.environ.pop(k, None)


def test_defaults_reproduce_base_behavior():
    """With no env set, every helper returns the original Base value."""
    _clear_env()
    assert spec_env.spec() == "base"
    assert spec_env.data_dir() == "data"
    assert spec_env.data_path("toc.json") == str(Path("data") / "toc.json")
    assert spec_env.pdf_path() == "nvme_spec/NVMe_spec_full.pdf"
    # page_offset returns the caller's per-module default when unset.
    assert spec_env.page_offset(24) == 24
    assert spec_env.page_offset(23) == 23
    assert spec_env.spec_document() == "NVM Express Base Specification"
    assert spec_env.spec_version() == "2.1"


def test_pcie_overrides_apply():
    _clear_env()
    os.environ.update({
        "NVME_SPEC": "pcie",
        "SPEC_DATA_DIR": "data/pcie",
        "SPEC_PDF_PATH": "nvme_spec/NVMe_PCIe_transport.pdf",
        "SPEC_PAGE_OFFSET": "12",
        "SPEC_DOCUMENT": "NVM Express PCIe Transport Specification",
        "SPEC_VERSION": "1.1",
    })
    try:
        assert spec_env.spec() == "pcie"
        assert spec_env.data_dir() == "data/pcie"
        assert spec_env.data_path("tables.json") == str(Path("data/pcie") / "tables.json")
        assert spec_env.pdf_path() == "nvme_spec/NVMe_PCIe_transport.pdf"
        # Explicit offset wins over the per-module default. SPEC_PAGE_OFFSET is
        # in the baseline (page-iteration, 23) convention; toc_rebuild's 24
        # default keeps its +1 delta so toc and content pages stay aligned.
        assert spec_env.page_offset(23) == 12
        assert spec_env.page_offset(24) == 13
        assert spec_env.spec_document() == "NVM Express PCIe Transport Specification"
        assert spec_env.spec_version() == "1.1"
    finally:
        _clear_env()


def test_spec_is_normalized_lowercase():
    _clear_env()
    os.environ["NVME_SPEC"] = "  PCIe  "
    try:
        assert spec_env.spec() == "pcie"
    finally:
        _clear_env()


def test_empty_page_offset_falls_back_to_default():
    _clear_env()
    os.environ["SPEC_PAGE_OFFSET"] = ""
    try:
        assert spec_env.page_offset(23) == 23
    finally:
        _clear_env()


if __name__ == "__main__":
    test_defaults_reproduce_base_behavior()
    test_pcie_overrides_apply()
    test_spec_is_normalized_lowercase()
    test_empty_page_offset_falls_back_to_default()
    print("all spec_env tests passed")
