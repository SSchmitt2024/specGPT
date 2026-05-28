"""Per-spec path / metadata resolution from the environment.

This lets the Phase-1 pipeline target either the NVMe **Base** specification
(the historical default) or the NVMe **PCIe Transport** specification without
editing any module constants. The interactive runner scripts
(``scripts/rerun_pipeline.sh`` and ``scripts/run_phase2.sh``) ask which spec to
build and export the variables below before invoking each ``python -m`` step.

Contract: **with no variables set, every helper returns the original
single-(Base)-spec value**, so existing behavior is unchanged.

Environment variables
----------------------
  NVME_SPEC         "base" | "pcie"   — logical spec id (default: "base")
  SPEC_DATA_DIR     output/intermediate JSON dir   (default: "data")
  SPEC_PDF_PATH     source PDF path                (default: per-module Base PDF)
  SPEC_PAGE_OFFSET  pdf_page → printed_page offset (default: per-module Base value)
  SPEC_DOCUMENT     spec_document tag on cards/chunks (default: Base title)
  SPEC_VERSION      spec_version tag on cards/chunks  (default: "2.1")

See docs/PCIE_MULTI_SPEC_PLAN.md for the full multi-spec design.
"""

from __future__ import annotations

import os
from pathlib import Path

# Canonical Base-spec defaults (the values these modules used before the
# multi-spec work). Kept here so the per-spec wiring has one source of truth.
DEFAULT_SPEC = "base"
DEFAULT_DATA_DIR = "data"
DEFAULT_PDF_PATH = "nvme_spec/NVMe_spec_full.pdf"
DEFAULT_SPEC_DOCUMENT = "NVM Express Base Specification"
DEFAULT_SPEC_VERSION = "2.1"


def spec() -> str:
    """Logical spec id, lower-cased. Defaults to ``"base"``."""
    return (os.getenv("NVME_SPEC") or DEFAULT_SPEC).strip().lower() or DEFAULT_SPEC


def data_dir() -> str:
    """Directory holding this spec's JSON artifacts. Defaults to ``"data"``."""
    return os.getenv("SPEC_DATA_DIR") or DEFAULT_DATA_DIR


def data_path(name: str) -> str:
    """Path to ``name`` inside the active spec's data dir, as a string."""
    return str(Path(data_dir()) / name)


def pdf_path(default: str = DEFAULT_PDF_PATH) -> str:
    """Source PDF path. ``SPEC_PDF_PATH`` wins; else the caller's Base default."""
    return os.getenv("SPEC_PDF_PATH") or default


def page_offset(default: int) -> int:
    """``pdf_page - printed_page`` offset. ``SPEC_PAGE_OFFSET`` wins; else the
    caller's existing per-module Base default (which differs slightly across
    modules, so it is passed in rather than centralized)."""
    raw = os.getenv("SPEC_PAGE_OFFSET")
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def spec_document(default: str = DEFAULT_SPEC_DOCUMENT) -> str:
    """``spec_document`` metadata tag written onto cards/chunks."""
    return os.getenv("SPEC_DOCUMENT") or default


def spec_version(default: str = DEFAULT_SPEC_VERSION) -> str:
    """``spec_version`` metadata tag written onto cards/chunks."""
    return os.getenv("SPEC_VERSION") or default
