"""Locations used by the reproduction scripts.

Every location can be overridden with an environment variable, so the scripts run unchanged on any machine:

    METABOCAST_DATA_ROOT  folder holding the raw datasets (DESIUM/, SMA/, HCC/, GBM/ ...)   default: data/raw
    DESIUM_DIR            DESIUM paired DESI-MSI/Visium sections                           default: $METABOCAST_DATA_ROOT/DESIUM/Correlation_NMF_analysis/data
    METABOCAST_WORK       intermediate and output files written by the scripts             default: work
    METABOCAST_BASELINE_DIR  scFEA/MGFEA inputs and outputs                                default: $METABOCAST_WORK/baselines
    SCFEA_REPO            clone of github.com/changwn/scFEA                                default: external/scFEA
    MGFEA_REPO            clone of github.com/Sunwenzhilab/MGFEA                           default: external/MGFEA
"""
import os

_env = os.environ.get
ROOT = _env("METABOCAST_DATA_ROOT", os.path.join("data", "raw"))
DESIUM = _env("DESIUM_DIR", os.path.join(ROOT, "DESIUM", "Correlation_NMF_analysis", "data"))
WORK = _env("METABOCAST_WORK", "work")
BASELINES = _env("METABOCAST_BASELINE_DIR", os.path.join(WORK, "baselines"))
SCFEA_REPO = _env("SCFEA_REPO", os.path.join("external", "scFEA"))
MGFEA_REPO = _env("MGFEA_REPO", os.path.join("external", "MGFEA"))
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(WORK, exist_ok=True)
