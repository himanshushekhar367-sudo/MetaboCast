"""MetaboCast — prediction of spatial metabolite (MSI ion) maps from spatial transcriptomics, with per-ion reliability."""
from .metabocast import MetaboCast, MZ_ATOL, ALPHA_GRID, common_mz_axis, group_sections_by_mz_axis, select_alpha
from .pathways import PathwayScorer, MetabolicActivity, load_gmt
from .histology import add_histology_features
from .histology_only import MorphologyMetabolitePredictor   # EXPERIMENTAL — H&E-only companion model
from .celltype import add_celltype_features, LINEAGE_MARKERS   # REDCAT-inspired optional cell-type block
from .io import prepare_query, gene_panel_overlap, find_coordinates
from . import annotation                                       # putative exact-mass candidates (one reference)
__all__ = ["MetaboCast", "MZ_ATOL", "ALPHA_GRID", "select_alpha", "common_mz_axis", "group_sections_by_mz_axis",
           "PathwayScorer", "MetabolicActivity", "load_gmt",
           "add_histology_features", "MorphologyMetabolitePredictor",
           "add_celltype_features", "LINEAGE_MARKERS",
           "prepare_query", "gene_panel_overlap", "find_coordinates"]
__version__ = "1.0.0"
