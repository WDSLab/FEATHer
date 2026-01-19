from .multiscale import FEATHer_Multiscale, get_all_variants as get_multiscale_variants, get_variant_name as get_multiscale_variant_name
from .gating import FEATHer_Gating, get_all_variants as get_gating_variants, get_variant_name as get_gating_variant_name
from .complexity import FEATHer_Complexity, get_all_variants as get_complexity_variants, get_variant_name as get_complexity_variant_name
from .dtk import FEATHer_DTK, get_all_variants as get_dtk_variants, get_variant_name as get_dtk_variant_name
from .head import FEATHer_Head, get_all_variants as get_head_variants, get_variant_name as get_head_variant_name

__all__ = [
    # Multiscale
    "FEATHer_Multiscale",
    "get_multiscale_variants",
    "get_multiscale_variant_name",
    # Gating
    "FEATHer_Gating",
    "get_gating_variants",
    "get_gating_variant_name",
    # Complexity
    "FEATHer_Complexity",
    "get_complexity_variants",
    "get_complexity_variant_name",
    # DTK
    "FEATHer_DTK",
    "get_dtk_variants",
    "get_dtk_variant_name",
    # Head
    "FEATHer_Head",
    "get_head_variants",
    "get_head_variant_name",
]