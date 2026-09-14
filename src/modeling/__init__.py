"""Contratos pequenos y reutilizables para portar los modelos."""

from .adapters import (
    DirichletAdapter,
    GLMAdapter,
    HierarchicalNBAdapter,
    M3Adapter,
    RFAdapter,
    RFHorizonAdapter,
)
from .bundle import load_bundle, publish_pointer, read_pointer, save_bundle
from .contracts import ModelAdapter, nonnegative, select_features, train_target
from .extrapolation import attach_extrapolation
from .registry import FAMILY_ORDER, OPERATIONAL_MODELS, best_by_family

__all__ = [
    "DirichletAdapter",
    "FAMILY_ORDER",
    "GLMAdapter",
    "HierarchicalNBAdapter",
    "M3Adapter",
    "ModelAdapter",
    "OPERATIONAL_MODELS",
    "RFAdapter",
    "RFHorizonAdapter",
    "attach_extrapolation",
    "best_by_family",
    "load_bundle",
    "nonnegative",
    "publish_pointer",
    "read_pointer",
    "save_bundle",
    "select_features",
    "train_target",
]