"""DeepDiscover: conservative differentiable hydrological models."""

from .api import DeepDiscover
from .model import (
    CRCNet, PositiveDense, InformationalMemoryCell, DiscoverUnit,
    MultiDUModel, RoutingKernel, conservative_allocation,
)
from .routing import TrainableRoutingMultiDUModel, DynamicRoutingMultiDUModel
from .boundaries import PrescribedSource, PrescribedSink, DemandLimitedSink, FunctionalSource, FunctionalSink
from .observations import (
    DirectObservation, StorageToHead, TrainableStorageToHead,
    StorageToDepth, StorageToMoisture,
)
from .snow import DifferentiableDegreeDaySnow, apply_snow
from .dataset import HydroDataset, HydroBatch
from .data import load_camels, load_camels_dataset, hamon_pet
from .structural import Candidate, compare_structures, equivalent_candidates, save_comparison
from .metrics import masked_mse, masked_mae, masked_rmse, masked_huber, masked_nse, hydro_metrics

__version__ = "1.0.0"

__all__ = [
    "DeepDiscover", "CRCNet", "PositiveDense", "InformationalMemoryCell",
    "DiscoverUnit", "MultiDUModel", "RoutingKernel", "conservative_allocation",
    "TrainableRoutingMultiDUModel", "DynamicRoutingMultiDUModel",
    "PrescribedSource", "PrescribedSink", "DemandLimitedSink",
    "FunctionalSource", "FunctionalSink", "DirectObservation", "StorageToHead",
    "TrainableStorageToHead", "StorageToDepth", "StorageToMoisture",
    "DifferentiableDegreeDaySnow", "apply_snow", "HydroDataset", "HydroBatch",
    "load_camels", "load_camels_dataset", "hamon_pet",
    "masked_mse", "masked_mae", "masked_rmse", "masked_huber", "masked_nse", "hydro_metrics",
    "Candidate", "compare_structures", "equivalent_candidates", "save_comparison",
]
