from .nuscenes_dataset import nuScenesDataset


__dataset_cls__ = {
    "nuScenesDataset": nuScenesDataset,
}

__all__ = ["__dataset_cls__"]