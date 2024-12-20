import os
import torch

from torch.utils.data import dataloader, distributed
from data.base import RTDETRDataset, YOLODataset

from backbones.yolov9.utils.general import LOGGER, colorstr
from backbones.yolov9.utils.torch_utils import torch_distributed_zero_first
from backbones.yolov9.utils.dataloaders import RANK, PIN_MEMORY, seed_worker

class _RepeatSampler:
    """
    Sampler that repeats forever.

    Args:
        sampler (Dataset.sampler): The sampler to repeat.
    """

    def __init__(self, sampler):
        """Initializes an object that repeats a given sampler indefinitely."""
        self.sampler = sampler

    def __iter__(self):
        """Iterates over the 'sampler' and yields its contents."""
        while True:
            yield from iter(self.sampler)

class InfiniteDataLoader(dataloader.DataLoader):
    """
    Dataloader that reuses workers.

    Uses same syntax as vanilla DataLoader.
    """

    def __init__(self, *args, **kwargs):
        """Dataloader that infinitely recycles workers, inherits from DataLoader."""
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        """Returns the length of the batch sampler's sampler."""
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        """Creates a sampler that repeats indefinitely."""
        for _ in range(len(self)):
            yield next(self.iterator)

    def reset(self):
        """
        Reset iterator.

        This is useful when we want to modify settings of dataset while training.
        """
        self.iterator = self._get_iterator()

def build_dataset(cfg, imgsz, cache, data, img_path, mode='val', batch=None):
    """Build RTDETR Dataset

    Args:
        img_path (str): Path to the folder containing images.
        mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
        batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
    """
    # return build_yolo_dataset(cfg, img_path, batch, data, mode=mode, rect=mode == "val", stride=gs)
    return RTDETRDataset(
        img_path=img_path,
        imgsz=imgsz,
        batch_size=batch,
        augment=mode == 'train',  # no augmentation
        hyp=cfg,
        rect=False,  # no rect
        cache=cache or None,
        prefix=colorstr(f'{mode}: '),
        data=data)

def build_dataloader(dataset, batch, workers, shuffle=True, rank=-1):
    """Return an InfiniteDataLoader or DataLoader for training or validation set."""
    batch = min(batch, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch, workers])  # number of workers
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    return InfiniteDataLoader(
        dataset=dataset,
        batch_size=batch,
        shuffle=shuffle and sampler is None,
        num_workers=nw,
        sampler=sampler,
        pin_memory=PIN_MEMORY,
        collate_fn=getattr(dataset, "collate_fn", None),
        worker_init_fn=seed_worker,
        generator=generator,
    )

def get_dataloader(cfg, imgsz, data, dataset_path, workers = 8, batch_size=16, rank=0, mode="train", cache = False):
    """Construct and return dataloader."""
    assert mode in ["train", "val"]
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        dataset = build_dataset(cfg, imgsz, cache, data, dataset_path, mode, batch_size)
    shuffle = mode == "train"
    if getattr(dataset, "rect", False) and shuffle:
        LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
        shuffle = False
    return build_dataloader(dataset, batch_size, workers, shuffle, rank)

def build_yolo_dataset(cfg, img_path, batch, data, mode="train", rect=False, stride=32):
    """Build YOLO Dataset."""
    return YOLODataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",  # augmentation
        hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
        rect=False,  # rectangular batches
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=int(stride),
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == "train" else 1.0,
    )

def build_val_dataset(opt, data, stride, img_path, mode="val", batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        return build_yolo_dataset(opt, img_path, batch, data, mode=mode, stride=stride)

def get_val_dataloader(opt, data, stride, dataset_path, batch_size, workers = 8):
    """Construct and return dataloader."""
    dataset = build_val_dataset(opt, data, stride, dataset_path, batch=batch_size, mode="val")
    return build_dataloader(dataset, batch_size, workers, shuffle=False, rank=-1) 