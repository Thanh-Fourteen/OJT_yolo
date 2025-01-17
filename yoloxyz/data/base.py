# Ultralytics YOLO 🚀, AGPL-3.0 license

import glob
import math
import os
import torch
import random
from tqdm import tqdm
from copy import deepcopy
from itertools import repeat
from pathlib import Path
from typing import Optional
from multiprocessing.pool import ThreadPool

import cv2
import numpy as np
import psutil
from torch.utils.data import Dataset

from data.instance import Instances
from data.augment import Compose, LetterBox, Format, v8_transforms
from data.utils import DATASET_CACHE_VERSION, verify_image_label, save_dataset_cache_file, load_dataset_cache_file
from yolov9.utils.dataloaders import IMG_FORMATS, LOCAL_RANK, HELP_URL
from yolov9.utils.general import NUM_THREADS, LOGGER, resample_segments
from yolov9.utils.dataloaders import get_hash, img2label_paths


class BaseDataset(Dataset):
    """
    Base dataset class for loading and processing image data.

    Args:
        img_path (str): Path to the folder containing images.
        imgsz (int, optional): Image size. Defaults to 640.
        cache (bool, optional): Cache images to RAM or disk during training. Defaults to False.
        augment (bool, optional): If True, data augmentation is applied. Defaults to True.
        hyp (dict, optional): Hyperparameters to apply data augmentation. Defaults to None.
        prefix (str, optional): Prefix to print in log messages. Defaults to ''.
        rect (bool, optional): If True, rectangular training is used. Defaults to False.
        batch_size (int, optional): Size of batches. Defaults to None.
        stride (int, optional): Stride. Defaults to 32.
        pad (float, optional): Padding. Defaults to 0.0.
        single_cls (bool, optional): If True, single class training is used. Defaults to False.
        classes (list): List of included classes. Default is None.
        fraction (float): Fraction of dataset to utilize. Default is 1.0 (use all data).

    Attributes:
        im_files (list): List of image file paths.
        labels (list): List of label data dictionaries.
        ni (int): Number of images in the dataset.
        ims (list): List of loaded images.
        npy_files (list): List of numpy file paths.
        transforms (callable): Image transformation function.
    """

    def __init__(
        self,
        img_path,
        imgsz=640,
        cache=False,
        augment=True,
        hyp=None,
        prefix="",
        rect=False,
        batch_size=16,
        stride=32,
        pad=0.5,
        single_cls=False,
        classes=None,
        fraction=1.0,
    ):
        """Initialize BaseDataset with given configuration and options."""
        super().__init__()
        self.img_path = img_path
        self.imgsz = imgsz
        self.augment = augment
        self.single_cls = single_cls
        self.prefix = prefix
        self.fraction = fraction
        self.im_files = self.get_img_files(self.img_path)
        self.labels = self.get_labels()
        self.update_labels(include_class=classes)  # single_cls and include_class
        self.ni = len(self.labels)  # number of images
        self.rect = rect
        self.batch_size = batch_size
        self.stride = stride
        self.pad = pad
        if self.rect:
            assert self.batch_size is not None
            self.set_rectangle()

        # Buffer thread for mosaic images
        self.buffer = []  # buffer size = batch size
        self.max_buffer_length = min((self.ni, self.batch_size * 8, 1000)) if self.augment else 0

        # Cache images
        if cache == "ram" and not self.check_cache_ram():
            cache = False
        self.ims, self.im_hw0, self.im_hw = [None] * self.ni, [None] * self.ni, [None] * self.ni
        self.npy_files = [Path(f).with_suffix(".npy") for f in self.im_files]
        if cache:
            self.cache_images(cache)

        # Transforms
        self.transforms = self.build_transforms(hyp=hyp)

    def get_img_files(self, img_path):
        """Read image files."""
        try:
            f = []  # image files
            for p in img_path if isinstance(img_path, list) else [img_path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / "**" / "*.*"), recursive=True)
                    # F = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace("./", parent) if x.startswith("./") else x for x in t]  # local to global path
                        # F += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise FileNotFoundError(f"{self.prefix}{p} does not exist")
            im_files = sorted(x.replace("/", os.sep) for x in f if x.split(".")[-1].lower() in IMG_FORMATS)
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert im_files, f"{self.prefix}No images found in {img_path}"
        except Exception as e:
            raise FileNotFoundError(f"{self.prefix}Error loading data from {img_path}\n{HELP_URL}") from e
        if self.fraction < 1:
            im_files = im_files[: round(len(im_files) * self.fraction)]
        return im_files

    def update_labels(self, include_class: Optional[list]):
        """Update labels to include only these classes (optional)."""
        include_class_array = np.array(include_class).reshape(1, -1)
        for i in range(len(self.labels)):
            if include_class is not None:
                cls = self.labels[i]["cls"]
                bboxes = self.labels[i]["bboxes"]
                segments = self.labels[i]["segments"]
                keypoints = self.labels[i]["keypoints"]
                j = (cls == include_class_array).any(1)
                self.labels[i]["cls"] = cls[j]
                self.labels[i]["bboxes"] = bboxes[j]
                if segments:
                    self.labels[i]["segments"] = [segments[si] for si, idx in enumerate(j) if idx]
                if keypoints is not None:
                    self.labels[i]["keypoints"] = keypoints[j]
            if self.single_cls:
                self.labels[i]["cls"][:, 0] = 0

    def load_image(self, i, rect_mode=True):
        """Loads 1 image from dataset index 'i', returns (im, resized hw)."""
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                try:
                    im = np.load(fn)
                except Exception as e:
                    LOGGER.warning(f"{self.prefix}WARNING ⚠️ Removing corrupt *.npy image file {fn} due to: {e}")
                    Path(fn).unlink(missing_ok=True)
                    im = cv2.imread(f)  # BGR
            else:  # read image
                im = cv2.imread(f)  # BGR
            if im is None:
                raise FileNotFoundError(f"Image Not Found {f}")

            h0, w0 = im.shape[:2]  # orig hw
            if rect_mode:  # resize long side to imgsz while maintaining aspect ratio
                r = self.imgsz / max(h0, w0)  # ratio
                if r != 1:  # if sizes are not equal
                    w, h = (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))
                    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            elif not (h0 == w0 == self.imgsz):  # resize by stretching image to square imgsz
                im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)

            # Add to buffer if training with augmentations
            if self.augment:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
                self.buffer.append(i)
                if len(self.buffer) >= self.max_buffer_length:
                    j = self.buffer.pop(0)
                    self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None

            return im, (h0, w0), im.shape[:2]

        return self.ims[i], self.im_hw0[i], self.im_hw[i]

    def cache_images(self, cache):
        """Cache images to memory or disk."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        fcn = self.cache_images_to_disk if cache == "disk" else self.load_image
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(fcn, range(self.ni))
            pbar = tqdm(enumerate(results), total=self.ni, disable=LOCAL_RANK > 0)
            for i, x in pbar:
                if cache == "disk":
                    b += self.npy_files[i].stat().st_size
                else:  # 'ram'
                    self.ims[i], self.im_hw0[i], self.im_hw[i] = x  # im, hw_orig, hw_resized = load_image(self, i)
                    b += self.ims[i].nbytes
                pbar.desc = f"{self.prefix}Caching images ({b / gb:.1f}GB {cache})"
            pbar.close()

    def cache_images_to_disk(self, i):
        """Saves an image as an *.npy file for faster loading."""
        f = self.npy_files[i]
        if not f.exists():
            np.save(f.as_posix(), cv2.imread(self.im_files[i]), allow_pickle=False)

    def check_cache_ram(self, safety_margin=0.5):
        """Check image caching requirements vs available memory."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        n = min(self.ni, 30)  # extrapolate from 30 random images
        for _ in range(n):
            im = cv2.imread(random.choice(self.im_files))  # sample image
            ratio = self.imgsz / max(im.shape[0], im.shape[1])  # max(h, w)  # ratio
            b += im.nbytes * ratio**2
        mem_required = b * self.ni / n * (1 + safety_margin)  # GB required to cache dataset into RAM
        mem = psutil.virtual_memory()
        cache = mem_required < mem.available  # to cache or not to cache, that is the question
        if not cache:
            LOGGER.info(
                f'{self.prefix}{mem_required / gb:.1f}GB RAM required to cache images '
                f'with {int(safety_margin * 100)}% safety margin but only '
                f'{mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, '
                f"{'caching images ✅' if cache else 'not caching images ⚠️'}"
            )
        return cache

    def set_rectangle(self):
        """Sets the shape of bounding boxes for YOLO detections as rectangles."""
        bi = np.floor(np.arange(self.ni) / self.batch_size).astype(int)  # batch index
        nb = bi[-1] + 1  # number of batches

        s = np.array([x.pop("shape") for x in self.labels])  # hw
        ar = s[:, 0] / s[:, 1]  # aspect ratio
        irect = ar.argsort()
        self.im_files = [self.im_files[i] for i in irect]
        self.labels = [self.labels[i] for i in irect]
        ar = ar[irect]

        # Set training image shapes
        shapes = [[1, 1]] * nb
        for i in range(nb):
            ari = ar[bi == i]
            mini, maxi = ari.min(), ari.max()
            if maxi < 1:
                shapes[i] = [maxi, 1]
            elif mini > 1:
                shapes[i] = [1, 1 / mini]

        self.batch_shapes = np.ceil(np.array(shapes) * self.imgsz / self.stride + self.pad).astype(int) * self.stride
        self.batch = bi  # batch index of image

    def __getitem__(self, index):
        """Returns transformed label information for given index."""
        return self.transforms(self.get_image_and_label(index))

    def get_image_and_label(self, index):
        """Get and return label information from the dataset."""
        label = deepcopy(self.labels[index])  # requires deepcopy() https://github.com/ultralytics/ultralytics/pull/1948
        label.pop("shape", None)  # shape is for rect, remove it
        label["img"], label["ori_shape"], label["resized_shape"] = self.load_image(index)
        label["ratio_pad"] = (
            label["resized_shape"][0] / label["ori_shape"][0],
            label["resized_shape"][1] / label["ori_shape"][1],
        )  # for evaluation
        if self.rect:
            label["rect_shape"] = self.batch_shapes[self.batch[index]]
        return self.update_labels_info(label)

    def __len__(self):
        """Returns the length of the labels list for the dataset."""
        return len(self.labels)

    def update_labels_info(self, label):
        """Custom your label format here."""
        return label

    def build_transforms(self, hyp=None):
        """
        Users can customize augmentations here.

        Example:
            ```python
            if self.augment:
                # Training transforms
                return Compose([])
            else:
                # Val transforms
                return Compose([])
            ```
        """
        raise NotImplementedError

    def get_labels(self):
        """
        Users can customize their own format here.

        Note:
            Ensure output is a dictionary with the following keys:
            ```python
            dict(
                im_file=im_file,
                shape=shape,  # format: (height, width)
                cls=cls,
                bboxes=bboxes, # xywh
                segments=segments,  # xy
                keypoints=keypoints, # xy
                normalized=True, # or False
                bbox_format="xyxy",  # or xywh, ltwh
            )
            ```
        """
        raise NotImplementedError

class YOLODataset(BaseDataset):
    """
    Dataset class for loading object detection and/or segmentation labels in YOLO format.

    Args:
        data (dict, optional): A dataset YAML dictionary. Defaults to None.
        task (str): An explicit arg to point current task, Defaults to 'detect'.

    Returns:
        (torch.utils.data.Dataset): A PyTorch dataset object that can be used for training an object detection model.
    """

    def __init__(self, *args, data=None, task="detect", **kwargs):
        """Initializes the YOLODataset with optional configurations for segments and keypoints."""
        self.use_segments = task == "segment"
        self.use_keypoints = task == "pose"
        self.use_obb = task == "obb"
        self.data = data
        assert not (self.use_segments and self.use_keypoints), "Can not use both segments and keypoints."
        super().__init__(*args, **kwargs)

    def cache_labels(self, path=Path("./labels.cache")):
        """
        Cache dataset labels, check images and read shapes.

        Args:
            path (Path): path where to save the cache file (default: Path('./labels.cache')).
        Returns:
            (dict): labels.
        """
        x = {"labels": []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{self.prefix}Scanning {path.parent / path.stem}..."
        total = len(self.im_files)
        nkpt, ndim = self.data.get("kpt_shape", (0, 0))
        if self.use_keypoints and (nkpt <= 0 or ndim not in (2, 3)):
            raise ValueError(
                "'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'"
            )
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(
                func=verify_image_label,
                iterable=zip(
                    self.im_files,
                    self.label_files,
                    repeat(self.prefix),
                    repeat(self.use_keypoints),
                    repeat(len(self.data["names"])),
                    repeat(nkpt),
                    repeat(ndim),
                ),
            )
            pbar = tqdm(results, desc=desc, total=total)
            for im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x["labels"].append(
                        dict(
                            im_file=im_file,
                            shape=shape,
                            cls=lb[:, 0:1],  # n, 1
                            bboxes=lb[:, 1:],  # n, 4
                            segments=segments,
                            keypoints=keypoint,
                            normalized=True,
                            bbox_format="xywh",
                        )
                    )
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            pbar.close()

        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            LOGGER.warning(f"{self.prefix}WARNING ⚠️ No labels found in {path}. {HELP_URL}")
        x["hash"] = get_hash(self.label_files + self.im_files)
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        x["msgs"] = msgs  # warnings
        save_dataset_cache_file(self.prefix, path, x)
        return x

    def get_labels(self):
        """Returns dictionary of labels for YOLO training."""
        self.label_files = img2label_paths(self.im_files)
        cache_path = Path(self.label_files[0]).parent.with_suffix(".cache")
        try:
            cache, exists = load_dataset_cache_file(cache_path), True  # attempt to load a *.cache file
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache["hash"] == get_hash(self.label_files + self.im_files)  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(cache_path), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop("results")  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in (-1, 0):
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            tqdm(None, desc=self.prefix + d, total=n, initial=n)  # display results
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))  # display warnings

        # Read cache
        [cache.pop(k) for k in ("hash", "version", "msgs")]  # remove items
        labels = cache["labels"]
        if not labels:
            LOGGER.warning(f"WARNING ⚠️ No images found in {cache_path}, training may not work correctly. {HELP_URL}")
        self.im_files = [lb["im_file"] for lb in labels]  # update im_files

        # Check if the dataset is all boxes or all segments
        lengths = ((len(lb["cls"]), len(lb["bboxes"]), len(lb["segments"])) for lb in labels)
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            LOGGER.warning(
                f"WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = {len_segments}, "
                f"len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. "
                "To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset."
            )
            for lb in labels:
                lb["segments"] = []
        if len_cls == 0:
            LOGGER.warning(f"WARNING ⚠️ No labels found in {cache_path}, training may not work correctly. {HELP_URL}")
        return labels

    def build_transforms(self, hyp=None, mask_ratio = 4, overlap_mask = True):
        """Builds and appends transforms to the list."""
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=mask_ratio,
                mask_overlap=overlap_mask,
            )
        )
        return transforms

    def close_mosaic(self, hyp):
        """Sets mosaic, copy_paste and mixup options to 0.0 and builds transformations."""
        hyp.mosaic = 0.0  # set mosaic ratio=0.0
        hyp.copy_paste = 0.0  # keep the same behavior as previous v8 close-mosaic
        hyp.mixup = 0.0  # keep the same behavior as previous v8 close-mosaic
        self.transforms = self.build_transforms(hyp)

    def update_labels_info(self, label):
        """Custom your label format here."""
        # NOTE: cls is not with bboxes now, classification and semantic segmentation need an independent cls label
        # We can make it also support classification and semantic segmentation by add or remove some dict keys there.
        bboxes = label.pop("bboxes")
        segments = label.pop("segments", [])
        keypoints = label.pop("keypoints", None)
        bbox_format = label.pop("bbox_format")
        normalized = label.pop("normalized")

        # NOTE: do NOT resample oriented boxes
        segment_resamples = 100 if self.use_obb else 1000
        if len(segments) > 0:
            # list[np.array(1000, 2)] * num_samples
            # (N, 1000, 2)
            segments = np.stack(resample_segments(segments, n=segment_resamples), axis=0)
        else:
            segments = np.zeros((0, segment_resamples, 2), dtype=np.float32)
        label["instances"] = Instances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
        return label

    # @staticmethod
    # def collate_fn(batch):
    #     """Collates data samples into batches."""
    #     new_batch = {}
    #     keys = batch[0].keys()
    #     values = list(zip(*[list(b.values()) for b in batch]))
    #     for i, k in enumerate(keys):
    #         value = values[i]
    #         if k == "img":
    #             value = torch.stack(value, 0)
    #         if k in ["masks", "keypoints", "bboxes", "cls", "segments", "obb"]:
    #             value = torch.cat(value, 0)
    #         new_batch[k] = value
    #     new_batch["batch_idx"] = list(new_batch["batch_idx"])
    #     for i in range(len(new_batch["batch_idx"])):
    #         new_batch["batch_idx"][i] += i  # add target image index for build_targets()
    #     new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
    #     return new_batch
    @staticmethod
    def collate_fn(batch):
        """Collates data samples into batches and logs input/output."""
        with open(r"C:\Users\admin\Desktop\log.txt", "a") as log_file:
            log_file.write("Input Batch:\n")
            log_file.write(f"{batch}\n\n")

            new_batch = {}
            keys = batch[0].keys()
            values = list(zip(*[list(b.values()) for b in batch]))

            log_file.write("Keys and Values Extracted:\n")
            log_file.write(f"Keys: {keys}\n")
            log_file.write(f"Values: {values}\n\n")

            for i, k in enumerate(keys):
                value = values[i]
                if k == "img":
                    value = torch.stack(value, 0)
                if k in ["masks", "keypoints", "bboxes", "cls", "segments", "obb"]:
                    value = torch.cat(value, 0)
                new_batch[k] = value
                
                log_file.write(f"Processed Key: {k}\n")
                log_file.write(f"Value: {value}\n\n")

            new_batch["batch_idx"] = list(new_batch["batch_idx"])

            for i in range(len(new_batch["batch_idx"])):
                new_batch["batch_idx"][i] += i  # add target image index for build_targets()

            log_file.write("Updated Batch Indices:\n")
            log_file.write(f"{new_batch['batch_idx']}\n\n")

            new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
            
            log_file.write("Final Collated Batch:\n")
            log_file.write(f"{new_batch}\n")

        return new_batch
class RTDETRDataset(YOLODataset):
    """
    Real-Time DEtection and TRacking (RT-DETR) dataset class extending the base YOLODataset class.

    This specialized dataset class is designed for use with the RT-DETR object detection model and is optimized for
    real-time detection and tracking tasks.
    """

    def __init__(self, *args, data=None, **kwargs):
        """Initialize the RTDETRDataset class by inheriting from the YOLODataset class."""
        super().__init__(*args, data=data, **kwargs)