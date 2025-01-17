import os
import csv
import time
import math
import torch
import random
import numpy as np
import torch.nn as nn
from pathlib import Path
from copy import deepcopy
from datetime import datetime
from torch.optim import lr_scheduler

from torchkit.metrics import DetMetrics
from yolov9.utils.metrics import ConfusionMatrix, box_iou
from yolov9.utils.general import LOGGER, Profile, xywh2xyxy, check_amp, one_cycle, one_flat_cycle, scale_boxes
from yolov9.utils.torch_utils import smart_optimizer, de_parallel, ModelEMA
from lightning.pytorch import LightningModule

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
GIT_INFO = None

# Code thành 1 class
class LitYOLO(LightningModule):
    def __init__( self, opt, num_classes, model, model_device,  hyp, loss_fn=None ):
        super(LitYOLO, self).__init__()
        self.opt = opt
        self.dist = True if len(self.opt.device) > 1 else False
        self.num_classes = num_classes
        self.mloss = None
        self.model = model
        self.model_device = model_device
        self.gs = max(int(model.stride.max() if hasattr(model, "stride") else 32), 32)
        self.hyp = hyp

        #validate
        self.iouv = torch.linspace(0.5, 0.95, 10, device=model_device)
        self.niou = self.iouv.numel()
        self.best_fitness = 0.0
        self.save_dir = Path()
        self.csv = self.save_dir / "results.csv"
        w = self.save_dir / 'weights'
        if not os.path.exists(w):
            os.makedirs(w)
        self.last, self.best = w / 'last.pt', w / 'best.pt'

        #optimizer
        amp = check_amp(model)
        self.scaler = torch.cuda.amp.GradScaler(enabled=amp)
        self.ema = ModelEMA(model) if RANK in {-1, 0} else None

        # auto optimizer
        self.automatic_optimization = False
        self.last_opt_step = -1
        torch.use_deterministic_algorithms(False)

        # scheduler
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.start_epoch = 0
    
    def preprocess_batch(self, batch):
        batch["img"] = batch["img"].to(self.model_device, non_blocking=True).float() / 255
        if self.opt.multi_scale:
            imgs = batch["img"]
            sz = (
                random.randrange(self.opt.imgsz * 0.5, self.opt.imgsz * 1.5 + self.gs)
                // self.gs
                * self.gs
            )  # size
            sf = sz / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                ns = [
                    math.ceil(x * sf / self.gs) * self.gs for x in imgs.shape[2:]
                ]  # new shape (stretched to gs-multiple)
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs

        bs = len(batch["img"])
        batch_idx = batch["batch_idx"]
        gt_bbox, gt_class = [], []
        for i in range(bs):
            gt_bbox.append(batch["bboxes"][batch_idx == i].to(batch_idx.device))
            gt_class.append(batch["cls"][batch_idx == i].to(device=batch_idx.device, dtype=torch.long))
        return batch
    
    def on_train_epoch_start(self):
        self.mloss = None
        self.optimizer.zero_grad()

        self.tloss = None
       
    def training_step(self, batch, batch_idx):

        nb = self.trainer.num_training_batches
        nw = max(round(self.hyp['warmup_epochs'] * nb), 100) if self.hyp['warmup_epochs'] > 0 else -1
        ni = batch_idx + nb * self.current_epoch

        # Warmup
        if ni <= nw:
            xi = [0, nw]  # x interp
            self.accumulate = max(1, np.interp(ni, xi, [1, self.nbs / self.opt.batch_size]).round())
            for j, x in enumerate(self.optimizer.param_groups):
                # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                x['lr'] = np.interp(ni, xi, [self.hyp['warmup_bias_lr'] if j == 0 else 0.0, x['initial_lr'] * self.lf(self.current_epoch)])
                if 'momentum' in x:
                    x['momentum'] = np.interp(ni, xi, [self.hyp['warmup_momentum'], self.hyp['momentum']])

        batch = self.preprocess_batch(batch)
        with open(r"C:\Users\admin\Desktop\test.txt", "a") as f:
            f.write(f"batch_idx: {batch_idx}\n{batch}\n\n")
        loss, loss_items = self.model(batch)
        if RANK != -1:
            loss *= WORLD_SIZE
        self.mloss = ((self.mloss * batch_idx + loss_items) / (batch_idx + 1) if self.mloss is not None else loss_items)

        self.log('train/loss', loss, on_epoch=True, on_step=True, prog_bar=True, logger=True, sync_dist=self.dist)
        for idx, x in enumerate(['obj', 'cls','l1']):
            self.log(
                f'train/{x}',
                self.mloss[idx],
                on_epoch=True, 
                on_step=True,
                prog_bar=True, 
                logger=True,
                sync_dist=self.dist
            )

        #optimizer
        self.scaler.scale(loss).backward()
        if ni - self.last_opt_step >= self.accumulate:
            self.scaler.unscale_(self.optimizer)
        
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # clip gradients
            self.scaler.step(self.optimizer)  # optimizer.step
            self.scaler.update()
            self.optimizer.zero_grad()
            if self.ema:
                self.ema.update(self.model)
            self.last_opt_step = ni
    
    def on_train_epoch_end(self):
        self.lr = [x['lr'] for x in self.optimizer.param_groups]
        self.scheduler.step()
        self.ema.update_attr(self.model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])

    # validation
    def preprocess(self, batch):
        """Preprocesses batch of images for YOLO training."""
        batch["img"] = batch["img"].to(self.model_device, non_blocking=True)
        batch["img"] = batch["img"].float() / 255
        for k in ["batch_idx", "cls", "bboxes"]:
            batch[k] = batch[k].to(self.model_device)

        if self.opt.save_hybrid:
            height, width = batch["img"].shape[2:]
            nb = len(batch["img"])
            bboxes = batch["bboxes"] * torch.tensor((width, height, width, height), device=self.model_device)
            self.lb = (
                [
                    torch.cat([batch["cls"][batch["batch_idx"] == i], bboxes[batch["batch_idx"] == i]], dim=-1)
                    for i in range(nb)
                ]
                if self.opt.save_hybrid
                else []
            )  # for autolabelling

        return batch

    def postprocess(self, preds):
        """Apply Non-maximum suppression to prediction outputs."""
        num_select = 300
        bs, _, nd = preds[0].shape
        bboxes, scores = preds[0].split((4, nd - 4), dim=-1)
        # bboxes *= self.args.imgsz
        outputs = [torch.zeros((0, 6), device=bboxes.device)] * bs
        topk_values, topk_indexes = torch.topk(scores.reshape(scores.shape[0], -1), num_select, dim=1)
        topk_boxes = topk_indexes // scores.shape[2]
        labels = topk_indexes % scores.shape[2]
        bboxes = torch.gather(bboxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
        scores = topk_values
        for i, bbox in enumerate(bboxes):  # (300, 4)
            bbox = xywh2xyxy(bbox)
            score = scores[i]
            cls = labels[i]
            # Do not need threshold for evaluation as only got 300 boxes here
            # idx = score > self.args.conf
            pred = torch.cat([bbox, score[..., None], cls[..., None]], dim=-1)  # filter
            # Sort by confidence to correctly get internal metrics
            pred = pred[score.argsort(descending=True)]
            outputs[i] = pred  # [idx]
        return outputs
    
    def _prepare_batch(self, si, batch):
        """Prepares a batch of images and annotations for validation."""
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if len(cls):
            bbox = xywh2xyxy(bbox) * torch.tensor(imgsz, device=self.model_device)[[1, 0, 1, 0]]  # target boxes
            scale_boxes(imgsz, bbox, ori_shape, ratio_pad=ratio_pad)  # native-space labels
        return dict(cls=cls, bbox=bbox, ori_shape=ori_shape, imgsz=imgsz, ratio_pad=ratio_pad)
    
    def _prepare_pred(self, pred, pbatch):
        """Prepares a batch of images and annotations for validation."""
        predn = pred.clone()
        scale_boxes(
            pbatch["imgsz"], predn[:, :4], pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"]
        )  # native-space pred
        return predn.float()
    
    def match_predictions(self, pred_classes, true_classes, iou, use_scipy=False):
        """
        Matches predictions to ground truth objects (pred_classes, true_classes) using IoU.

        Args:
            pred_classes (torch.Tensor): Predicted class indices of shape(N,).
            true_classes (torch.Tensor): Target class indices of shape(M,).
            iou (torch.Tensor): An NxM tensor containing the pairwise IoU values for predictions and ground of truth
            use_scipy (bool): Whether to use scipy for matching (more precise).

        Returns:
            (torch.Tensor): Correct tensor of shape(N,10) for 10 IoU thresholds.
        """
        # Dx10 matrix, where D - detections, 10 - IoU thresholds
        correct = np.zeros((pred_classes.shape[0], self.iouv.shape[0])).astype(bool)
        # LxD matrix where L - labels (rows), D - detections (columns)
        correct_class = true_classes[:, None] == pred_classes
        iou = iou * correct_class  # zero out the wrong classes
        iou = iou.cpu().numpy()
        for i, threshold in enumerate(self.iouv.cpu().tolist()):
            if use_scipy:
                # WARNING: known issue that reduces mAP in https://github.com/ultralytics/ultralytics/pull/4708
                import scipy  # scope import to avoid importing for all commands

                cost_matrix = iou * (iou >= threshold)
                if cost_matrix.any():
                    labels_idx, detections_idx = scipy.optimize.linear_sum_assignment(cost_matrix, maximize=True)
                    valid = cost_matrix[labels_idx, detections_idx] > 0
                    if valid.any():
                        correct[detections_idx[valid], i] = True
            else:
                matches = np.nonzero(iou >= threshold)  # IoU > threshold and classes match
                matches = np.array(matches).T
                if matches.shape[0]:
                    if matches.shape[0] > 1:
                        matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                        # matches = matches[matches[:, 2].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                    correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)

    def _process_batch(self, detections, gt_bboxes, gt_cls):
        """
        Return correct prediction matrix.

        Args:
            detections (torch.Tensor): Tensor of shape [N, 6] representing detections.
                Each detection is of the format: x1, y1, x2, y2, conf, class.
            labels (torch.Tensor): Tensor of shape [M, 5] representing labels.
                Each label is of the format: class, x1, y1, x2, y2.

        Returns:
            (torch.Tensor): Correct prediction matrix of shape [N, 10] for 10 IoU levels.
        """
        iou = box_iou(gt_bboxes, detections[:, :4])
        return self.match_predictions(detections[:, 5], gt_cls, iou)
    
    def on_validation_epoch_start(self):
        LOGGER.info(f'\nValidating...')

        self.dt = (
            Profile(),
            Profile(),
            Profile(),
            Profile(),
        )
        self.t = 0
        self.loss = 0
        self.seen = 0
        self.batch_val = 0
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[])
        self.metrics = DetMetrics(save_dir=self.save_dir)
        self.metrics.names = self.model.names
        self.confusion_matrix = ConfusionMatrix(nc=self.num_classes)

    def validation_step(self, batch, augment = False):
        # Preprocess
        with self.dt[0]:
            batch = self.preprocess(batch)

        # Inference
        with self.dt[1]:
            preds = self.model(batch["img"], augment=augment)
        
        # Loss
        with self.dt[2]:
            self.loss += self.model.loss(batch, preds)[1]
        
        # Postprocess
        with self.dt[3]:
            preds = self.postprocess(preds)

        # Metrics
        self.update_metrics(preds, batch)

    def on_validation_epoch_end(self):
        stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in self.stats.items()}  # to numpy
        if len(stats) and stats["tp"].any():
            self.metrics.process(**stats)
        self.nt_per_class = np.bincount(
            stats["target_cls"].astype(int), minlength=self.num_classes
        )
        stats = self.metrics.results_dict
        self.metrics.confusion_matrix = self.confusion_matrix
        self.print_results()
        verbose = bool(self.current_epoch == self.trainer.max_epochs - 1)
        if (verbose):
            pf = "%12s" + "%11i" * 2 + "%11.3g" * len(self.metrics.keys)
            for i, c in enumerate(self.metrics.ap_class_index):
                LOGGER.info(pf % (self.metrics.names[c], self.seen, self.nt_per_class[c], *self.metrics.class_result(i)))
            self.save_metrics_to_csv()
        fi = self.validate(stats)
        self.save_model(fi)
    
    def validate(self, stats):
        results = {**stats, **self.label_loss_items(self.loss.cpu() / self.batch_val, prefix="val")}
        metrics = {k: round(float(v), 5) for k, v in results.items()}
        fitness = metrics.pop("fitness", -self.loss.detach().cpu().numpy())  # use loss as fitness measure if not found
        if not self.best_fitness or self.best_fitness < fitness:
            self.best_fitness = fitness
        return fitness
    
    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        Returns a loss dict with labelled training loss items tensor.

        Not needed for classification but necessary for segmentation & detection
        """
        self.loss_names = ("giou_loss", "cls_loss", "l1_loss")
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys
    
    def update_metrics(self, preds, batch):
        """Metrics."""
        for si, pred in enumerate(preds):
            self.seen += 1
            npr = len(pred)
            stat = dict(
                conf=torch.zeros(0, device=self.model_device),
                pred_cls=torch.zeros(0, device=self.model_device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.model_device),
            )
            pbatch = self._prepare_batch(si, batch)
            cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
            nl = len(cls)
            stat["target_cls"] = cls
            if npr == 0:
                if nl:
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                    # TODO: obb has not supported confusion_matrix yet.
                    if self.opt.plots and self.opt.task != "obb":
                        self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
                continue

            # Predictions
            if self.opt.single_cls:
                pred[:, 5] = 0
            predn = self._prepare_pred(pred, pbatch)
            stat["conf"] = predn[:, 4]
            stat["pred_cls"] = predn[:, 5]

            # Evaluate
            if nl:
                stat["tp"] = self._process_batch(predn, bbox, cls)
                # TODO: obb has not supported confusion_matrix yet.
            for k in self.stats.keys():
                self.stats[k].append(stat[k])

            # Save
            if self.opt.save_json:
                pass
                # self.pred_to_json(predn, batch["im_file"][si])
            if self.opt.save_txt:
                pass
                # file = self.save_dir / "labels" / f'{Path(batch["im_file"][si]).stem}.txt'
                # self.save_one_txt(predn, self.opt.save_conf, pbatch["ori_shape"], file)

    def save_metrics(self, metrics):
        """Saves training metrics to a CSV file."""
        keys, vals = list(metrics.keys()), list(metrics.values())
        n = len(metrics) + 1  # number of cols
        s = "" if self.csv.exists() else (("%23s," * n % tuple(["epoch"] + keys)).rstrip(",") + "\n")  # header
        with open(self.csv, "a") as f:
            f.write(s + ("%23.5g," * n % tuple([self.current_epoch + 1] + vals)).rstrip(",") + "\n")

    def save_model(self, fi):
        final_epoch = bool(self.current_epoch == self.trainer.max_epochs - 1)
        if (self.opt.nosave) or (final_epoch and not self.opt.evolve):  # if save
            ckpt = {
                'epoch': self.current_epoch,
                'best_fitness': self.best_fitness,
                'model': deepcopy(de_parallel(self.model)).half(),
                'ema': deepcopy(self.ema.ema).half(),
                'updates': self.ema.updates,
                'optimizer': self.optimizer.state_dict(),
                'opt': vars(self.opt),
                'git': GIT_INFO,  # {remote, branch, commit} if a git repo
                'date': datetime.now().isoformat()}
            torch.save(ckpt, self.last)
            if self.best_fitness == fi:
                torch.save(ckpt, self.best)
            if self.opt.save_period > 0 and self.current_epoch % self.opt.save_period == 0:
                torch.save(ckpt, self.save_dir / 'weights' / f'epoch{self.current_epoch}.pt')
            del ckpt

    def print_results(self):
        """Prints training/validation set metrics per class."""
        s = ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "Box(P", "R", "mAP50", "mAP50-95)")
        LOGGER.info(s)
        pf = "%22s" + "%11i" * 2 + "%11.3g" * len(self.metrics.keys)  # print format
        LOGGER.info(pf % ("all", self.seen, self.nt_per_class.sum(), *self.metrics.mean_results()))
        if self.nt_per_class.sum() == 0:
            LOGGER.warning(f"WARNING ⚠️ no labels found in {self.opt.task} set, can not compute metrics without labels")

    def save_metrics_to_csv(self, file_path=r'C:\Users\admin\Desktop\metrics.csv'):
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)

            # Viết header vào file CSV
            header = ["Class", "Images", "Instances", "Precision", "Recall", "mAP50", "mAP50-95", "seconds"]
            writer.writerow(header)

            # Ghi kết quả tổng hợp
            pf = "%12s" + "%11i" * 2 + "%11.3g" * len(self.metrics.keys) + "%15.3f"  # Định dạng in ra
            row = ["all", self.seen, self.nt_per_class.sum(), *self.metrics.mean_results(), time.time() - self.t]
            writer.writerow(row)

            # Ghi kết quả cho từng lớp
            pf = "%12s" + "%11i" * 2 + "%11.3g" * len(self.metrics.keys)
            for i, c in enumerate(self.metrics.ap_class_index):
                class_result = self.metrics.class_result(i)
                row = [self.metrics.names[c], self.seen, self.nt_per_class[c], *class_result]
                writer.writerow(row)

        print(f"Metrics saved to {file_path}")

    def configure_optimizers(self):
        self.nbs = 64  # nominal batch size
        self.accumulate = max(round(self.nbs / self.opt.batch_size), 1)  # accumulate loss before optimizing
        self.hyp['weight_decay'] *= self.opt.batch_size * self.accumulate / self.nbs
        optimizer = smart_optimizer(self.model, self.opt.optimizer, self.hyp['lr0'], self.hyp['momentum'], self.hyp['weight_decay'])
        
        if self.opt.cos_lr:
            self.lf = one_cycle(1, self.hyp['lrf'], self.opt.epochs)  # cosine 1->hyp['lrf']
        elif self.opt.flat_cos_lr:
            self.lf = one_flat_cycle(1, self.hyp['lrf'], self.opt.epochs)  # flat cosine 1->hyp['lrf']        
        elif self.opt.fixed_lr:
            self.lf = lambda x: 1.0
        else:
            self.lf = lambda x: (1 - x / self.opt.epochs) * (1.0 - self.hyp['lrf']) + self.hyp['lrf']  # linear
        
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=self.lf)
        scheduler.last_epoch = -1
        self.optimizer = optimizer  # Save for manual control
        self.scheduler = scheduler 
        return [optimizer], [scheduler]
