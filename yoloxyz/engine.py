import os
import math
import torch
import random
import numpy as np
import torch.nn as nn
from pathlib import Path
from copy import deepcopy
from datetime import datetime
from torch.optim import lr_scheduler
from lightning.pytorch import LightningModule

from yolov9.utils.metrics import ConfusionMatrix, box_iou, ap_per_class, fitness
from yolov9.utils.general import LOGGER, Profile, non_max_suppression, scale_boxes, xywh2xyxy, xyxy2xywh, check_amp, one_cycle, one_flat_cycle
from yolov9.utils.loss_tal_dual import ComputeLoss
from yolov9.utils.torch_utils import smart_optimizer, ModelEMA, de_parallel
from multitasks.utils.loss_rtdetr import RTDETRDetectionLoss

# Environment variables
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

class LitYOLO(LightningModule):
    def __init__(self, opt, model, hyp, num_classes):
        super().__init__()
        self.opt = opt
        self.model = model
        self.hyp = hyp
        self.dist = len(opt.device) > 1
        self.gs = max(int(model.stride.max()), 32)
        self.detr = opt.lastlayer == 'DETR'
        LOGGER.info(f"\n*** DETR = {self.detr} ***\n")

        self.compute_loss = RTDETRDetectionLoss(num_classes, use_vfl=True) if self.detr else ComputeLoss(model)
        self.scaler = torch.cuda.amp.GradScaler(enabled=check_amp(model))
        self.ema = ModelEMA(model)
        self.automatic_optimization = False
        self.last_opt_step = -1
        torch.use_deterministic_algorithms(False)
        self.names = self.model.names if hasattr(self.model, 'names') else self.model.module.names
        self.best_fitness = 0.0

    def configure_optimizers(self):
        self.nbs = 64
        self.accumulate = max(round(self.nbs / self.opt.batch_size), 1)
        self.hyp['weight_decay'] *= self.opt.batch_size * self.accumulate / self.nbs
        optimizer = smart_optimizer(self.model, self.opt.optimizer, self.hyp['lr0'], self.hyp['momentum'], self.hyp['weight_decay'])

        if self.opt.cos_lr:
            self.lf = one_cycle(1, self.hyp['lrf'], self.opt.epochs)
        elif self.opt.flat_cos_lr:
            self.lf = one_flat_cycle(1, self.hyp['lrf'], self.opt.epochs)
        elif self.opt.fixed_lr:
            self.lf = lambda x: 1.0
        else:
            self.lf = lambda x: (1 - x / self.opt.epochs) * (1.0 - self.hyp['lrf']) + self.hyp['lrf']

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=self.lf)
        scheduler.last_epoch = -1
        self.optimizer = optimizer
        self.scheduler = scheduler
        return [optimizer], [scheduler]

    def on_train_epoch_start(self):
        self.mloss = torch.zeros(3, device=self.device)
        self.optimizer.zero_grad()

    def training_step(self, batch, batch_idx):
        imgs, targets, _, _ = batch
        imgs = imgs.to(self.device, non_blocking=True).float() / 255
        nb = self.trainer.num_training_batches
        ni = batch_idx + nb * self.current_epoch
        nw = max(round(self.hyp['warmup_epochs'] * nb), 100)

        # Warmup
        if ni <= nw:
            xi = [0, nw]
            self.accumulate = max(1, np.interp(ni, xi, [1, self.nbs / self.opt.batch_size]).round())
            for j, x in enumerate(self.optimizer.param_groups):
                x['lr'] = np.interp(ni, xi, [self.hyp['warmup_bias_lr'] if j == 0 else 0.0, x['initial_lr'] * self.lf(self.current_epoch)])
                if 'momentum' in x:
                    x['momentum'] = np.interp(ni, xi, [self.hyp['warmup_momentum'], self.hyp['momentum']])

        # Multi-scale
        if self.opt.multi_scale:
            sz = random.randrange(self.opt.imgsz * 0.5, self.opt.imgsz * 1.5 + self.gs) // self.gs * self.gs
            sf = sz / max(imgs.shape[2:])
            if sf != 1:
                ns = [math.ceil(x * sf / self.gs) * self.gs for x in imgs.shape[2:]]
                imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

        # Forward and loss
        if self.detr:
            _targets = self._prepare_detr_targets(imgs, targets)
            pred = self.model(imgs, batch=_targets, detr=True)
            loss, loss_items = self._compute_detr_loss(pred, _targets)
        else:
            pred = self.model(imgs)
            loss, loss_items = self.compute_loss(pred, targets.to(self.device))

        self.mloss = (self.mloss * batch_idx + loss_items) / (batch_idx + 1)
        self._log_training_metrics(loss, batch_idx)

        # Backward
        self.scaler.scale(loss).backward()
        if ni - self.last_opt_step >= self.accumulate:
            self.scaler.unscale_(self.optimizer)
            self.clip_gradients(self.optimizer, gradient_clip_val=10.0, gradient_clip_algorithm="norm")
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            if RANK in {-1, 0}:
                self.ema.update(self.model)
            self.last_opt_step = ni

        return loss

    def _prepare_detr_targets(self, imgs, targets):
        bs = len(imgs)
        batch_idx = targets[:, 0]
        gt_groups = [(batch_idx == i).sum().item() for i in range(bs)]
        return {
            "cls": targets[:, 1].to(self.device, dtype=torch.long),
            "bboxes": targets[:, 2:].to(self.device),
            "batch_idx": batch_idx.to(self.device, dtype=torch.long).view(-1),
            "gt_groups": gt_groups,
        }

    def _compute_detr_loss(self, pred, _targets):
        dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = pred
        dn_bboxes, dn_scores = (None, None) if dn_meta is None else torch.split(dec_bboxes, dn_meta["dn_num_split"], dim=2) + torch.split(dec_scores, dn_meta["dn_num_split"], dim=2)
        dec_bboxes = torch.cat([enc_bboxes.unsqueeze(0), dec_bboxes])
        dec_scores = torch.cat([enc_scores.unsqueeze(0), dec_scores])
        loss = self.compute_loss((dec_bboxes, dec_scores), _targets, dn_bboxes=dn_bboxes, dn_scores=dn_scores, dn_meta=dn_meta)
        return sum(loss.values()), torch.as_tensor([loss[k].detach() for k in ["loss_giou", "loss_class", "loss_bbox"]], device=self.device)

    def _log_training_metrics(self, loss, batch_idx):
        self.log('train/loss', loss, on_epoch=True, on_step=False, prog_bar=True, logger=True, sync_dist=self.dist)
        for idx, x in enumerate(['box_loss', 'diff_loss', 'cls_loss']):
            self.log(f'train/{x}', self.mloss[idx], on_epoch=True, on_step=True, prog_bar=True, logger=True, sync_dist=self.dist)

    def on_train_epoch_end(self):
        self.lr = [x['lr'] for x in self.optimizer.param_groups]
        self.scheduler.step()
        if RANK in {-1, 0}:
            self.ema.update_attr(self.model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])

    def _initialize_validation(self):
        self.cuda = self.device != 'cpu'
        self.val_mloss = torch.zeros(3, device=self.device)
        self.dt = Profile(), Profile(), Profile()
        self.stats, self.jdict, self.val_idx = [], [], []
        self.confusion_matrix = ConfusionMatrix(nc=self.model.nc)
        self.seen = 0
        self.iouv = torch.linspace(0.5, 0.95, 10, device=self.device)
        self.niou = self.iouv.numel()
        w = os.path.join(self.opt.save_dir, 'weights')
        os.makedirs(w, exist_ok=True)
        self.last, self.best = os.path.join(w, 'last.pt'), os.path.join(w, 'best.pt')

    def on_validation_epoch_start(self):
        self._initialize_validation()
        LOGGER.info(f"\n*** Validating epoch {self.current_epoch}***\n")

    def validation_step(self, batch, batch_idx):
        imgs, targets, paths, shapes = batch
        imgs = imgs.to(self.device, non_blocking=True)
        imgs = imgs.half() if next(self.model.parameters()).dtype == torch.float16 else imgs.float()
        imgs /= 255

        nb, _, height, width = imgs.shape
        with self.dt[1]:
            if self.detr:
                _targets = self._prepare_detr_targets(imgs, targets.to(self.device))
                preds = self.model(imgs, batch=_targets, detr=True)
            else:
                preds, train_out = self.model(imgs) if self.compute_loss else (self.model(imgs, augment=self.opt.augment), None)

        # Compute loss and predictions
        if self.detr:
            loss, loss_items = self._compute_detr_loss(preds[1], _targets)
            self.val_mloss = (self.val_mloss * batch_idx + loss_items) / (batch_idx + 1)
            preds = self._process_detr_predictions(preds[0], nb)
        else:
            if self.compute_loss:
                preds = preds[1]
                self.val_mloss += self.compute_loss(train_out, targets)[1]
            else:
                preds = preds[0][1]
            preds = non_max_suppression(preds, self.opt.conf_thres, self.opt.iou_thres, labels=[], multi_label=True, agnostic=self.opt.single_cls, max_det=self.opt.max_det)

        self._evaluate_predictions(imgs, preds, targets, paths, shapes, batch_idx, nb, height, width)

    def _process_detr_predictions(self, pred, bs):
        bboxes, scores = pred.split((4, pred.shape[-1] - 4), dim=-1)
        outputs = [torch.zeros((0, 6), device=bboxes.device)] * bs
        topk_values, topk_indexes = torch.topk(scores.reshape(scores.shape[0], -1), self.opt.max_det, dim=1)
        topk_boxes = topk_indexes // scores.shape[2]
        lbs = topk_indexes % scores.shape[2]
        bboxes = torch.gather(bboxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
        scores = topk_values

        for i, bbox in enumerate(bboxes):
            bbox = xywh2xyxy(bbox)
            score = scores[i]
            cls = lbs[i]
            pred = torch.cat([bbox, score[..., None], cls[..., None]], dim=-1)
            pred = pred[score.argsort(descending=True)]
            outputs[i] = pred
        return outputs

    def _evaluate_predictions(self, imgs, preds, targets, paths, shapes, batch_idx, nb, height, width):
        for si, pred in enumerate(preds):
            labels = targets[targets[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0]
            path, shape = Path(paths[si]), shapes[si][0]
            correct = torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device)
            self.seen += 1

            if npr == 0:
                if nl:
                    self.stats.append((correct, *torch.zeros((2, 0), device=self.device), labels[:, 0]))
                continue

            if self.opt.single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_boxes(imgs[si].shape[1:], predn[:, :4], shape, shapes[si][1])

            if nl:
                tbox = xywh2xyxy(labels[:, 1:5]) * (torch.tensor(imgs[si].shape[1:], device=self.device)[[1, 0, 1, 0]] if self.detr else 1)
                scale_boxes(imgs[si].shape[1:], tbox, shape, shapes[si][1])
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)
                correct = process_batch(predn, labelsn, self.iouv)

            self.stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))
            if self.opt.save_txt:
                save_one_txt(predn, self.opt.save_conf, shape, file=self.opt.save_dir / 'labels' / f'{path.stem}.txt')

        self.val_idx.append(batch_idx)

    def on_validation_epoch_end(self):
        self.stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*self.stats)]
        metrics = self._compute_metrics()
        loss = (self.val_mloss.cpu() / len(self.val_idx)).tolist()
        fi = fitness(np.array([metrics['mp'], metrics['mr'], metrics['map50'], metrics['map']]).reshape(1, -1))[0]

        for idx, name in enumerate(['box_loss', 'diff_loss', 'cls_loss']):
            self.log(f"val/{name}", loss[idx], on_epoch=True, on_step=False, prog_bar=True, logger=True, sync_dist=self.dist)

        if fi > self.best_fitness:
            self.best_fitness = fi
        self.save_model(fi)

        for name, value in metrics.items():
            self.log(f'metrics/{name}', value, on_epoch=True, on_step=False, prog_bar=True, logger=True, sync_dist=self.dist)

        total_loss = sum(loss) / len(loss)
        self.log('val/loss', total_loss, on_epoch=True, on_step=False, prog_bar=True, logger=True, sync_dist=self.dist)
        LOGGER.info(('%22s' + '%11.3g' * 4) % ('all', metrics['mp'], metrics['mr'], metrics['map50'], metrics['map']))

        return total_loss

    def _compute_metrics(self):
        metrics = {'mp': 0.0, 'mr': 0.0, 'map50': 0.0, 'map': 0.0}
        if len(self.stats) and self.stats[0].any():
            tp, fp, p, r, f1, ap, ap_class = ap_per_class(*self.stats, plot=self.opt.plots, save_dir=self.opt.save_dir, names=self.names)
            ap50, ap = ap[:, 0], ap.mean(1)
            metrics.update({'mp': p.mean(), 'mr': r.mean(), 'map50': ap50.mean(), 'map': ap.mean()})
            self.maps = np.zeros(self.model.nc) + metrics['map']
            for i, c in enumerate(ap_class):
                self.maps[c] = ap[i]
        return metrics

    def save_model(self, fi):
        final_epoch = self.current_epoch == self.trainer.max_epochs - 1
        if not (self.opt.nosave or (final_epoch and not self.opt.evolve)):
            ckpt = {
                'epoch': self.current_epoch,
                'best_fitness': self.best_fitness,
                'model': deepcopy(de_parallel(self.model)).half(),
                'ema': deepcopy(self.ema.ema).half(),
                'updates': self.ema.updates,
                'optimizer': self.optimizer.state_dict(),
                'opt': vars(self.opt),
                'date': datetime.now().isoformat()
            }
            torch.save(ckpt, self.last)
            if self.best_fitness == fi:
                torch.save(ckpt, self.best)
            if self.opt.save_period > 0 and self.current_epoch % self.opt.save_period == 0:
                torch.save(ckpt, self.opt.save_dir / 'weights' / f'epoch{self.current_epoch}.pt')

def prepare_batch(si, batch, device):
    idx = batch["batch_idx"] == si
    imgsz = batch["img"][si].shape[1:]
    cls = batch['cls'][batch["batch_idx"] == si]
    bbox = batch["bboxes"][idx]
    ori_shape = batch["ori_shape"]
    ratio_pad = batch["ratio_pad"]
    
    if len(cls):
        bbox = xywh2xyxy(bbox) * torch.tensor(imgsz, device=device)[[1, 0, 1, 0]]
        scale_boxes(imgsz, bbox, ori_shape, ratio_pad=ratio_pad)
    return dict(cls=cls, bbox=bbox, ori_shape=ori_shape, imgsz=imgsz, ratio_pad=ratio_pad)

def process_batch(detections, labels, iouv):
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)

def save_one_txt(predn, save_conf, shape, file):
    gn = torch.tensor(shape)[[1, 0, 1, 0]]
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')