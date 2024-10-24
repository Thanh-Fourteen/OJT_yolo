import sys
sys.path.append("D:/FPT/AI/Major6/OJT_yolo/yoloxyz")

import pytorch_lightning as pl
import get_model as gs
from backbones.yolov9.utils.metrics import ap_per_class, box_iou
import random
import math
import numpy as np
import copy
from pathlib import Path

import os
import torch.nn as nn
import torch
from arguments import training_arguments
from backbones.yolov9.utils.general import scale_boxes, xywh2xyxy, non_max_suppression, Profile

opt = training_arguments(True)

RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

def get_average_tensor(tensor_list):
    # Check if the tensors have the same shape
    if not all(tensor.shape == tensor_list[0].shape for tensor in tensor_list):
        raise ValueError("Tensors in the list must have the same shape.")

    # Sum the tensors element-wise
    summed_tensor = torch.stack(tensor_list).sum(dim=0)

    # Calculate the average by dividing by the number of tensors
    average_tensor = summed_tensor / len(tensor_list)

    return average_tensor

def process_batch(detections, labels, iouv):
    """
    Return correct prediction matrix
    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)

class YOLOv9LightningModule(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super(YOLOv9LightningModule, self).__init__()
        self.loss_fn, self.gs, self.imgsz, self.model_device, self.amp, self.yolo_model, self.data_dict, self.ema = gs.get_model(opt)
        self.lr = lr
        self.train_outputs = []
        self.val_outputs = []

        self.seen = 0

        self.stats = []
        self.dt = Profile(), Profile(), Profile()
        self.iouv = torch.linspace(0.5, 0.95, 10, device=self.model_device)  # IOU thresholds for evaluation
        self.niou = self.iouv.numel()
        self.cuda = self.model_device.type != 'cpu'

    def forward(self, x):
        return self.yolo_model(x)

    def training_step(self, batch, batch_idx):
        self.mloss = torch.zeros(3, device=self.model_device)
        loss, acc, loss_items = self._common_step(batch, batch_idx)
        
        if batch_idx == 0:
            self.mloss = loss_items  # Khởi tạo mloss cho epoch mới
        else:
            self.mloss = (self.mloss * batch_idx + loss_items) / (batch_idx + 1)

        self.train_outputs.append({"box_loss": self.mloss[0].item(), "obj_loss": self.mloss[1].item(), "cls_loss": self.mloss[2].item()})
        return loss

    # def validation_step(self, batch, batch_idx):
    #     loss, acc, loss_items = self._common_step(batch, batch_idx)
    #     self.val_outputs.append({"loss": loss, "acc": acc})
    #     return loss
    def validation_step(self, batch, batch_idx):
        # Perform common validation step (e.g., forward pass, loss calculation)
        loss, acc, loss_items = self._common_step(batch, batch_idx)

        imgs, targets, paths, shapes = batch
        # model_eval = self.yolo_model
        model_eval = copy.deepcopy(self.yolo_model)
        # training = model_eval is not None
        # half=True
        # if training: 
        #     half &= self.model_device.type != 'cpu'
        #     model_eval.half() if half else model_eval.float() 
        # else: 
        #     print("Done't have model")
        #     return
        
        model_eval.eval()  
        self.names = model_eval.names if hasattr(model_eval, 'names') else model_eval.module.names  # get class names
        if isinstance(self.names, (list, tuple)):  # old format
            self.names = dict(enumerate(self.names))

        with self.dt[0]:
            if self.cuda:
                imgs = imgs.to(self.model_device, non_blocking=True).float()
                targets = targets.to(self.model_device)
            # imgs = imgs.half() if half else imgs.float()  # uint8 to fp16/32
            imgs /= 255  # 0 - 255 to 0.0 - 1.0
            nb, _, height, width = imgs.shape

        with self.dt[1]:
            with torch.no_grad():
                preds, train_out = model_eval(imgs) if self.loss_fn else (model_eval(imgs, augment=True), None)
        
        # NMS
        save_hybrid = False
        conf_thres = 0.001
        iou_thres = 0.7
        max_det = 300
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=self.model_device)  # to pixels
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        with self.dt[2]:
            preds = non_max_suppression(preds,
                                        conf_thres,
                                        iou_thres,
                                        labels=lb,
                                        multi_label=True,
                                        agnostic=opt.single_cls,
                                        max_det=max_det)

        for si, pred in enumerate(preds):
            self.seen += 1
            labels = targets[targets[:, 0] == si, 1:]  
            nl, npr = labels.shape[0], pred.shape[0] 
            correct = torch.zeros(npr, self.niou, dtype=torch.bool, device=self.model_device) 

            if npr == 0:
                if nl:
                    self.stats.append((correct, *torch.zeros((2, 0), device=self.model_device), labels[:, 0]))
                continue

            if opt.single_cls:
                pred[:, 5] = 0

            predn = pred.clone()  

            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  
                correct = process_batch(predn, labelsn, self.iouv)  
            self.stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  

        return loss


    def on_train_epoch_end(self):
        avg_box_loss = torch.tensor([x["box_loss"] for x in self.train_outputs]).mean()
        avg_obj_loss = torch.tensor([x["obj_loss"] for x in self.train_outputs]).mean()
        avg_cls_loss = torch.tensor([x["cls_loss"] for x in self.train_outputs]).mean()
        mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)

        if isinstance(self.targets, dict):
            tgt_name = list(self.targets.keys())
            total_instances = sum([self.targets[x].shape[0] for x in tgt_name])
        else:
            total_instances = self.targets.shape[0]

        img_size = self.imgsz

        print(f"{'Epoch':>7} {'GPU_mem':>9} {'box_loss':>10} {'obj_loss':>10} {'cls_loss':>10} {'Instances':>12} {'Size':>8}")
        # Print values in the next line, aligned under the headers
        print(f"{self.current_epoch + 1}/{opt.epochs:<5} {mem:>9} {avg_box_loss:>10.5f} {avg_obj_loss:>10.5f} "
            f"{avg_cls_loss:>10.5f} {total_instances:>12} {img_size:>8}")
        
        self.seen = 0
        self.train_outputs.clear()

    # def on_validation_epoch_end(self):
    #     results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)

    #     # avg_val_loss = torch.stack([x["loss"] for x in self.val_outputs]).mean()
    #     # avg_val_acc = torch.stack([x["acc"] for x in self.val_outputs]).mean()

    #     # print(f"/n/nEpoch {self.current_epoch}: Val Loss: {avg_val_loss:.4f}, Val Accuracy: {avg_val_acc:.4f}")
        
    #     self.val_outputs.clear()

    def on_validation_epoch_end(self):
        tp, fp, p, r, f1, mp, mr, map50, ap50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        # Concatenate stats across batches
        self.stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*self.stats)]
        
        if len(self.stats) and self.stats[0].any():
            tp, fp, p, r, f1, ap, ap_class = ap_per_class(*self.stats, names=self.names)
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()

        nc = 1 if opt.single_cls else self.data_dict['nc']
        nt = np.bincount(self.stats[3].astype(int), minlength=nc)  # Count number of targets per class

        # Log results
        # Print the header
        print(f"{'Class':>10} {'Images':>10} {'Instances':>10} {'P':>10} {'R':>10} {'mAP50':>10} {'mAP':>10}")

        # Print the values in the next line, aligned under the headers
        print(f"{'':>10} {'all':>10} {self.seen:>10} {nt.sum():>10} {mp:>10.3f} {mr:>10.3f} {map50:>10.3f} {map:>10.3f}")


        # Clear stats for the next epoch
        self.stats.clear()

    def _common_step(self, batch, batch_idx):
        imgs, self.targets, paths, _ = batch
        imgs = imgs.to(self.model_device, non_blocking=True).float() / 255
        if opt.multi_scale:
            sz = random.randrange(self.imgsz * 0.5, self.imgsz * 1.5 + self.gs) // self.gs * self.gs
            sf = sz / max(imgs.shape[2:])
            if sf != 1:
                ns = [math.ceil(x * sf / self.gs) * self.gs for x in imgs.shape[2:]]
                imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

        with torch.cuda.amp.autocast(self.amp):
            model_outputs = self.yolo_model(imgs)
            _loss, _loss_item = [], []
            for name, _loss_fn in self.loss_fn.items():
                if isinstance(self.targets, dict) and isinstance(model_outputs, dict):
                    _ls, _ls_items = _loss_fn(model_outputs[name], self.targets[name].to(self.model_device))
                else:
                    _ls, _ls_items = _loss_fn(model_outputs, self.targets.to(self.model_device))
                _loss.append(_ls)
                _loss_item.append(_ls_items)

            loss = _loss[0] if len(_loss) == 1 else sum(_loss) / len(_loss)
            loss_items = _loss_item[0] if len(_loss_item) == 1 else get_average_tensor(_loss_item)
            if RANK != -1:
                loss *= WORLD_SIZE
            if opt.quad:
                loss *= 4.

            acc = self._calculate_accuracy(model_outputs, self.targets)
        
        return loss, acc, loss_items

    def _calculate_accuracy(self, outputs, targets):
        # Implement logic to calculate accuracy here
        acc = torch.tensor(0.9)
        return acc

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer