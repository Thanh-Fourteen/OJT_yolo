import os
import sys
import csv
import yaml
import time
import torch
import argparse
import numpy as np
from tqdm import tqdm

import onnxruntime as ort
from yolov9.utils.torch_utils import select_device
from yolov9.utils.metrics import ConfusionMatrix, box_iou
from yolov9.utils.general import TQDM_BAR_FORMAT, Profile, xywh2xyxy, scale_boxes

sys.path.append(os.path.join(os.getcwd(), 'yoloxyz'))
from torchkit.metrics import DetMetrics
from data.datasets import get_val_dataloader
from multitasks.onnx_sp.deyo_export import get_model

def onnx_model(model_path):
    model = ort.InferenceSession(
        model_path,
        providers=[
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
    )
    return model

def preprocess(batch, device):
    """Preprocesses batch of images for YOLO training."""
    batch["img"] = batch["img"].to(device, non_blocking=True)
    batch["img"] = batch["img"].float() / 255
    for k in ["batch_idx", "cls", "bboxes"]:
        batch[k] = batch[k].to(device)

    return batch

def postprocess(preds):
    """Apply Non-maximum suppression to prediction outputs."""
    num_select = 300
    bs, _, nd = preds[0].shape
    bboxes, scores = preds[0].split((4, nd - 4), dim=-1)
    # bboxes *= args.imgsz
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
        # idx = score > args.conf
        pred = torch.cat([bbox, score[..., None], cls[..., None]], dim=-1)  # filter
        # Sort by confidence to correctly get internal metrics
        pred = pred[score.argsort(descending=True)]
        outputs[i] = pred  # [idx]
    return outputs

def _prepare_batch(si, batch, device):
    """Prepares a batch of images and annotations for validation."""
    idx = batch["batch_idx"] == si
    cls = batch["cls"][idx].squeeze(-1)
    bbox = batch["bboxes"][idx]
    ori_shape = batch["ori_shape"][si]
    imgsz = batch["img"].shape[2:]
    ratio_pad = batch["ratio_pad"][si]
    if len(cls):
        bbox = xywh2xyxy(bbox) * torch.tensor(imgsz, device=device)[[1, 0, 1, 0]]  # target boxes
        scale_boxes(imgsz, bbox, ori_shape, ratio_pad=ratio_pad)  # native-space labels
    return dict(cls=cls, bbox=bbox, ori_shape=ori_shape, imgsz=imgsz, ratio_pad=ratio_pad)

def _prepare_pred(pred, pbatch):
    """Prepares a batch of images and annotations for validation."""
    predn = pred.clone()
    scale_boxes(
        pbatch["imgsz"], predn[:, :4], pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"]
    )  # native-space pred
    return predn.float()

def match_predictions(pred_classes, true_classes, iou, iouv, use_scipy=False):
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
    correct = np.zeros((pred_classes.shape[0], iouv.shape[0])).astype(bool)
    # LxD matrix where L - labels (rows), D - detections (columns)
    correct_class = true_classes[:, None] == pred_classes
    iou = iou * correct_class  # zero out the wrong classes
    iou = iou.cpu().numpy()
    for i, threshold in enumerate(iouv.cpu().tolist()):
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



def _process_batch(detections, gt_bboxes, gt_cls, iouv):
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
    return match_predictions(detections[:, 5], gt_cls, iou, iouv)
    

def update_metrics(preds, batch, device, seen, stats, confusion_matrix):
    """Metrics."""
    for si, pred in enumerate(preds):
        iouv = torch.linspace(0.5, 0.95, 10, device=device)
        niou = iouv.numel()
        seen += 1
        npr = len(pred)
        stat = dict(
            conf=torch.zeros(0, device=device),
            pred_cls=torch.zeros(0, device=device),
            tp=torch.zeros(npr, niou, dtype=torch.bool, device=device),
        )
        pbatch = _prepare_batch(si, batch, device)
        cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
        nl = len(cls)
        stat["target_cls"] = cls
        if npr == 0:
            if nl:
                for k in stats.keys():
                    stats[k].append(stat[k])
                # TODO: obb has not supported confusion_matrix yet.
                if opt.plots and opt.task != "obb":
                    confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
            continue

        # Predictions
        if opt.single_cls:
            pred[:, 5] = 0
        predn = _prepare_pred(pred, pbatch)
        stat["conf"] = predn[:, 4]
        stat["pred_cls"] = predn[:, 5]

        # Evaluate
        if nl:
            stat["tp"] = _process_batch(predn, bbox, cls, iouv)
            # TODO: obb has not supported confusion_matrix yet.
        for k in stats.keys():
            stats[k].append(stat[k])

    return seen

def save_metrics_to_csv(metrics, seen, nt_per_class, data_dict, t, file_path='metrics.csv'):
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Viết header vào file CSV
        header = ["Class", "Images", "Instances", "Precision", "Recall", "mAP50", "mAP50-95", "seconds"]
        writer.writerow(header)

        # Ghi kết quả tổng hợp
        pf = "%12s" + "%11i" * 2 + "%11.3g" * len(metrics.keys) + "%15.3f"  # Định dạng in ra
        row = ["all", seen, nt_per_class.sum(), *metrics.mean_results(), time.time() - t]
        writer.writerow(row)

        # Ghi kết quả cho từng lớp
        pf = "%12s" + "%11i" * 2 + "%11.3g" * len(metrics.keys)
        for i, c in enumerate(metrics.ap_class_index):
            class_result = metrics.class_result(i)
            row = [data_dict["names"][c], seen, nt_per_class[c], *class_result]
            writer.writerow(row)

    print(f"Metrics saved to {file_path}")


def main(opt):
    #dataset
    data_dict = None
    with open(opt.data, "r") as f:
        data_dict = yaml.safe_load(f)

    device = select_device(opt.device)
    model_pt = get_model(opt.weights, device)
    gs = max(int(model_pt.stride.max() if hasattr(model_pt, "stride") else 32), 32)

    val_path = data_dict['val']
    val_loader = get_val_dataloader(opt, data= data_dict, stride = gs, dataset_path= val_path, batch_size= opt.batch_size, workers = opt.workers)

    model = onnx_model(opt.onnx)

    # validating
    dt = (Profile(), Profile(), Profile(), Profile())
    loss, seen, batch_val = 0, 0, 0
    stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[])
    inp_name = [x.name for x in model.get_inputs()]
    opt_name = [x.name for x in model.get_outputs()]
    metrics = DetMetrics(save_dir=opt.save_dir)
    metrics.names = data_dict["names"]
    confusion_matrix = ConfusionMatrix(nc=data_dict["nc"])

    s = ("%12s" + "%11s" * 6 + "%11s") % ("Class", "Images", "Instances", "Box(P", "R", "mAP50", "mAP50-95)", "seconds")
    pbar = tqdm(val_loader, desc=s, bar_format=TQDM_BAR_FORMAT)
    t = time.time()

    for batch_i, batch in enumerate(pbar):
        with dt[0]:
            batch = preprocess(batch, device)

        # Inference
        with dt[1]:
            tensor = batch["img"].cpu().numpy()
            if (opt.fp == 'fp16'):
                tensor = tensor.astype(np.float16)      # fp16
            tensor = np.expand_dims(tensor, axis=0)
            preds = model.run(opt_name, dict(zip(inp_name, tensor)))
        
        # Loss
        with dt[2]:
            ans_0 = torch.tensor(preds[0]).to(device)
            torch_tensors = [torch.tensor(pred, device=device) for pred in preds[1:5]]
            ans_1 = tuple(torch_tensors) + (None,)
            preds = [ans_0, ans_1]

            loss += model_pt.loss(batch, preds)[1]
        
        # Postprocess
        with dt[3]:
            preds = postprocess(preds)

        # Metrics
        seen = update_metrics(preds, batch, device, seen, stats, confusion_matrix)
    
    stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in stats.items()}
    if len(stats) and stats["tp"].any():
        metrics.process(**stats)
    nt_per_class = np.bincount(
        stats["target_cls"].astype(int), minlength=data_dict["nc"]
    )
    metrics.confusion_matrix = confusion_matrix
    pf = "%12s" + "%11i" * 2 + "%11.3g" * len(metrics.keys) + "%15.3f"  # print format
    print(pf % ("all", seen, nt_per_class.sum(), *metrics.mean_results(), time.time() - t))

    pf = "%12s" + "%11i" * 2 + "%11.3g" * len(metrics.keys)
    for i, c in enumerate(metrics.ap_class_index):
        print(pf % (data_dict["names"][c], seen, nt_per_class[c], *metrics.class_result(i)))
    
    save_metrics_to_csv(metrics, seen, nt_per_class, data_dict, t, opt.save_metric)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp', type=str, default='fp32', help="Define backbone model", choices=['fp16', 'fp32'])
    parser.add_argument('--weights', type=str, default="C:/Users/admin/Desktop/weights/minicoco/best.pt", help='initial weights path')
    parser.add_argument('--onnx', type=str, default='C:/Users/admin/Desktop/weights/minicoco/best32.onnx', help='initial weights path')
    parser.add_argument('--data', type=str, default="D:/FPT/AI/Major6/OJT_yolo/yoloxyz/cfg/data/coco_dataset.yaml", help='data.yaml path')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--batch-size', type=int, default=1, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=320, help='train, val image size (pixels)')
    parser.add_argument('--workers', type=int, default=4, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='image --cache ram/disk')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--augment', type=bool, default=False, help=' (True or False).')

    #validation
    parser.add_argument('--task', type=str, default='detect', help="deyo")
    parser.add_argument('--classes', type=list[int] or int, help="deyo")
    parser.add_argument('--save-dir', type=str, default="C:/Users/admin/Desktop/", help='save path')
    parser.add_argument('--save-metric', type=str, default="C:/Users/admin/Desktop/weights/minicoco/onnx32_metrics.csv", help='save path')

    opt = parser.parse_args()
    main(opt)