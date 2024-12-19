import argparse

def training_arguments(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='C:/Users/admin/Desktop/datasets/yolov9-c.pt', help='initial weights path')
    # parser.add_argument('--weights', type=str, default="D:/FPT/AI/Major6/OJT_yolo/yoloxyz/backbones/yolov9/runs/train/yolov10-c27/weights/best.pt", help='initial weights path')
    parser.add_argument('--cfg', type=str, default='D:/FPT/AI/Major6/OJT_yolo/yolo_ptln/cfg/architecture/deyo.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default="D:/FPT/AI/Major6/OJT_yolo/yolo_ptln/cfg/data/abjad.yaml", help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='D:/FPT/AI/Major6/OJT_yolo/yolo_ptln/cfg/hyp/hyp.deyo.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=10, help='total training epochs')
    parser.add_argument('--batch-size', type=int, default=2, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=320, help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable AutoAnchor')
    parser.add_argument('--noplots', action='store_true', help='save no plot files')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='image --cache ram/disk')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW', 'LION'], default='AdamW', help='optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
    parser.add_argument('--flat-cos-lr', action='store_true', help='flat cosine LR scheduler')
    parser.add_argument('--fixed-lr', action='store_true', help='fixed LR scheduler')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')
    parser.add_argument('--min-items', type=int, default=0, help='Experimental')
    parser.add_argument('--close-mosaic', type=int, default=24, help='Experimental')
    parser.add_argument('--iou-loss', type=str, default='EIoU', help= 'use iou_loss (EIoU default) for loss bounding box')
    parser.add_argument('--kpt-label', type=int, default=0, help='number of keypoints')
    # parser.add_argument('--freeze', type=str, default=None, help='Freeze layers: 0-10: backbone')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
    parser.add_argument('--multilosses', action='store_true', help="Multihead loss")
    parser.add_argument('--detect-layer', type=str, default='DualDDetect', help="Calculated loss")
    parser.add_argument('--warmup', action='store_true', help="Warmup epochs")

    #validation
    parser.add_argument('--task', type=str, default='detect', help="deyo")
    parser.add_argument('--classes', type=list[int] or int, help="deyo")
    parser.add_argument('--save_json', type=bool, default=False, help='deyo')
    parser.add_argument('--save_txt', type=bool, default=False, help='deyo')

    # Logger arguments
    parser.add_argument('--entity', default=None, help='Entity')
    parser.add_argument('--upload_dataset', nargs='?', const=True, default=False, help='Upload data, "val" option')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval')
    parser.add_argument('--artifact_alias', type=str, default='latest', help='Version of dataset artifact to use')
    parser.add_argument('--accelerator', default='auto', help='cpu, gpu, tpu or auto')

    # Pytorch lightning
    parser.add_argument('--log-steps', type=int, default=1, help='Loging step')
    parser.add_argument('--do-train', action='store_true', help='Do training')
    parser.add_argument('--do-eval', action='store_true', help='Do eval')

    # Deyo transforms
    parser.add_argument('--mosaic', type=float, default=1.0, help='yolov8 transforms')
    parser.add_argument('--mixup', type=float, default=0.0, help='yolov8 transforms')
    parser.add_argument('--copy_paste', type=float, default=0.0, help='yolov8 transforms')
    parser.add_argument('--degrees', type=float, default=0.0, help='yolov8 transforms')
    parser.add_argument('--translate', type=float, default=0.1, help='yolov8 transforms')
    parser.add_argument('--scale', type=float, default=0.5, help='yolov8 transforms')
    parser.add_argument('--shear', type=float, default=0.0, help='yolov8 transforms')
    parser.add_argument('--perspective', type=float, default=0.0, help='yolov8 transforms')
    parser.add_argument('--fliplr', type=float, default=0.5, help='yolov8 transforms')
    parser.add_argument('--flipud', type=float, default=0.0, help='yolov8 transforms')
    parser.add_argument('--hsv_s', type=float, default=0.7, help='yolov8 transforms')
    parser.add_argument('--hsv_h', type=float, default=0.015, help='yolov8 transforms')
    parser.add_argument('--hsv_v', type=float, default=0.4, help='yolov8 transforms')
    parser.add_argument('--save_hybrid', type=bool, default=False, help='yolov8 transforms')

    return parser.parse_known_args()[0] if known else parser.parse_args()