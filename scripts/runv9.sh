CUDA_VISIBLE_DEVICES=0,1 python OJT_yolo/yoloxyz/train_ptln.py \
    --weights minicoco/yolov9-c.pt \
    --cfg OJT_yolo/yoloxyz/cfg/architecture/deyo.yaml \
    --hyp OJT_yolo/yoloxyz/cfg/hyp/hyp.deyo.yaml \
    --data OJT_yolo/yoloxyz/cfg/data/coco_dataset.yaml \
    --name finetune_v9 \
    --batch 16 \
    --epochs 1000 \
    --imgsz 640 \
    --device 0 1 \
    --workers 8 \
    --close-mosaic 15