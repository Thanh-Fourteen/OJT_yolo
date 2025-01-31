python yoloxyz/train_ptln.py \
    --basemodel 'v9' \
    --weights C:/Users/admin/Desktop/datasets/yolov9-c.pt \
    --cfg D:/FPT/AI/Major6/OJT_yolo/yoloxyz/cfg/architecture/yolov9-s-rtdetr.yaml \
    --hyp D:/FPT/AI/Major6/OJT_yolo/yoloxyz/cfg/hyp/hyp.deyo.yaml \
    --data D:/FPT/AI/Major6/OJT_yolo/yoloxyz/cfg/data/ptlnab.yaml \
    --do-train \
    --name test_deyo_yolov9 \
    --batch 2 \
    --epochs 10 \
    --imgsz 320 \
    --device 0 \
    --workers 2 \
    --close-mosaic 15 \
    --min-items 0