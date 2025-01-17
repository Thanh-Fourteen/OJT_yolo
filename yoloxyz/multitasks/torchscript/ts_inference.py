import cv2
import time
import torch
import numpy as np
from pathlib import Path

from yolov9.utils.general import xywh2xyxy, scale_boxes, check_dataset

class YoloV9DeyoTensor:
    def __init__(self, model_path):
        self.load_model(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

    def load_model(self, model_path):
        self.model = torch.jit.load(model_path)

    def preprocess(self, im:np.array, fp, new_shape=(640, 640), color=(114, 114, 114), scaleup=True) -> list:
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]

        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=color
                                )  # add border

        im = im.transpose((2, 0, 1))
        im = np.expand_dims(im, 0)
        if (fp == 32):
            im = torch.from_numpy(im).float()
        elif (fp == 16):
            im = torch.from_numpy(im).half()  # Half precision float16
        else:
            print("Error fp")
            exit()
        im /= 255

        return im, r, (dw, dh)

    def postprocess(self, preds, img, orig_img, conf = 0.4):
        if isinstance(preds, np.ndarray):
            preds = torch.from_numpy(preds)
        nd = preds.shape[-1]
        bboxes, scores = preds.split((4, nd - 4), dim=-1)

        results = []
        for i, bbox in enumerate(bboxes):  # (300, 4)
            bbox = xywh2xyxy(bbox)
            bbox = scale_boxes(img.shape[2:], bbox, orig_img.shape)
            score, cls = scores[i].max(-1, keepdim=True)  # (300, 1)
            idx = score.squeeze(-1) > conf  # (300, )
            pred = torch.cat([bbox, score, cls], dim=-1)[idx]
            results.extend(pred.cpu().numpy())

        return results

    def draw_predictions(self, yaml_path, orig_img, predictions, color=(114, 114, 114), thickness=2, 
                         font_scale=0.5, font_color=(0, 255, 0), font_thickness=1):
        with open(yaml_path, 'r') as file:
            data_dict = {} or check_dataset(yaml_path) 
            class_names = data_dict.get('names', {})

        for box in predictions:
            x_min, y_min, x_max, y_max, score, cls_id = box[:6]
            score = round(score, 2)
            x_min, y_min, x_max, y_max, cls_id = map(int, [x_min, y_min, x_max, y_max, cls_id])
            label = f"{class_names.get(cls_id, 'Unknown')}: {score:.2f}"

            cv2.rectangle(orig_img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
            cv2.putText(orig_img, label, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness)
    
        return orig_img


    def inference(self, img, yaml_path, fp, save = True, model_inpsize = (640, 640), save_path = Path("predict.jpg")):
        tensor, ratio, dwdh = self.preprocess(img, new_shape = model_inpsize, fp = fp)
        tensor = tensor.to(self.device)
        
        # model prediction
        s0 = time.time()
        outputs = self.model(tensor)[0]
        s1 = time.time()
        print("Inference time: ", round(s1 - s0, 4))
        predictions = self.postprocess(outputs, tensor, img)
        annotated_img = self.draw_predictions(yaml_path, img, predictions)

        if save:
            cv2.imwrite(save_path, annotated_img)
            print(f"Image saved as {save_path}")

        return annotated_img


if __name__ == '__main__':

    fp = 32
    save_path = r"C:\Users\admin\Desktop\output" + str(fp) + ".jpg"
    model_path = "C:/Users/admin/Desktop/weights/minicoco/best_sp.pt"
    img_path = r"C:\Users\admin\Desktop\minicoco\images\train\000000018380.jpg"
    yaml_path = r"D:\FPT\AI\Major6\OJT_yolo\yoloxyz\cfg\data\coco_dataset.yaml"

    model = YoloV9DeyoTensor(model_path)
    image = cv2.imread(img_path)
    result = model.inference(image, yaml_path, fp, save_path= save_path)
