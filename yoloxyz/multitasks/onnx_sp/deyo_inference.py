import os
import sys
import cv2
import torch
import numpy as np
from pathlib import Path
import onnxruntime as ort
from yolov9.utils.general import xywh2xyxy, scale_boxes, check_dataset

sys.path.append(os.path.join(os.getcwd(), 'yoloxyz'))


class YoloV9Deyo:
    def __init__(self, model_path):
        self.load_model(model_path)

    def load_model(self, model_path):
        self.model = ort.InferenceSession(
            model_path,
            providers=[
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        )
        self.inp_name = [x.name for x in self.model.get_inputs()]
        self.opt_name = [x.name for x in self.model.get_outputs()]
        _, _, h, w = self.model.get_inputs()[0].shape
        self.model_inpsize = (w, h)

    def preprocess(self, im:np.array, new_shape=(640, 640), color=(114, 114, 114), scaleup=True) -> list:
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
        im = np.ascontiguousarray(im, dtype=np.float32)
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
            bbox = scale_boxes(img.shape[3:], bbox, orig_img.shape)
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


    def inference(self, img, yaml_path, save = True, save_path = Path("predict.jpd")):
        tensor, ratio, dwdh = self.preprocess(img, self.model_inpsize)
        tensor = np.expand_dims(tensor, axis=0)
        # model prediction
        outputs = self.model.run(self.opt_name, dict(zip(self.inp_name, tensor)))[0]

        predictions = self.postprocess(outputs, tensor, img)
        annotated_img = self.draw_predictions(yaml_path, img, predictions)

        if save:
            cv2.imwrite(save_path, annotated_img)
            print(f"Image saved as {save_path}")

        return annotated_img


if __name__ == '__main__':
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    save_path = r"C:\Users\admin\Desktop\output.jpg"
    model_path = "C:/Users/admin/Desktop/weights/Abjadyolov9/best.onnx"
    img_path = r"C:\Users\admin\Desktop\deyo\images\train\E3_png.rf.566982fdaf2e3bb4030b0911432f25c9.jpg"
    yaml_path = r"D:\FPT\AI\Major6\OJT_yolo\yoloxyz\cfg\data\abjad.yaml"

    model = YoloV9Deyo(model_path)
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = model.inference(image, yaml_path, save_path= save_path)
