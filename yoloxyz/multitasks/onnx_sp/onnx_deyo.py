import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'yoloxyz'))

import time
import onnx
import torch
import argparse
import torchvision
import torch.nn as nn
import numpy as np 
import onnxruntime as ort 

from data.augment import LetterBox
from torchkit.tasks import attempt_load_weights
from yolov9.models import common
from yolov9.utils.torch_utils import select_device
from yolov9.utils.general import LOGGER, Profile, check_img_size
from yolov9.utils.dataloaders import LoadImages

def get_model(weights, device, fuse = True):
    w = str(weights[0] if isinstance(weights, list) else weights)
    
    model = attempt_load_weights(
            weights if isinstance(weights, list) else w, device=device, inplace=True, fuse=fuse
        )
    model.float()

    for p in model.parameters():
        p.requires_grad = False
    return model

def export_onnx(opt):
    '''export onnx
    '''
    device = select_device(opt.device)
    model = get_model(opt.weights, device)
    t = time.time()

    # Checks
    gs = max(int(model.stride.max() if hasattr(model, "stride") else 32), 32)
    opt.img_size = [check_img_size(x, gs, floor=gs * 2) for x in opt.img_size] 
    img = torch.zeros(opt.batch_size, 3, *opt.img_size).to(device)

    # set Detect() layer grid export
    model.model[-1].export = not opt.grid
    model.model[-2].export = not opt.grid

    y = model(img) 
    print("\nStarting ONNX export with onnx %s..." % onnx.__version__)
    f = opt.weights.replace(".pt", ".onnx")  # filename
    model.eval()

    # output = y
    # print(f"Output: {output.shape}")
    # print(opt.dynamic)
    # with open(r"C:\Users\admin\Desktop\test.txt", "w") as f:
    #     for i in range(output.shape[0]):
    #         f.write(f"batch {i}:\n")  # Ghi thông tin slice
    #         for idx_row, row in enumerate(output[i]):
    #             f.write(f"{idx_row} row {row}:\n") 
    #             for idx_val, val in enumerate(row):
    #                 f.write(f"{idx_val} val {val}:\n") 
    #         f.write("\n")
            
    # exit()

    dynamic_axes = None
    # if opt.dynamic:
    #     dynamic_axes = {
    #         "images": {0: "batch", 2: "height", 3: "width"},  # size(1,3,640,640)
    #     }
    
    # if opt.dynamic_batch:
    #     opt.batch_size = "batch"
    #     dynamic_axes = {"images": {0: "batch"}}
    #     output_axes = {
    #         "output": {0: "batch"} # "output" là tên của output tensor
    #     }
    #     dynamic_axes.update(output_axes)

    # if opt.grid:
    #     model.model[-1].concat = True

    torch.onnx.export(
        model,
        img,
        f,
        verbose=False,
        opset_version=16,
        input_names=["images"],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
    )

    # Checks
    onnx_model = onnx.load(f)  # load onnx model
    onnx.checker.check_model(onnx_model)

    output = [node for node in onnx_model.graph.output]
    print("Outputs: ", output)
    onnx.save(onnx_model, f)
    print("ONNX export success, saved as %s" % f)

    # Finish
    print(
        "\nExport complete (%.2fs). Visualize with https://github.com/lutzroeder/netron."
        % (time.time() - t)
    )
    return f

def pre_transform(im, imgsz):
    letterbox = LetterBox(imgsz)
    return [letterbox(image=x) for x in im]

def preprocess(im, imgsz, device):
    not_tensor = not isinstance(im, torch.Tensor)
    if not_tensor:
        im = np.stack(pre_transform(im, imgsz))
        im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
        im = np.ascontiguousarray(im)  # contiguous
        im = torch.from_numpy(im)

    im = im.to(device)
    im = im.float()  # uint8 to fp16/32
    if not_tensor:
        im /= 255  # 0 - 255 to 0.0 - 1.0
    return im

def inference(f, imgsz, device, vid_stride = 1):
    session = ort.InferenceSession(f)

    input_names = [input_.name for input_ in session.get_inputs()]
    input_shapes = [input_.shape for input_ in session.get_inputs()]
    input_types = [input_.type for input_ in session.get_inputs()]
    print("Input names:", input_names)
    print("Input shapes:", input_shapes)
    print("Input types:", input_types)

    output_names = [output_.name for output_ in session.get_outputs()]
    output_shapes = [output_.shape for output_ in session.get_outputs()]
    output_types = [output_.type for output_ in session.get_outputs()]
    print("Output names:", output_names)
    print("Output shapes:", output_shapes)
    print("Output types:", output_types)

    device = select_device(device)
    source = r"C:\Users\admin\Desktop\deyo\images\train\A1_png.rf.94ef7ce6d60bc1c5a02d92bab8b7329f.jpg"
    dataset = LoadImages(source, img_size=imgsz, vid_stride=vid_stride)

    profilers = (Profile(), Profile(), Profile(), Profile())

    for batch in dataset:
        path, im, im0, cap, s = batch
        path, im0 = [path], [im0]

        # Preprocess
        with profilers[0]:
            im = preprocess(im0, imgsz, device)
            im = im.cpu().numpy()
           
        input_feed = {"images": im}
        output_tensor = session.run(output_names, input_feed)
        print("Output:", output_tensor)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default=0, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--weights', type=str, default='C:/Users/admin/Desktop/weights/best.pt', help='initial weights path')
    parser.add_argument("--img-size", nargs="+", type=int, default=[320, 320], help="image size")
    parser.add_argument("--imgsz", type=int, default=320, help="image size")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--grid", action="store_true", help="export Detect() layer grid")
    parser.add_argument("--dynamic", action="store_true", help="dynamic ONNX axes")
    parser.add_argument("--dynamic-batch", action="store_true", help="dynamic batch onnx for tensorrt and onnx-runtime",)
    parser.add_argument("--end2end", action="store_true", help="export end2end onnx")
    parser.add_argument("--trt", action="store_true", help="True for tensorrt, false for onnx-runtime")
    
    opt = parser.parse_args()
    f = export_onnx(opt)
    # f = r"C:\Users\admin\Desktop\weights\best.onnx"
    inference(f, opt.imgsz, opt.device)
