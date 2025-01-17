import os
import sys
import time
import torch
import argparse
sys.path.append(os.path.join(os.getcwd(), 'yoloxyz'))

from torchkit.tasks import attempt_load_weights
from yolov9.utils.torch_utils import select_device
from yolov9.utils.general import check_img_size

def get_model(weights, device, fuse = True):
    w = str(weights[0] if isinstance(weights, list) else weights)
    
    model = attempt_load_weights(
            weights if isinstance(weights, list) else w, device=device, inplace=True, fuse=fuse
        )
    model.float()

    for p in model.parameters():
        p.requires_grad = False
    return model

class WrapperModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        y = self.model(x)  
        y0, y1 = y
        if y1[-1] is not None:
            return y0, *y1 
        else:
           return y0, *y1[:-1]
        

def export_ts(opt):
    '''export onnx
    '''
    device = select_device(opt.device)
    model = get_model(opt.weights, device)
    t = time.time()

    # Checks
    gs = max(int(model.stride.max() if hasattr(model, "stride") else 32), 32)
    opt.img_size = [check_img_size(x, gs, floor=gs * 2) for x in opt.img_size] 
    img = torch.zeros(opt.batch_size, 3, *opt.img_size).to(device)

    model = WrapperModel(model)
    y = model(img)


    f = opt.weights.replace(".pt", "_sp.pt")  # filename
    model.eval()

    traced_script_module = torch.jit.trace(model, img)

    # Checks
    traced_output = traced_script_module(img)
    print(len(traced_output))

    traced_script_module.save(f)
    print("TorchScript export success, saved as %s" % f)
    return f

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default=0, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--weights', type=str, default="C:/Users/admin/Desktop/weights/minicoco/best.pt", help='initial weights path')
    parser.add_argument("--img-size", nargs="+", type=int, default=[640, 640], help="image size")
    parser.add_argument("--imgsz", type=int, default=640, help="image size")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--grid", action="store_true", help="export Detect() layer grid")
    parser.add_argument("--dynamic", action="store_true", help="dynamic ONNX axes")
    parser.add_argument("--dynamic-batch", action="store_true", help="dynamic batch onnx for tensorrt and onnx-runtime",)
    parser.add_argument("--end2end", action="store_true", help="export end2end onnx")
    parser.add_argument("--trt", action="store_true", help="True for tensorrt, false for onnx-runtime")
    
    opt = parser.parse_args()
    f = export_ts(opt)
    print(f"model save as {f}")
