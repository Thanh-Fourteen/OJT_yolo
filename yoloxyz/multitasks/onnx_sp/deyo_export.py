import os
import sys
import time
import onnx
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
    # model.model[-1].export = not opt.grid
    # model.model[-2].export = not opt.grid

    y = model(img) 
    print("\nStarting ONNX export with onnx %s..." % onnx.__version__)
    f = opt.weights.replace(".pt", ".onnx")  # filename
    model.eval()

    dynamic_axes = None

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default=0, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--weights', type=str, default="C:/Users/admin/Desktop/weights/minicoco/best.pt", help='initial weights path')
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
    print(f"model save as {f}")
