import os
import cv2
import torch
import platform
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from data.augment import LetterBox
from arguments import training_arguments
from torchkit.general import convert_torch2numpy_batch
from backbones.yolov9.utils.torch_utils import select_device
from backbones.yolov9.utils.general import LOGGER, Profile, check_file, colorstr, increment_path, check_yaml, xywh2xyxy, scale_boxes
from backbones.yolov9.utils.dataloaders import LoadImages

from ultralytics.engine.results import Results
# from ultralytics.nn.tasks import attempt_load_weights
from torchkit.tasks import attempt_load_weights

def get_model(opt, fuse = True, fp16 = False):
    device = select_device(opt.device)
    weights = opt.weights
    w = str(weights[0] if isinstance(weights, list) else weights)
    
    model = attempt_load_weights(
            weights if isinstance(weights, list) else w, device=device, inplace=True, fuse=fuse
        )
    model.float()

    for p in model.parameters():
        p.requires_grad = False
    return model

def pre_transform(im, imgsz, model):
    """
    Pre-transform input image before inference.

    opt:
        im (List(np.ndarray)): (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.

    Returns:
        (list): A list of transformed images.
    """
    letterbox = LetterBox(imgsz, stride=model.stride)
    return [letterbox(image=x) for x in im]

def preprocess(im, imgsz, model, device):
    """
    Prepares input image before inference.

    opt:
        im (torch.Tensor | List(np.ndarray)): BCHW for tensor, [(HWC) x B] for list.
    """
    not_tensor = not isinstance(im, torch.Tensor)
    if not_tensor:
        im = np.stack(pre_transform(im, imgsz, model))
        im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
        im = np.ascontiguousarray(im)  # contiguous
        im = torch.from_numpy(im)

    im = im.to(device)
    im = im.float()  # uint8 to fp16/32
    if not_tensor:
        im /= 255  # 0 - 255 to 0.0 - 1.0
    return im

def postprocess(preds, img, orig_imgs, model, batch, conf=0.5):
    """
    Postprocess the raw predictions from the model to generate bounding boxes and confidence scores.

    The method filters detections based on confidence and class if specified in `opt`.

    opt:
        preds (torch.Tensor): Raw predictions from the model.
        img (torch.Tensor): Processed input images.
        orig_imgs (list or torch.Tensor): Original, unprocessed images.

    Returns:
        (list[Results]): A list of Results objects containing the post-processed bounding boxes, confidence scores,
            and class labels.
    """
    
    nd = preds[0].shape[-1]
    bboxes, scores = preds[0].split((4, nd - 4), dim=-1)

    if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
        orig_imgs = convert_torch2numpy_batch(orig_imgs)

    results = []
    for i, bbox in enumerate(bboxes):  # (300, 4)
        bbox = xywh2xyxy(bbox)
        orig_img = orig_imgs[i]
        bbox = scale_boxes(img.shape[2:], bbox, orig_img.shape)
        score, cls = scores[i].max(-1, keepdim=True)  # (300, 1)
        idx = score.squeeze(-1) > conf  # (300, )
        if opt.classes is not None:
            idx = (cls == torch.tensor(opt.classes, device=cls.device)).any(1) & idx
        pred = torch.cat([bbox, score, cls], dim=-1)[idx]  # filter
        '''
        orig_img = orig_imgs[i]
        oh, ow = orig_img.shape[:2]
        pred[..., [0, 2]] *= ow
        pred[..., [1, 3]] *= oh
        '''
        img_path = batch[0][i]
        results.append(Results(orig_img, path=img_path, names=model.names, boxes=pred))
    return results

def write_results(opt, idx, results, batch, dataset, save_dir):
    """Write inference results to a file or directory."""
    plotted_img = None
    p, im, _ = batch
    log_string = ""
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    frame = getattr(dataset, "frame", 0)
    data_path = p
    txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")
    log_string += "%gx%g " % im.shape[2:]  # print string
    result = results[idx]
    log_string += result.verbose()

    if opt.save or opt.show:  # Add bbox to image
        plot_opt = {
            "line_width": opt.line_width,
            "boxes": opt.show_boxes,
            "conf": opt.show_conf,
            "labels": opt.show_labels,
        }
        if not opt.retina_masks:
            plot_opt["im_gpu"] = im[idx]
        plotted_img = result.plot(**plot_opt)

    # Write
    if opt.save_txt:
        result.save_txt(f"{txt_path}.txt", save_conf=opt.save_conf)
    if opt.save_crop:
        result.save_crop(
            save_dir=save_dir / "crops",
            file_name=data_path.stem + ("" if dataset.mode == "image" else f"_{frame}"),
        )

    return log_string, plotted_img

def show(p, plotted_img, windows, batch):
    """Display an image in a window using OpenCV imshow()."""
    im0 = plotted_img
    if platform.system() == "Linux" and p not in windows:
        windows.append(p)
        cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
        cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
    # cv2.imshow(str(p), im0)
    # cv2.waitKey(500 if batch[3].startswith("image") else 1)  # 1 millisecond
    plt.imshow(im0[:, :, ::-1])  # convert BGR to RGB
    plt.title(str(p))  
    plt.show()  

    # plt.pause(0.5 if batch[3].startswith("image") else 0.001)  

def save_preds(save_path, plotted_img, dataset):
    """Save video predictions as mp4 at specified path."""
    im0 = plotted_img
    # Save imgs
    if dataset.mode == "image":
        cv2.imwrite(save_path, im0)
    else:
        print("Not imgs")

def predict(opt, source, vid_stride = 1, save_dir = Path("output")):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model = get_model(opt)
    device = select_device(opt.device)
    model.eval()
    dataset = LoadImages(source, img_size=opt.imgsz, vid_stride=vid_stride)

    seen, windows, b = 0, [], None
    profilers = (Profile(), Profile(), Profile(), Profile())

    for batch in dataset:
        path, im, im0, cap, s = batch
        path, im0 = [path], [im0]

        # Preprocess
        with profilers[0]:
            im = preprocess(im0, opt.imgsz, model, device)

        # Inference
        with profilers[1]:
            preds = model(im)

        # Postprocess
        with profilers[2]:
            results = postprocess(preds, im, im0, model, batch)
        
        n = len(im0)
        for i in range(n):
            seen += 1
            results[i].speed = {
                "preprocess": profilers[0].dt * 1e3 / n,
                "inference": profilers[1].dt * 1e3 / n,
                "postprocess": profilers[2].dt * 1e3 / n,
            }

            p, im0 = path[i], im0[i].copy()
            p = Path(p)
            plotted_img = None
            if opt.verbose or opt.save or opt.save_txt or opt.show:
                log_string, plotted_img = write_results(opt, i, results, (p, im, im0), dataset, save_dir)
                s += log_string
            if opt.save or opt.save_txt:
                results[i].save_dir = save_dir.__str__()
            if opt.show and plotted_img is not None:
                show(p, plotted_img, windows, b)
            if opt.save and plotted_img is not None:
                save_preds(str(save_dir / p.name), plotted_img, dataset)
        if opt.verbose:
            LOGGER.info(f"{s}{profilers[1].dt * 1E3:.1f}ms")

    # Print results
    if opt.verbose and seen:
        t = tuple(x.t / seen * 1e3 for x in profilers)  # speeds per image
        LOGGER.info(
            f"Speed: {t[0]:.1f}ms preprocess, {t[1]:.1f}ms inference, {t[2]:.1f}ms postprocess per image at shape {(1, 3, *im.shape[2:])}"
        )

    if opt.save or opt.save_txt or opt.save_crop:
        nl = len(list(save_dir.glob("labels/*.txt")))  # number of labels
        s = f"\n{nl} label{'s' * (nl > 1)} saved to {save_dir / 'labels'}" if opt.save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")

if __name__ == "__main__":
    opt = training_arguments(True)
    # check config
    opt.noval, opt.nosave = True, True
    
    opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = \
            check_file(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.weights), str(opt.project) 
    assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
    
    if opt.name == 'cfg':
        opt.name = Path(opt.cfg).stem  # use model.yaml as name
    opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    source = r"C:\Users\admin\Desktop\deyo\images\train\P66_png.rf.bf38ad20afa863ccfea14e699587adba.jpg"
    predict(opt, source)