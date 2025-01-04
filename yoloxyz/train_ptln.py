import os
import sys
import yaml
import torch
from pathlib import Path

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from engine import LitYOLO
from backbones.yolov9.utils.torch_utils import select_device, torch_distributed_zero_first, de_parallel
from backbones.yolov9.utils.general import LOGGER, check_file, init_seeds, intersect_dicts, check_img_size, colorstr, increment_path, check_yaml, check_dataset
from backbones.yolov9.utils.loggers import Loggers
from backbones.yolov9.utils.downloads import attempt_download
from model.yolo import Model as YOLO
from arguments import training_arguments
from data.datasets import get_dataloader, get_val_dataloader

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

def main(opt):
    save_dir = Path(opt.save_dir)
    opt.cfg = check_file(opt.cfg)  # check file
    device = select_device(opt.device)
    init_seeds(opt.seed + 1 + RANK, deterministic=True)

    # Hyperparameters
    if isinstance(opt.hyp, str):
        with open(opt.hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    hyp['anchor_t'] = 5.0
    opt.hyp = hyp.copy()  # for saving hyps to checkpoints

    # Loggers
    data_dict = None
    if RANK  in {-1, 0}:
        loggers = Loggers(save_dir, opt.weights, opt, hyp, LOGGER) 

        # Process custom dataset artifact link
        data_dict = loggers.remote_dataset
        if opt.resume:  # If resuming runs from remote artifact
            weights, hyp = opt.weights, opt.hyp

    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = data_dict or check_dataset(opt.data) 
    wandb_logger = WandbLogger(project=opt.name, log_model="all")

    cuda = device.type != 'cpu'
    num_classes = 1 if opt.single_cls else int(data_dict['nc'])  # number of classes
    names = ['item'] if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    assert len(names) == num_classes, '%g names found for nc=%g dataset in %s' % (len(names), num_classes, opt.data)  # check
    train_path, val_path = data_dict['train'], data_dict['val']

    pretrained = opt.weights.endswith('.pt')
    if pretrained:
        with torch_distributed_zero_first(LOCAL_RANK):
            weights = attempt_download(opt.weights)  # download if not found locally
        ckpt = torch.load(weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
        model = YOLO(opt.cfg or ckpt['model'].yaml, ch=3, nc=num_classes, anchors=hyp.get('anchors')).to(device)  # create
        exclude = ['anchor'] if (opt.cfg or hyp.get('anchors')) and not opt.resume else []  # exclude keys
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(csd, strict=False)  # load
        LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')
    else:
        model = YOLO(opt.cfg, ch=3, nc=num_classes, anchors=hyp.get('anchors')).to(device)
    
    # Freeze
    freeze = [f'model.{x}.' for x in (opt.freeze if len(opt.freeze) > 1 else range(opt.freeze[0]))]  # layers to freeze
    for k, v in model.named_parameters():
        if any(x in k for x in freeze):
            LOGGER.info(f'freezing {k}')
            v.requires_grad = False
    
    # Optimizer
    norm_batch_size = 64  # nominal batch size
    accumulate = max(round(norm_batch_size / opt.batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= opt.batch_size * accumulate / norm_batch_size  # scale weight_decay
    
    gs = max(int(model.stride.max() if hasattr(model, "stride") else 32), 32)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)

    # SyncBatchNorm
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info('Using SyncBatchNorm()')
    
    train_loader = get_dataloader(opt, imgsz, data_dict, train_path, workers=opt.workers,batch_size=opt.batch_size, rank=RANK, mode="train")
 
    # Process 0
    if RANK in {-1, 0}:  
        val_loader = get_val_dataloader(opt, data= data_dict, stride = gs, dataset_path= val_path, batch_size= opt.batch_size, workers = opt.workers)
        if not opt.resume:
            model.half().float()  # pre-reduce anchor precision

    # Model attributes
    dist = True if len(opt.device) > 1 else False
    nl = de_parallel(model).model[-1].nl  
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = num_classes  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.names = names

    lit_yolo = LitYOLO(opt = opt, model=model, model_device = device, num_classes = num_classes, hyp = hyp)

    # Create callback functions
    model_checkpoint = ModelCheckpoint(save_top_k=3,
                        monitor="train/loss",
                        mode="min", dirpath="output/",
                        filename="sample-{epoch:02d}",
                        save_weights_only=True)
    
    opt.device = [int(x) for x in opt.device]
    trainer = Trainer(max_epochs=opt.epochs,
                      accelerator=opt.accelerator,
                      devices=1,
                      callbacks=[model_checkpoint],
                      strategy='ddp_find_unused_parameters_true' if dist else 'auto',
                      log_every_n_steps=opt.log_steps,
                      logger=wandb_logger)
    
    # if opt.do_train:
    LOGGER.info("*** Start training ***")
    trainer.fit(
        model=lit_yolo, 
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )
    
    # Saves only on the main process    
    saved_ckpt_path = f'{opt.save_dir}/weights'
    os.makedirs(saved_ckpt_path, exist_ok=True)
    saved_ckpt_path = f'{saved_ckpt_path}/best.pt'
    trainer.save_checkpoint(saved_ckpt_path)

    if opt.do_eval:
        LOGGER.info("\n\n*** Evaluate ***")
        trainer.devices = 0
        trainer.test(lit_yolo, dataloaders=val_loader, ckpt_path="best")
    
if __name__ == '__main__':
    opt = training_arguments(True)
    # check config
    opt.noval, opt.nosave = True, True
    
    opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = \
            check_file(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.weights), str(opt.project) 
    assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
    if opt.evolve:
        if opt.project == str(ROOT / 'runs/train'):  # if default project name, rename to runs/evolve
            opt.project = str(ROOT / 'runs/evolve')
        opt.exist_ok, opt.resume = opt.resume, False  # pass resume to exist_ok and disable resume
    if opt.name == 'cfg':
        opt.name = Path(opt.cfg).stem  # use model.yaml as name
    opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    main(opt)
