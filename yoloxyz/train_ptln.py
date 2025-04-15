import os
import sys
import yaml
import torch
import shutil
import numpy as np
import torch.distributed as dist
from pathlib import Path
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from yolov9.utils.torch_utils import select_device, torch_distributed_zero_first, de_parallel
from yolov9.utils.general import LOGGER, check_file, init_seeds, intersect_dicts, check_img_size, colorstr, labels_to_class_weights, increment_path, check_yaml, check_dataset, yaml_save
from yolov9.utils.downloads import attempt_download
from yolov9.utils.dataloaders import create_dataloader
from engine import LitYOLO
from arguments import training_arguments
from multitasks.models.yolov9.yolo import Model as YOLO

# Constants
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

def setup_environment(opt, device):
    """Setup save directory, seeds, and hyperparameters."""
    save_dir = Path(opt.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    opt.cfg = check_file(opt.cfg)
    init_seeds(opt.seed + 1 + RANK, deterministic=True)

    # Load hyperparameters
    with open(opt.hyp, errors='ignore') as f:
        hyp = yaml.safe_load(f)
    hyp['anchor_t'] = 5.0
    opt.hyp = hyp.copy()

    # Save configurations
    if not opt.evolve:
        yaml_save(save_dir / 'hyp.yaml', hyp)
        yaml_save(save_dir / 'opt.yaml', vars(opt))
        shutil.copy('scripts/deyo2gpu.sh', save_dir / 'train.sh')
        shutil.copy(opt.cfg, save_dir / 'cfg.yaml')
    
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    return hyp

def init_model(opt, hyp, num_classes, device):
    """Initialize YOLO model with optional pretrained weights."""
    pretrained = opt.weights.endswith('.pt')
    if pretrained:
        with torch_distributed_zero_first(LOCAL_RANK):
            weights = attempt_download(opt.weights)
        ckpt = torch.load(weights, map_location='cpu')
        model = YOLO(opt.cfg or ckpt['model'].yaml, ch=3, nc=num_classes, anchors=hyp.get('anchors')).to(device)
        exclude = ['anchor'] if (opt.cfg or hyp.get('anchors')) and not opt.resume else []
        csd = intersect_dicts(ckpt['model'].float().state_dict(), model.state_dict(), exclude=exclude)
        model.load_state_dict(csd, strict=False)
        LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')
    else:
        model = YOLO(opt.cfg, ch=3, nc=num_classes, anchors=hyp.get('anchors')).to(device)
    
    # Freeze layers
    freeze = [f'model.{x}.' for x in (opt.freeze if len(opt.freeze) > 1 else range(opt.freeze[0]))]
    for k, v in model.named_parameters():
        if any(x in k for x in freeze):
            LOGGER.info(f'freezing {k}')
            v.requires_grad = False
    
    return model

def create_dataloaders(opt, hyp, imgsz, gs, train_path, val_path, num_classes):
    """Create train and validation dataloaders."""
    train_loader, dataset = create_dataloader(
        train_path, imgsz, opt.batch_size, gs, opt.single_cls, hyp=hyp, augment=True,
        cache=None if opt.cache == 'val' else opt.cache, rect=opt.rect, rank=LOCAL_RANK,
        workers=opt.workers, image_weights=opt.image_weights, close_mosaic=opt.close_mosaic != 0,
        quad=opt.quad, prefix=colorstr('train: '), shuffle=False, min_items=opt.min_items
    )
    
    val_loader = create_dataloader(
        val_path, imgsz, opt.batch_size, gs, opt.single_cls, hyp=hyp,
        cache=None if opt.noval else opt.cache, rank=LOCAL_RANK, workers=opt.workers,
        pad=0.5, prefix=colorstr('val: ')
    )[0]
    
    labels = np.concatenate(dataset.labels, 0)
    mlc = int(labels[:, 0].max())
    assert mlc < num_classes, f'Label class {mlc} exceeds nc={num_classes} in {opt.data}'
    
    return train_loader, val_loader, dataset

def setup_trainer(opt, model, hyp, num_classes, train_loader, val_loader, device, names):
    """Setup PyTorch Lightning trainer and model."""
    wandb_logger = WandbLogger(project=opt.project_wandb, name=opt.name, log_model="all")
    model_checkpoint = ModelCheckpoint(
        save_top_k=3, monitor="val/loss", mode="min", dirpath=f'{opt.save_dir}/weights',
        filename="sample-{epoch:02d}", save_weights_only=True
    )
    
    # Configure model attributes
    model.nc = num_classes
    model.hyp = hyp
    model.class_weights = labels_to_class_weights(train_loader.dataset.labels, num_classes).to(device) * num_classes
    model.names = names
    
    lit_yolo = LitYOLO(opt=opt, model=model, hyp=hyp, num_classes=num_classes)
    
    trainer = Trainer(
        max_epochs=opt.epochs, accelerator=opt.accelerator, devices='auto',
        callbacks=[model_checkpoint], strategy='ddp_find_unused_parameters_true' if len(opt.device) > 1 else 'auto',
        log_every_n_steps=opt.log_steps, logger=wandb_logger, precision=16, enable_progress_bar=True
    )
    
    return trainer, lit_yolo

def main(opt, device):
    """Main training function."""
    # Setup
    hyp = setup_environment(opt, device)
    
    # Data
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = check_dataset(opt.data)
    num_classes = 1 if opt.single_cls else int(data_dict['nc'])
    names = ['item'] if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']
    assert len(names) == num_classes, f'{len(names)} names found for nc={num_classes} in {opt.data}'
    train_path, val_path = data_dict['train'], data_dict['val']
    
    # Model
    model = init_model(opt, hyp, num_classes, device)
    
    # Image size and batch norm
    gs = max(int(model.stride.max()), 32)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)
    if opt.sync_bn and device.type != 'cpu' and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info('Using SyncBatchNorm()')
    
    # Dataloaders
    train_loader, val_loader, dataset = create_dataloaders(opt, hyp, imgsz, gs, train_path, val_path, num_classes)
    
    # Optimizer
    norm_batch_size = 64
    accumulate = max(round(norm_batch_size / opt.batch_size), 1)
    hyp['weight_decay'] *= opt.batch_size * accumulate / norm_batch_size
    
    # Trainer
    if not opt.resume:
        model.half().float()
    trainer, lit_yolo = setup_trainer(opt, model, hyp, num_classes, train_loader, val_loader, device, names)
    
    # Train
    LOGGER.info("\n*** Start training ***\n")
    trainer.fit(
        model=lit_yolo,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader if opt.do_eval else None
    )
    
if __name__ == '__main__':
    opt = training_arguments(True)
    opt.noval, opt.nosave = True, True
    
    # Check and prepare configurations
    opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = \
        check_file(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.weights), str(opt.project)
    assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
    
    if opt.evolve:
        if opt.project == str(ROOT / 'runs/train'):
            opt.project = str(ROOT / 'runs/evolve')
        opt.exist_ok, opt.resume = opt.resume, False
    if opt.name == 'cfg':
        opt.name = Path(opt.cfg).stem
    opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    
    # Device and DDP setup
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        assert not opt.image_weights, '--image-weights not compatible with DDP'
        assert not opt.evolve, '--evolve not compatible with DDP'
        assert opt.batch_size != -1, 'AutoBatch with --batch-size -1 not compatible with DDP'
        assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} must be multiple of WORLD_SIZE'
        assert torch.cuda.device_count() > LOCAL_RANK, 'Insufficient CUDA devices for DDP'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")
    
    main(opt, device)