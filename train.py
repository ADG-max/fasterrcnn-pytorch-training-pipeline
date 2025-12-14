"""
USAGE

# training with Faster RCNN ResNet50 FPN model without mosaic or any other augmentation:
python train.py --model fasterrcnn_resnet50_fpn --epochs 2 --data data_configs/voc.yaml --mosaic 0 --batch 4

# Training on ResNet50 FPN with custom project folder name with mosaic augmentation (ON by default):
python train.py --model fasterrcnn_resnet50_fpn --epochs 2 --data data_configs/voc.yaml --name resnet50fpn_voc --batch 4

# Training on ResNet50 FPN with custom project folder name with mosaic augmentation (ON by default) and added training augmentations:
python train.py --model fasterrcnn_resnet50_fpn --epochs 2 --use-train-aug --data data_configs/voc.yaml --name resnet50fpn_voc --batch 4

# Distributed training:
export CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.launch --nproc_per_node=2 --use_env train.py --data data_configs/smoke.yaml --epochs 100 --model fasterrcnn_resnet50_fpn --name smoke_training --batch 16
"""
import torch
import argparse
import yaml
import numpy as np
import torchinfo
import os

from torch_utils.engine import (
    train_one_epoch, evaluate, utils
)
from torch.utils.data import (
    distributed, WeightedRandomSampler, SequentialSampler
)
from datasets import (
    create_train_dataset, create_valid_dataset, 
    create_train_loader, create_valid_loader
)
from models.create_fasterrcnn_model import create_model
from utils.general import (
    set_training_dir, Averager, 
    save_model, save_loss_plot,
    show_tranformed_image,
    save_mAP, save_model_state, SaveBestModel,
    yaml_save, init_seeds, EarlyStopping
)
from utils.logging import (
    set_log, coco_log,
    set_summary_writer, 
    tensorboard_loss_log, 
    tensorboard_map_log,
    csv_log,
    wandb_log, 
    wandb_save_model,
    wandb_init
)
from utils.visual_config import get_visual_config

torch.multiprocessing.set_sharing_strategy('file_system')

RANK = int(os.getenv('RANK', -1))

# For same annotation colors each time.
np.random.seed(42)

def parse_opt():
    # Construct the argument parser.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--model', 
        default='fasterrcnn_resnet50_fpn_v2',
        help='name of the model'
    )
    parser.add_argument(
        '--data', 
        default=None,
        help='path to the data config file'
    )
    parser.add_argument(
        '-d', '--device', 
        default='cuda',
        help='computation/training device, default is GPU if GPU present'
    )
    parser.add_argument(
        '-e', '--epochs', 
        default=5,
        type=int,
        help='number of epochs to train for'
    )
    parser.add_argument(
        '-j', '--workers', 
        default=4,
        type=int,
        help='number of workers for data processing/transforms/augmentations'
    )
    parser.add_argument(
        '-b', '--batch', 
        default=4, 
        type=int, 
        help='batch size to load the data'
    )
    parser.add_argument(
        '--lr', 
        default=0.001,
        help='learning rate for the optimizer',
        type=float
    )
    parser.add_argument(
        '-ims', '--imgsz',
        default=640, 
        type=int, 
        help='image size to feed to the network'
    )
    parser.add_argument(
        '-n', '--name', 
        default=None, 
        type=str, 
        help='training result dir name in outputs/training/, (default res_#)'
    )
    parser.add_argument(
        '-vt', '--vis-transformed', 
        dest='vis_transformed', 
        action='store_true',
        help='visualize transformed images fed to the network'
    )
    parser.add_argument(
        '--mosaic', 
        default=0.0,
        type=float,
        help='probability of applying mosaic, (default, always apply)'
    )
    parser.add_argument(
        '-uta', '--use-train-aug', 
        dest='use_train_aug', 
        action='store_true',
        help='whether to use train augmentation, blur, gray, \
              brightness contrast, color jitter, random gamma \
              all at once'
    )
    parser.add_argument(
        '-ca', '--cosine-annealing', 
        dest='cosine_annealing', 
        action='store_true',
        help='use cosine annealing warm restarts'
    )
    parser.add_argument(
        '-w', '--weights', 
        default=None, 
        type=str,
        help='path to model weights if using pretrained weights'
    )
    parser.add_argument(
        '-r', '--resume-training', 
        dest='resume_training', 
        action='store_true',
        help='whether to resume training, if true, \
            loads previous training plots and epochs \
            and also loads the otpimizer state dictionary'
    )
    parser.add_argument(
        '-st', '--square-training',
        dest='square_training',
        action='store_true',
        help='Resize images to square shape instead of aspect ratio resizing \
              for single image training. For mosaic training, this resizes \
              single images to square shape first then puts them on a \
              square canvas.'
    )
    parser.add_argument(
        '--world-size', 
        default=1, 
        type=int, 
        help='number of distributed processes'
    )
    parser.add_argument(
        '--dist-url',
        default='env://',
        type=str,
        help='url used to set up the distributed training'
    )
    parser.add_argument(
        '-dw', '--disable-wandb',
        dest="disable_wandb",
        action='store_true',
        help='whether to use the wandb'
    )
    parser.add_argument(
        '--sync-bn',
        dest='sync_bn',
        help='use sync batch norm',
        action='store_true'
    )
    parser.add_argument(
        '--amp',
        action='store_true',
        help='use automatic mixed precision'
    )
    parser.add_argument(
        '--patience',
        default=10,
        help='number of epochs to wait for when mAP does not increase to \
              trigger early stopping',
        type=int
    )
    parser.add_argument(
        '--seed',
        default=0,
        type=int ,
        help='golabl seed for training'
    )
    parser.add_argument(
        '--project-dir',
        dest='project_dir',
        default=None,
        help='save resutls to custom dir instead of `outputs` directory, \
              --project-dir will be named if not already present',
        type=str
    )
    parser.add_argument(
        '--label-type',
        dest='label_type',
        default='pascal_voc',
        choices=['pascal_voc', 'yolo'],
        help='label files type, you can either Pascal VOC XML type or, \
              yolo txt type label files'
    )
    parser.add_argument(
        '--optimizer',
        default=None,
        choices=['SGD', 'AdamW']
    )

    args = vars(parser.parse_args())
    return args

class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, warmup_factor=1e-3, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.warmup_factor = warmup_factor
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.warmup_epochs:
            return [base_lr for base_lr in self.base_lrs]
        alpha = float(self.last_epoch) / float(max(1, self.warmup_epochs))
        factor = self.warmup_factor + alpha * (1 - self.warmup_factor)
        return [base_lr * factor for base_lr in self.base_lrs]

def main(args):
    # Initialize distributed mode.
    utils.init_distributed_mode(args)

    # ==============================
    # DDP: Prevent folder creation race-condition
    # ==============================
    if RANK != 0:
        torch.distributed.barrier()
    else:
        os.makedirs('outputs/training', exist_ok=True)
        torch.distributed.barrier()

    # Initialize W&B with project name.
    if not args['disable_wandb']:
        wandb_init(name=args['name'])
    # Load the data configurations
    with open(args['data']) as file:
        data_configs = yaml.safe_load(file)

    # Hitung distribusi jumlah sampel per kelas (VOC)
    import xml.etree.ElementTree as ET
    from collections import defaultdict
    
    def count_classes_from_voc(xml_dir, class_names):
        counts = defaultdict(int)
    
        xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]
        print(f"\n📂 Total XML files di {xml_dir}: {len(xml_files)}")
    
        for xml_file in xml_files:
            xml_path = os.path.join(xml_dir, xml_file)
            tree = ET.parse(xml_path)
            root = tree.getroot()
    
            for obj in root.findall("object"):
                label = obj.find("name").text.strip()
    
                if label in class_names:
                    counts[label] += 1
                else:
                    counts["__unknown__"] += 1
    
        return counts
    
    
    # Ambil direktori dari YAML
    train_xml_dir = data_configs['TRAIN_DIR_LABELS']
    valid_xml_dir = data_configs['VALID_DIR_LABELS']
    test_xml_dir  = data_configs['TEST_DIR_LABELS']
    
    # background tidak dihitung karena tidak ada di XML
    VOC_CLASSES = [c for c in data_configs["CLASSES"] if c != "__background__"]
    
    print("\n==============================")
    print("   HITUNG DISTRIBUSI KELAS   ")
    print("==============================")
    
    train_counts = count_classes_from_voc(train_xml_dir, VOC_CLASSES)
    valid_counts = count_classes_from_voc(valid_xml_dir, VOC_CLASSES)
    test_counts  = count_classes_from_voc(test_xml_dir, VOC_CLASSES)
    
    print("\n=== DISTRIBUSI TRAIN ===")
    print(train_counts)
    print("\n=== DISTRIBUSI VALID ===")
    print(valid_counts)
    print("\n=== DISTRIBUSI TEST ===")
    print(test_counts)
    
    # Total
    total_counts = defaultdict(int)
    for d in (train_counts, valid_counts, test_counts):
        for k, v in d.items():
            total_counts[k] += v
    
    print("\n=== TOTAL DISTRIBUSI ===")
    print(total_counts)
    print("=====================================\n")

    init_seeds(args['seed'] + 1 + RANK, deterministic=True)
    
    # Settings/parameters/constants.
    TRAIN_DIR_IMAGES = os.path.normpath(data_configs['TRAIN_DIR_IMAGES'])
    TRAIN_DIR_LABELS = os.path.normpath(data_configs['TRAIN_DIR_LABELS'])
    VALID_DIR_IMAGES = os.path.normpath(data_configs['VALID_DIR_IMAGES'])
    VALID_DIR_LABELS = os.path.normpath(data_configs['VALID_DIR_LABELS'])
    CLASSES = data_configs['CLASSES']
    NUM_CLASSES = data_configs['NC']
    COLORS, LABEL_BG, LABEL_TEXT_COLOR = get_visual_config(CLASSES)
    NUM_WORKERS = args['workers']
    DEVICE = torch.device(args['device'])
    print("device",DEVICE)
    NUM_EPOCHS = args['epochs']
    SAVE_VALID_PREDICTIONS = data_configs['SAVE_VALID_PREDICTION_IMAGES']
    BATCH_SIZE = args['batch']
    VISUALIZE_TRANSFORMED_IMAGES = args['vis_transformed']
    OUT_DIR = set_training_dir(args['name'], args['project_dir'])
    SCALER = torch.amp.GradScaler("cuda") if args['amp'] else None
    # Set logging file.
    set_log(OUT_DIR)
    writer = set_summary_writer(OUT_DIR)

    yaml_save(file_path=os.path.join(OUT_DIR, 'opt.yaml'), data=args)

    # Model configurations
    IMAGE_SIZE = args['imgsz']
    
    train_dataset = create_train_dataset(
        TRAIN_DIR_IMAGES, 
        TRAIN_DIR_LABELS,
        IMAGE_SIZE, 
        CLASSES,
        use_train_aug=args['use_train_aug'],
        mosaic=args['mosaic'],
        square_training=args['square_training'],
        label_type=args['label_type']
    )
    valid_dataset = create_valid_dataset(
        VALID_DIR_IMAGES, 
        VALID_DIR_LABELS, 
        IMAGE_SIZE, 
        CLASSES,
        square_training=args['square_training'],
        label_type=args['label_type']
    )
    print('Creating data loaders')
    if args['distributed']:
        train_sampler = distributed.DistributedSampler(
            train_dataset
        )
        valid_sampler = distributed.DistributedSampler(
            valid_dataset, shuffle=False
        )
    else:
        # Ambil label per bbox
        labels = np.array(train_dataset.bbox_labels)
        valid_idx = labels >= 0
        filtered_labels = labels[valid_idx]
    
        # Hitung jumlah kelas
        class_counts = np.bincount(filtered_labels)
        class_counts[class_counts == 0] = 1
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[filtered_labels]
    
        # Sampler
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(filtered_labels),
            replacement=True
        )

        valid_sampler = SequentialSampler(valid_dataset)

    train_loader = create_train_loader(
        train_dataset, BATCH_SIZE, NUM_WORKERS, batch_sampler=train_sampler
    )
    valid_loader = create_valid_loader(
        valid_dataset, BATCH_SIZE, NUM_WORKERS, batch_sampler=valid_sampler
    )
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(valid_dataset)}\n")

    if VISUALIZE_TRANSFORMED_IMAGES:
        show_tranformed_image(train_loader, DEVICE, CLASSES, COLORS)

    # Initialize the Averager class.
    train_loss_hist = Averager()
    # Train and validation loss lists to store loss values of all
    # iterations till ena and plot graphs for all iterations.
    train_loss_list = []
    loss_cls_list = []
    loss_box_reg_list = []
    loss_objectness_list = []
    loss_rpn_list = []
    train_loss_list_epoch = []
    val_map_05 = []
    val_map = []
    start_epochs = 0

    if args['weights'] is None:
        print('Building model from models folder...')
        build_model = create_model[args['model']]
        model = build_model(num_classes=NUM_CLASSES, pretrained=True)

    # Load pretrained weights if path is provided.
    if args['weights'] is not None:
        print('Loading pretrained weights...')
        
        # Load the pretrained checkpoint.
        checkpoint = torch.load(args['weights'], map_location=DEVICE) 
        keys = list(checkpoint['model_state_dict'].keys())
        ckpt_state_dict = checkpoint['model_state_dict']
        # Get the number of classes from the loaded checkpoint.
        old_classes = ckpt_state_dict['roi_heads.box_predictor.cls_score.weight'].shape[0]

        # Build the new model with number of classes same as checkpoint.
        build_model = create_model[args['model']]
        model = build_model(num_classes=old_classes)
        # Load weights.
        model.load_state_dict(ckpt_state_dict)

        # Change output features for class predictor and box predictor
        # according to current dataset classes.
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor.cls_score = torch.nn.Linear(
            in_features=in_features, out_features=NUM_CLASSES, bias=True
        )
        model.roi_heads.box_predictor.bbox_pred = torch.nn.Linear(
            in_features=in_features, out_features=NUM_CLASSES*4, bias=True
        )

        if args['resume_training']:
            print('RESUMING TRAINING...')
            # Update the starting epochs, the batch-wise loss list, 
            # and the epoch-wise loss list.
            if checkpoint['epoch']:
                start_epochs = checkpoint['epoch']
                print(f"Resuming from epoch {start_epochs}...")
            if checkpoint['train_loss_list']:
                print('Loading previous batch wise loss list...')
                train_loss_list = checkpoint['train_loss_list']
            if checkpoint['train_loss_list_epoch']:
                print('Loading previous epoch wise loss list...')
                train_loss_list_epoch = checkpoint['train_loss_list_epoch']
            if checkpoint['val_map']:
                print('Loading previous mAP list')
                val_map = checkpoint['val_map']
            if checkpoint['val_map_05']:
                val_map_05 = checkpoint['val_map_05']

    # Make the model transform's `min_size` same as `imgsz` argument. 
    model.transform.min_size = (args['imgsz'], )
    model = model.to(DEVICE)
    if args['sync_bn'] and args['distributed']:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if args['distributed']:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args['gpu']]
        )
    try:
        torchinfo.summary(
            model, 
            device=DEVICE, 
            input_size=(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE),
            row_settings=["var_names"],
            col_names=("input_size", "output_size", "num_params") 
        )
    except:
        print(model)
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    # Get the model parameters.
    params = [p for p in model.parameters() if p.requires_grad]
    # Define the optimizer.

    if args['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=args['lr'], weight_decay=1e-4)
        print(f"Using AdamW for {args['model']}")
    elif args['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(params, lr=args['lr'], momentum=0.9, nesterov=True)
        print(f"Using SGD for {args['model']}")
    else:
        # default behavior
        if 'dinov3' in args['model']:
            optimizer = torch.optim.AdamW(params, lr=args['lr'], weight_decay=1e-4)
            print(f"Using default AdamW for {args['model']}")
        else:
            optimizer = torch.optim.SGD(params, lr=args['lr'], momentum=0.9, nesterov=True)
            print(f"Using default SGD for {args['model']}")

    if args['resume_training']: 
        # LOAD THE OPTIMIZER STATE DICTIONARY FROM THE CHECKPOINT.
        print('Loading optimizer state dictionary...')
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if args['cosine_annealing']:
        warmup_epochs = 3
        steps = NUM_EPOCHS
    
        warmup_scheduler = WarmupScheduler(
            optimizer,
            warmup_epochs=warmup_epochs,
            warmup_factor=0.001
        )
    
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=steps,
            T_mult=1
        )
    
        scheduler = {
            "warmup": warmup_scheduler,
            "cosine": cosine_scheduler,
            "warmup_epochs": warmup_epochs
        }
    else:
        scheduler = None

    save_best_model = SaveBestModel()
    early_stopping = EarlyStopping(patience=args['patience'])

    for epoch in range(start_epochs, NUM_EPOCHS):
        train_loss_hist.reset()

        _, batch_loss_list, \
            batch_loss_cls_list, \
            batch_loss_box_reg_list, \
            batch_loss_objectness_list, \
            batch_loss_rpn_list = train_one_epoch(
            model, 
            optimizer, 
            train_loader, 
            DEVICE, 
            epoch, 
            train_loss_hist,
            print_freq=100,
            scheduler=scheduler,
            scaler=SCALER
        )

        stats, val_pred_image = evaluate(
            model, 
            valid_loader, 
            device=DEVICE,
            save_valid_preds=SAVE_VALID_PREDICTIONS,
            out_dir=OUT_DIR,
            classes=CLASSES,
            colors=COLORS,
            label_bg=LABEL_BG,
            label_text_color=LABEL_TEXT_COLOR
        )

        # Append the current epoch's batch-wise losses to the `train_loss_list`.
        train_loss_list.extend(batch_loss_list)
        loss_cls_list.append(np.mean(np.array(batch_loss_cls_list,)))
        loss_box_reg_list.append(np.mean(np.array(batch_loss_box_reg_list)))
        loss_objectness_list.append(np.mean(np.array(batch_loss_objectness_list)))
        loss_rpn_list.append(np.mean(np.array(batch_loss_rpn_list)))

        # Append curent epoch's average loss to `train_loss_list_epoch`.
        train_loss_list_epoch.append(train_loss_hist.value)
        val_map_05.append(stats[1])
        val_map.append(stats[0])

        # Save loss plot for batch-wise list.
        save_loss_plot(OUT_DIR, train_loss_list)
        # Save loss plot for epoch-wise list.
        save_loss_plot(
            OUT_DIR, 
            train_loss_list_epoch,
            'epochs',
            'train loss',
            save_name='train_loss_epoch' 
        )
        # Save all the training loss plots.
        save_loss_plot(
            OUT_DIR, 
            loss_cls_list, 
            'epochs', 
            'loss cls',
            save_name='train_loss_cls'
        )
        save_loss_plot(
            OUT_DIR, 
            loss_box_reg_list, 
            'epochs', 
            'loss bbox reg',
            save_name='train_loss_bbox_reg'
        )
        save_loss_plot(
            OUT_DIR,
            loss_objectness_list,
            'epochs',
            'loss obj',
            save_name='train_loss_obj'
        )
        save_loss_plot(
            OUT_DIR,
            loss_rpn_list,
            'epochs',
            'loss rpn bbox',
            save_name='train_loss_rpn_bbox'
        )

        # Save mAP plots.
        save_mAP(OUT_DIR, val_map_05, val_map)

        # Save batch-wise train loss plot using TensorBoard. Better not to use it
        # as it increases the TensorBoard log sizes by a good extent (in 100s of MBs).
        # tensorboard_loss_log('Train loss', np.array(train_loss_list), writer)

        # Save epoch-wise train loss plot using TensorBoard.
        tensorboard_loss_log(
            'Train loss', 
            np.array(train_loss_list_epoch), 
            writer,
            epoch
        )

        # Save mAP plot using TensorBoard.
        tensorboard_map_log(
            name='mAP', 
            val_map_05=np.array(val_map_05), 
            val_map=np.array(val_map),
            writer=writer,
            epoch=epoch
        )

        coco_log(OUT_DIR, stats)
        csv_log(
            OUT_DIR, 
            stats, 
            epoch,
            train_loss_list,
            loss_cls_list,
            loss_box_reg_list,
            loss_objectness_list,
            loss_rpn_list
        )

        # WandB logging.
        if not args['disable_wandb']:
            wandb_log(
                train_loss_hist.value,
                batch_loss_list,
                loss_cls_list,
                loss_box_reg_list,
                loss_objectness_list,
                loss_rpn_list,
                stats[1],
                stats[0],
                val_pred_image,
                IMAGE_SIZE
            )

        # Save the current epoch model state. This can be used 
        # to resume training. It saves model state dict, number of
        # epochs trained for, optimizer state dict, and loss function.
        save_model(
            epoch, 
            model, 
            optimizer, 
            train_loss_list, 
            train_loss_list_epoch,
            val_map,
            val_map_05,
            OUT_DIR,
            data_configs,
            args['model']
        )
        # Save the model dictionary only for the current epoch.
        save_model_state(model, OUT_DIR, data_configs, args['model'])
        # Save best model if the current mAP @0.5:0.95 IoU is
        # greater than the last hightest.
        save_best_model(
            model, 
            val_map[-1], 
            epoch, 
            OUT_DIR,
            data_configs,
            args['model']
        )

            # Early stopping check.
        early_stopping(stats[0])
        if early_stopping.early_stop:
            break
    
    # Save models to Weights&Biases.
    if not args['disable_wandb']:
        wandb_save_model(OUT_DIR)

    # Shutdown DDP properly
    if args['distributed']:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    args = parse_opt()
    main(args)

