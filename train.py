import argparse
import os

import torch
from torch.utils.data import DataLoader
import tqdm
import time

from models import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from test import evaluate
from model_new import *

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
parser.add_argument("--batch_size", type=int, default=32, help="size of each image batch")
parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
parser.add_argument("--model_def", type=str, default="config/yolov3-voc.cfg", help="path to model definition file")
parser.add_argument("--data_config", type=str, default="config/voc.data", help="path to data config file")
parser.add_argument("--pretrained_weights", type=str, default='weights/darknet53.conv.74',
                    help="if specified starts from checkpoint model")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
parser.add_argument("--multiscale_training", type=bool, default=True, help="allow for multi-scale training")
parser.add_argument('--new_model', type=bool, default=True)
args = parser.parse_args()
print(args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
now = time.strftime('%y%m%d_%H%M%S', time.localtime(time.time()))

# Tensorboard writer 객체 생성
log_dir = os.path.join('logs', now)
os.makedirs(log_dir, exist_ok=True)
logger = Logger(log_dir)

# Get data configuration
data_config = parse_data_config(args.data_config)
train_path = data_config["train"]
valid_path = data_config["valid"]
class_names = load_classes(data_config["names"])

# Initiate model
if args.new_model:
    model = YOLOv3(416, 20).to(device)
else:
    model = Darknet(args.model_def, img_size=args.img_size).to(device)
model.apply(init_weights_normal)

# If specified we start from checkpoint
if args.pretrained_weights:
    if args.pretrained_weights.endswith(".pth"):
        model.load_state_dict(torch.load(args.pretrained_weights))
    else:
        model.load_darknet_weights(args.pretrained_weights)

# Get dataloader
dataset = ListDataset(train_path, augment=True, multiscale=args.multiscale_training)
dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=args.batch_size,
                                         shuffle=True,
                                         num_workers=args.n_cpu,
                                         pin_memory=True,
                                         collate_fn=dataset.collate_fn)

# Set optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Set learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=9, gamma=0.8)

# Set printing current batch loss tqdm
loss_log = tqdm.tqdm(total=0, position=2, bar_format='{desc}', leave=False)

# Training code.
for epoch in tqdm.tqdm(range(args.epochs), desc='Epoch'):
    model.train()
    tensorboard_log = []

    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc='Batch', leave=False)):
        step = len(dataloader) * epoch + batch_i

        imgs = imgs.to(device)
        targets = targets.to(device)

        loss, outputs = model(imgs, targets)
        loss.backward()

        # Accumulates gradient before each step
        if step % args.gradient_accumulations == 0:
            optimizer.step()
            optimizer.zero_grad()

        # Print total loss
        loss_log.set_description_str('Loss: {:.6f}'.format(loss.item()))

        # Tensorboard logging
        for i, yolo_layer in enumerate(model.yolo_layers):
            tensorboard_log += [(f"{'layer_loss'}_{i + 1}", yolo_layer.metrics['layer_loss'])]
        tensorboard_log += [("total_loss", loss.item())]
        logger.list_of_scalars_summary(tensorboard_log, step)

    scheduler.step()

    # Evaluate the model on the validation set
    precision, recall, AP, f1, ap_class = evaluate(model,
                                                   path=valid_path,
                                                   iou_thres=0.5,
                                                   conf_thres=0.5,
                                                   nms_thres=0.5,
                                                   img_size=args.img_size,
                                                   batch_size=args.batch_size,
                                                   num_workers=args.n_cpu)
    evaluation_metrics = [
        ("val_precision", precision.mean()),
        ("val_recall", recall.mean()),
        ("val_mAP", AP.mean()),
        ("val_f1", f1.mean()),
    ]
    logger.list_of_scalars_summary(evaluation_metrics, epoch)

    # Save checkpoint file
    save_dir = os.path.join('checkpoints', now)
    os.makedirs(save_dir, exist_ok=True)
    dataset_name = os.path.split(args.data_config)[-1].split('.')[0]
    torch.save(model.state_dict(), os.path.join(save_dir, "yolov3_{}_{}.pth".format(dataset_name, epoch)))
