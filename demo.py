import argparse
import os
import random

import torch
import torch.utils.data
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import tqdm

import model.yolov3
import model.yolov3_proposed
import utils.datasets
import utils.utils

parser = argparse.ArgumentParser()
parser.add_argument("--image_folder", type=str, default="../../data/voc_test", help="path to image folder")
parser.add_argument("--save_folder", type=str, default='../../detect', help='path to saving result folder')
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--data_config", type=str, default="config/voc.data", help="path to data config file")
parser.add_argument("--pretrained_weights", type=str, default="weights/yolov3_voc.pth",
                    help="path to pretrained weights file")
parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
args = parser.parse_args()
print(args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_config = utils.utils.parse_data_config(args.data_config)
classes = utils.utils.load_classes(data_config["names"])

# Set up model
model = model.yolov3.YOLOv3(args.img_size, int(data_config['classes'])).to(device)
if args.pretrained_weights.endswith('.pth'):
    model.load_state_dict(torch.load(args.pretrained_weights))
else:
    model.load_darknet_weights(args.pretrained_weights)

# Set dataloader
dataset = utils.datasets.ImageFolder(args.image_folder, img_size=args.img_size)
dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=args.batch_size,
                                         shuffle=False,
                                         num_workers=args.n_cpu)

# Set in evaluation mode
model.eval()

img_paths = []  # Stores image paths
img_detections = []  # Stores detections for each image index
for paths, imgs in tqdm.tqdm(dataloader, desc='Batch'):
    with torch.no_grad():
        imgs = imgs.to(device)
        prediction = model(imgs)
        prediction = utils.utils.non_max_suppression(prediction, args.conf_thres, args.nms_thres)

    # Save image and detections
    img_paths.extend(paths)
    img_detections.extend(prediction)

# Bounding-box colors
cmap = plt.get_cmap("tab20b")
colors = [cmap(i) for i in np.linspace(0, 1, 20)]

os.makedirs(args.save_folder, exist_ok=True)

# Save result images
for path, detection in tqdm.tqdm(zip(img_paths, img_detections), desc='Save images'):

    # Replace Windows path separator to Linux path separator
    path = path.replace('\\', '/')

    # Open original image
    image = Image.open(path)
    draw = ImageDraw.Draw(image)

    # Draw bounding boxes and labels of detections
    if detection is not None:
        # Rescale boxes to original image
        detection = utils.utils.rescale_boxes(detection, args.img_size, image.size)

        unique_labels = detection[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        bbox_colors = random.sample(colors, n_cls_preds)
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:
            color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]

            # Draw bounding box
            draw.rectangle(((x1, y1), (x2, y2)), outline=(0, 0, 255), width=2)

            # Draw label
            text = '{}{:.3f}'.format(classes[int(cls_pred)], cls_conf.item())
            font = ImageFont.truetype('calibri.ttf', size=12)
            text_width, text_height = font.getsize(text)
            draw.rectangle(((x1, y1), (x1 + text_width, y1 + text_height)), fill=(0, 0, 255))
            draw.text((x1, y1), text, fill=(255, 255, 255), font=font)

    # Save result image
    filename = path.split("/")[-1].split(".")[0]
    image.save("{}/{}.jpg".format(args.save_folder, filename))
