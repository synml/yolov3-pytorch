import argparse
import os

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_config = utils.utils.parse_data_config(args.data_config)
num_classes = int(data_config['classes'])
class_names = utils.utils.load_classes(data_config['names'])

# Set up model
model = model.yolov3.YOLOv3(args.img_size, num_classes).to(device)
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

# Detect objects
img_paths = []  # Stores image paths
img_predictions = []  # Stores predictions for each image index
for paths, imgs in tqdm.tqdm(dataloader, desc='Batch'):
    with torch.no_grad():
        imgs = imgs.to(device)
        prediction = model(imgs)
        prediction = utils.utils.non_max_suppression(prediction, args.conf_thres, args.nms_thres)

    # Save image and prediction
    img_paths.extend(paths)
    img_predictions.extend(prediction)

# Bounding-box colors
cmap = np.array(plt.cm.get_cmap('Paired').colors)
cmap_rgb: list = np.multiply(cmap, 255).astype(np.int32).tolist()

# Save result images
os.makedirs(args.save_folder, exist_ok=True)
for path, prediction in tqdm.tqdm(zip(img_paths, img_predictions), desc='Save images', total=dataset.__len__()):
    # Open original image
    image = Image.open(path)
    draw = ImageDraw.Draw(image)

    if prediction is not None:
        # Rescale boxes to original image
        prediction = utils.utils.rescale_boxes_original(prediction, args.img_size, image.size)

        for x1, y1, x2, y2, conf, cls_conf, cls_pred in prediction:
            # Set bounding box color
            color = tuple(cmap_rgb[int(cls_pred) % len(cmap_rgb)])

            # Draw bounding box
            draw.rectangle(((x1, y1), (x2, y2)), outline=color, width=2)

            # Draw label
            text = '{}{:.3f}'.format(class_names[int(cls_pred)], cls_conf.item())
            font = ImageFont.truetype('calibri.ttf', size=12)
            text_width, text_height = font.getsize(text)
            draw.rectangle(((x1, y1), (x1 + text_width, y1 + text_height)), fill=color)
            draw.text((x1, y1), text, fill=(0, 0, 0), font=font)

    # Save result image
    filename = path.split("/")[-1].split(".")[0]
    image.save("{}/{}.jpg".format(args.save_folder, filename))
