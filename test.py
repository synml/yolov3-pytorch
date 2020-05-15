import argparse
import csv
import os
import time
import tqdm

import torch
from torch.utils.data import DataLoader

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *


def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size, num_workers):
    model.eval()

    # Get dataloader
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=dataset.collate_fn
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects", leave=False)):

        if targets is None:
            continue

        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = imgs.to(device)

        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    # Concatenate sample statistics
    if len(sample_metrics) == 0:
        true_positives, pred_scores, pred_labels = np.array([]), np.array([]), np.array([])
    else:
        true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3-voc.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/voc.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3_ckpt_99.pth", help="path to weights file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.001, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou threshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    args = parser.parse_args()
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = parse_data_config(args.data_config)
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(args.model_def, img_size=args.img_size).to(device)
    if args.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(args.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(args.weights_path))

    print("Compute mAP...")

    precision, recall, AP, f1, ap_class = evaluate(
        model,
        path=valid_path,
        iou_thres=args.iou_thres,
        conf_thres=args.conf_thres,
        nms_thres=args.nms_thres,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.n_cpu
    )

    # Print AP and mAP.
    print("Average Precisions:")
    for i, class_num in enumerate(ap_class):
        print('\tClass {} ({}) - AP: {:.02f}'.format(class_num, class_names[class_num], AP[i] * 100))
    print('mAP: {:.02f}'.format(AP.mean() * 100))

    # Saving AP and mAP to csv file.
    os.makedirs('csv', exist_ok=True)

    now = time.strftime('%y%m%d_%H%M%S', time.localtime(time.time()))
    with open('csv/test{}.csv'.format(now), mode='w') as f:
        writer = csv.writer(f, delimiter=',')

        writer.writerow(['Class Number', 'Class Name', 'AP'])
        for i, class_num in enumerate(ap_class):
            writer.writerow([class_num, class_names[class_num], AP[i] * 100])
        writer.writerow(['mAP', AP.mean() * 100, ' '])
    print('Saved result csv file.')
