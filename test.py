import argparse
import csv
import os
import time

import torch
import torch.utils.data
import numpy as np
import tqdm

import model.yolov3
import model.yolov3_proposed
import utils.datasets
import utils.utils


def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size, num_workers, device):
    model.eval()

    # 데이터셋, 데이터로더 설정
    dataset = utils.datasets.YOLODataset(path, img_size=img_size, augmentation=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=num_workers,
                                             collate_fn=dataset.collate_fn)

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc='Detecting objects', leave=False)):

        if targets is None:
            continue

        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = utils.utils.xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        with torch.no_grad():
            imgs = imgs.to(device)
            outputs = model(imgs)
            outputs = utils.utils.non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

        sample_metrics += utils.utils.get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    # Concatenate sample statistics
    if len(sample_metrics) == 0:
        true_positives, pred_scores, pred_labels = np.array([]), np.array([]), np.array([])
    else:
        true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = utils.utils.ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32, help="size of each image batch")
    parser.add_argument("--data_config", type=str, default="config/voc.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, default="weights/yolov3_voc.pth",
                        help="path to pretrained weights file")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.001, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou threshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    args = parser.parse_args()
    print(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 데이터셋 설정값을 가져오기
    data_config = utils.utils.parse_data_config(args.data_config)
    valid_path = data_config['valid']
    num_classes = int(data_config['classes'])
    class_names = utils.utils.load_classes(data_config['names'])

    # 모델 준비하기
    model = model.yolov3.YOLOv3(args.img_size, num_classes).to(device)
    if args.pretrained_weights.endswith('.pth'):
        model.load_state_dict(torch.load(args.pretrained_weights))
    else:
        model.load_darknet_weights(args.pretrained_weights)

    # 검증 데이터셋으로 모델을 평가
    precision, recall, AP, f1, ap_class = evaluate(
        model,
        path=valid_path,
        iou_thres=args.iou_thres,
        conf_thres=args.conf_thres,
        nms_thres=args.nms_thres,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.n_cpu,
        device=device
    )

    # AP와 mAP 출력
    print('Average Precisions:')
    for i, class_num in enumerate(ap_class):
        print('\tClass {} ({}) - AP: {:.02f}'.format(class_num, class_names[class_num], AP[i] * 100))
    print('mAP: {:.02f}'.format(AP.mean() * 100))

    # AP와 mAP를 csv 파일로 저장
    os.makedirs('csv', exist_ok=True)
    now = time.strftime('%y%m%d_%H%M%S', time.localtime(time.time()))
    with open('csv/test{}.csv'.format(now), mode='w') as f:
        writer = csv.writer(f, delimiter=',')

        writer.writerow(['Class Number', 'Class Name', 'AP'])
        for i, class_num in enumerate(ap_class):
            writer.writerow([class_num, class_names[class_num], AP[i] * 100])
        writer.writerow(['mAP', AP.mean() * 100, ' '])
    print('Saved result csv file.')
