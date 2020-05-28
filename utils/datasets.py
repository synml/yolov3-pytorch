import glob
import os
import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

import utils.utils


def horisontal_flip(images, targets):
    images = torch.flip(images, [-1])
    targets[:, 2] = 1 - targets[:, 2]
    return images, targets


class ImageFolder(Dataset):
    def __init__(self, folder_path: str, img_size: int):
        self.image_paths = sorted(glob.glob('{}/*.*'.format(folder_path)))
        self.img_size = img_size

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        # Replace Windows path separator to Linux path separator
        image_path = image_path.replace('\\', '/')

        transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor()
        ])
        image = transform(Image.open(image_path).convert('RGB'))
        return image_path, image

    def __len__(self):
        return len(self.image_paths)


class YOLODataset(Dataset):
    def __init__(self, list_path: str, img_size: int, rescale_bbox: bool, augmentation: bool, multiscale: bool):
        with open(list_path, 'r') as f:
            self.image_paths = f.readlines()
        for i in range(len(self.image_paths)):
            self.image_paths[i] = self.image_paths[i].strip()

        self.target_paths = [path.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt')
                                 .replace('JPEGImages', 'labels') for path in self.image_paths]
        self.img_size = img_size
        self.rescale_bbox = rescale_bbox
        self.augmentation = augmentation
        self.multiscale = multiscale

    def __getitem__(self, index):
        # 1. Image
        # ----------------------------------------------------------------------------
        image_path = self.image_paths[index]

        transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor()
        ])
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        image = transform(image)

        # 2. Target bounding box
        # ----------------------------------------------------------------------------
        target_path = self.target_paths[index]
        targets = torch.from_numpy(np.loadtxt(target_path).reshape(-1, 5))

        # Rescale bounding boxes to the YOLO input shape
        if self.rescale_bbox:
            targets = utils.utils.rescale_boxes_yolo(targets, original_size, self.img_size)

        # Apply augmentations
        if self.augmentation:
            if np.random.random() < 0.5:
                image, targets = horisontal_flip(image, targets)

        return image, targets

    def __len__(self):
        return len(self.image_paths)

    def collate_fn(self, batch):
        imgs, targets = list(zip(*batch))

        batch_size = len(imgs)
        channel = imgs[0].shape[0]
        row = imgs[0].shape[1]
        col = imgs[0].shape[2]
        imgs_batch = torch.zeros(batch_size, channel, row, col)
        for i in range(batch_size):
            imgs_batch[i] = imgs[i]

        targets = torch.cat(targets, 0)
        return imgs_batch, targets
