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


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode='bilinear', align_corners=True).squeeze(0)
    return image


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


class ListDataset(Dataset):
    def __init__(self, list_path: str, img_size: int, augmentation: bool, multiscale: bool):
        with open(list_path, 'r') as file:
            self.image_paths = file.readlines()

        self.target_paths = [path.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt')
                                 .replace('JPEGImages', 'labels') for path in self.image_paths]
        self.img_size = img_size
        self.max_objects = 100
        self.augmentation = augmentation
        self.multiscale = multiscale
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

    def __getitem__(self, index):
        # 1. Image
        # ----------------------------------------------------------------------------
        image_path = self.image_paths[index].rstrip()

        transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor()
        ])
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        image = transform(image)

        # 2. Label
        # ----------------------------------------------------------------------------
        target_path = self.target_paths[index].rstrip()
        targets = torch.from_numpy(np.loadtxt(target_path).reshape(-1, 5))

        # Rescale bounding boxes to the YOLO input shape
        targets = utils.utils.rescale_boxes_yolo(targets, original_size, self.img_size)

        # Apply augmentations
        if self.augmentation:
            if np.random.random() < 0.5:
                image, targets = horisontal_flip(image, targets)

        return image, targets

    def __len__(self):
        return len(self.image_paths)

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i

        try:
            targets = torch.cat(targets, 0)
        except RuntimeError:
            targets = None  # No boxes for an image

        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets
