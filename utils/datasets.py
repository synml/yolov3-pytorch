import glob
import os
import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image


def horisontal_flip(images, targets):
    images = torch.flip(images, [-1])
    targets[:, 2] = 1 - targets[:, 2]
    return images, targets


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    
    # 너비와 높이의 차
    difference = abs(h - w)

    # (top, bottom) padding or (left, right) padding
    if h <= w:
        top = difference // 2
        bottom = difference - difference // 2
        pad = [0, 0, top, bottom]
    else:
        left = difference // 2
        right = difference - difference // 2
        pad = [left, right, 0, 0]

    # Add padding
    img = F.pad(img, pad, mode='constant', value=pad_value)
    return img, pad


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
        img = transform(Image.open(image_path).convert('RGB'))
        return image_path, img

    def __len__(self):
        return len(self.image_paths)


class ListDataset(Dataset):
    def __init__(self, list_path: str, img_size: int, augmentation: bool, multiscale: bool):
        with open(list_path, 'r') as file:
            self.image_paths = file.readlines()

        self.label_paths = [path.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt')
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
        # ----------------------------------------------------------------
        image_path = self.image_paths[index].rstrip()

        transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor()
        ])
        image = transform(Image.open(image_path).convert('RGB'))

        _, h_factor, w_factor = image.shape
        # Pad to square resolution
        image, pad = pad_to_square(image, 0)
        _, padded_h, padded_w = image.shape

        #  2. Label
        # ----------------------------------------------------------------
        label_path = self.label_paths[index % len(self.image_paths)].rstrip()

        targets = None
        if os.path.exists(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
            # Extract coordinates for original image (x, y, w, h) -> (x1, y1, x2, y2)
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]
            # Returns (x, y, w, h)
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes

        # Apply augmentations
        if self.augmentation:
            if np.random.random() < 0.5:
                image, targets = horisontal_flip(image, targets)

        return image_path, image, targets

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
