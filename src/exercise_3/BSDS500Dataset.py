import os
import random

import numpy as np
import scipy
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class BSDS500Dataset(Dataset):

    num_classes = 2

    def __init__(self, split):
        if split not in ["train", "val", "test"]:
            return

        self.split = split
        self.IMAGES_PATH = f"../../BSDS500/BSDS500/data/images/{split}"
        self.MASKS_PATH = self.IMAGES_PATH.replace("images", "groundTruth")
        self.files = os.listdir(self.IMAGES_PATH)
        self.files = [file for file in self.files if file.endswith(".jpg")]

        self.size = len(self.files)

        # Werte zur Normalisierung stammen von calculate_metrics.py
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=(0.43423752017764605,
                                                                        0.4430257654788147,
                                                                        0.367037588525307),
                                                                  std=(0.24956402339232378,
                                                                       0.23434712400701474,
                                                                       0.24183644572722385))])

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.IMAGES_PATH, self.files[idx]))
        img = img.convert('RGB')

        masks = scipy.io.loadmat(os.path.join(self.MASKS_PATH, self.files[idx].replace(".jpg", ".mat")))
        masks = [masks["groundTruth"][0][i][0][0][1] for i in range(len(masks["groundTruth"][0]))]

        # Sollte das Bild (und damit auch die Maske) vertikal sein, werden sie zu horizontalen Bildern rotiert
        width, height = img.size
        if height > width:
            img = img.rotate(90, expand=True)

            for i in range(len(masks)):
                mask = Image.fromarray(masks[i])
                mask = mask.rotate(90, expand=True)
                mask = np.asarray(mask)
                mask = (np.array(mask) >= 128).astype(int)
                masks[i] = mask

        # Da die beiden Dimensionen der Bilder für das U-Net idealerweise gerade Zahlen sind, werden die letzte Zeile und Spalte abgeschnitten
        img = img.crop((0, 0, 480, 320))
        masks = [mask[:480, :320] for mask in masks]

        img = self.transform(img)

        if self.split == "train":
            masks = masks[random.randint(0, len(masks) - 1)]
            masks = torch.from_numpy(masks).long()
            return img, masks
        elif self.split == "val":
            masks = [torch.from_numpy(mask).long() for mask in masks]
            return img, masks
        else: # self.split == "test"
            masks = [torch.from_numpy(mask).long() for mask in masks]
            return img, masks, self.files[idx].split(".")[0]