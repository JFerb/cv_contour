import json
import os
import random
import sys

import numpy as np
import scipy
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms

from solution import UNet
from Utils import GetBestDevice, InitRNG

import warnings
warnings.filterwarnings("ignore")



if len(sys.argv) != 2:
    sys.exit("Bitte Nummer des Run angeben.")

HYPERPARAMETERS = {}

run_path = os.path.join("models", f"run-{sys.argv[1]}")

with open(os.path.join(run_path, "hyperparameters.txt")) as f:
    for line in f:
        if line == "\n":
            continue

        line = line.replace('\n', '')
        line = line.replace(' ', '')
        param, _, val = line.partition('=')

        try:
            HYPERPARAMETERS[param] = float(val) if '.' in val else int(val)
        except ValueError:
            HYPERPARAMETERS[param] = val

# InitRNG(HYPERPARAMETERS["SEED"])

model = UNet(HYPERPARAMETERS)

with open(os.path.join(run_path, "metrics.json")) as f:
    epoch = json.load(f)["best_epoch"]

best_model = torch.load(os.path.join(run_path, f"best_model_epoch_{epoch}.pt"))
model.load_state_dict(best_model.state_dict())
model = model.to(GetBestDevice())
model.eval()

IMAGES_PATH = "../../BSDS500/BSDS500/data/images/test"
MASKS_PATH = IMAGES_PATH.replace("images", "groundTruth")

files = os.listdir(IMAGES_PATH)
files = [file for file in files if file.endswith(".jpg")]
file = random.choice(files)

img = Image.open(os.path.join(IMAGES_PATH, file))
masks = scipy.io.loadmat(os.path.join(MASKS_PATH, file.replace(".jpg", ".mat")))
masks = [masks["groundTruth"][0][i][0][0][1] for i in range(len(masks["groundTruth"][0]))]

width, height = img.size
if height > width:
    img = img.rotate(90, expand=True)

    for i in range(len(masks)):
        mask = Image.fromarray(masks[i])
        mask = mask.rotate(90, expand=True)
        mask = np.asarray(mask)
        mask = (np.array(mask) > 0).astype(int)
        masks[i] = mask

img = img.crop((0, 0, 480, 320))
masks = [mask[:320, :480] for mask in masks]

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=(0.43423752017764605,
                                                           0.4430257654788147,
                                                           0.367037588525307),
                                                     std=(0.24956402339232378,
                                                          0.23434712400701474,
                                                          0.24183644572722385))])
input_img = transform(img)
input_img = input_img.unsqueeze(0).to(GetBestDevice())
output = model.predict(input_img)

n_gts = len(masks)

rows = 2
cols = max(1, 1 + (n_gts + 1) // 2)
# 1 Spalte für Original/Output + genug Spalten für GTs (verteilt auf 2 Zeilen)

fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 6))

# Falls nur 1 Spalte existiert, axes korrekt dimensionieren
if cols == 1:
    axes = np.array(axes).reshape(2, 1)

# Oben links: Original
axes[0, 0].imshow(np.asarray(img))
axes[0, 0].set_title("Original")
axes[0, 0].axis("off")

# Unten links: Model Output
axes[1, 0].imshow(output, cmap="gray")
axes[1, 0].set_title("Model Output")
axes[1, 0].axis("off")

# Ground Truth Masken verteilen (ab Spalte 1)
for i, mask in enumerate(masks):
    col = 1 + (i // 2)
    row = i % 2
    axes[row, col].imshow(mask, cmap="gray")
    axes[row, col].set_title(f"GT {i+1}")
    axes[row, col].axis("off")

# Leere Felder ausblenden
for col in range(1, cols):
    for row in range(2):
        idx = 2*(col-1) + row
        if idx >= n_gts:
            axes[row, col].axis("off")

plt.tight_layout()
plt.show()