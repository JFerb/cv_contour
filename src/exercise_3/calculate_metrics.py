import os

import numpy as np
import scipy
from PIL import Image

BASE_PATH = f"../../BSDS500/BSDS500/data/images/train"
MASKS_PATH = BASE_PATH.replace("images", "groundTruth")
files = os.listdir(BASE_PATH)
files = [file for file in files if file.endswith(".jpg")]
count = len(files)

sums = np.zeros(3, dtype=np.float64)
square_sums = np.zeros(3, dtype=np.float64)
pixel_count = 0

positive_fraction = []

i = 1
for file in files:
    print(f"Verarbeite Bild {i} von {count}.")
    i += 1

    path = os.path.join(BASE_PATH, file)
    img = Image.open(path)

    # Enforce RGB type
    img = img.convert('RGB')
    img = np.asarray(img, dtype=np.float64)
    # Normalize
    img = img / 255.0
    # Get raw pixels per channel
    img = img.reshape(-1, 3)

    sums += img.sum(axis=0)
    square_sums += (img ** 2).sum(axis=0)
    pixel_count += img.shape[0]

    path = os.path.join(MASKS_PATH, file.replace(".jpg", ".mat"))
    masks = scipy.io.loadmat(path)
    masks = [masks["groundTruth"][0][i][0][0][1] for i in range(len(masks["groundTruth"][0]))]

    # Berechne durchschnittlichen Anteil positiver Pixel über alle Annotatoren dieses Bildes
    avg_positive = []
    for mask in masks:
        mask = (mask > 0).astype(int)
        frac = np.sum(mask) / mask.size
        avg_positive.append(frac)

    avg_positive = np.mean(avg_positive)
    positive_fraction.append(avg_positive)

means = sums / pixel_count
square_means = square_sums / pixel_count
variances = square_means - means ** 2
stds = np.sqrt(np.maximum(variances, 0))

avg_positive = np.mean(positive_fraction)
avg_negative = 1 - avg_positive

print("Means:")
print(f"R: {means[0]}, G: {means[1]}, B: {means[2]}")
print()
print("Standard Deviations:")
print(f"R: {stds[0]}, G: {stds[1]}, B: {stds[2]}")

print()

print(f"Durchschnittlich Positiv: {avg_positive:.6f} ({avg_positive*100:.2f}%)")
print(f"Durchschnittlich Negativ: {avg_negative:.6f} ({avg_negative*100:.2f}%)")
print(f"Gewichtungsverhältnis:    {(avg_negative / avg_positive):.4f}")