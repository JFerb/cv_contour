import os

import numpy as np
from PIL import Image

BASE_PATH = f"../../BSDS500/BSDS500/data/images/train"
files = os.listdir(BASE_PATH)
count = sum(1 for file in files if file.lower().endswith(".jpg"))

sums = np.zeros(3, dtype=np.float64)
square_sums = np.zeros(3, dtype=np.float64)
pixel_count = 0

i = 1
for file in files:
    if file.lower().endswith(".jpg"):
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

means = sums / pixel_count
square_means = square_sums / pixel_count
variances = square_means - means ** 2
stds = np.sqrt(np.maximum(variances, 0))

print("Means:")
print(f"R: {means[0]}, G: {means[1]}, B: {means[2]}")
print()
print("Standard Deviations:")
print(f"R: {stds[0]}, G: {stds[1]}, B: {stds[2]}")