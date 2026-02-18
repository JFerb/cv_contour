import os

from PIL import Image, ImageFilter
import numpy as np
import scipy
from scipy.ndimage import convolve, binary_dilation
import pandas as pd

########################################################################################################################

# Constants used later for indexing

# Ground truth data
SEGMENTATION = 0
BOUNDARIES   = 1

# Sobel directions
H  = 0    # Horizontal
V  = 1    # Vertical
DR = 2    # Diagonal, rising
DF = 3    # Diagonal, falling

########################################################################################################################

# Smooths an image using a Gaussian kernel
def gaussian_smooth(img, std):
    return img.filter(ImageFilter.GaussianBlur(radius=std))

########################################################################################################################

# Calculates edge strengths and directions using the Sobel operator on an image given as an array
def sobel(img):
    img = np.asarray(img, dtype=np.float32)

    # Define and apply kernels
    x_kernel = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]], dtype=np.float32)

    y_kernel = np.array([[-1, -2, -1],
                         [0, 0, 0],
                         [1, 2, 1]], dtype=np.float32)

    dx = convolve(img, x_kernel, mode='reflect')
    dy = convolve(img, y_kernel, mode='reflect')

    # Calculate edge strength and direction for every pixel
    strengths = np.hypot(dx, dy)
    angles = np.arctan2(dy, dx) % np.pi

    # Quantization of directions
    directions = np.zeros_like(angles, dtype=int)

    # Remember: The angles describe gradient direction, but we need edge direction which is orthogonal to it
    directions[(angles < np.pi / 8) | (angles >= 7 * np.pi / 8)] = V       # Gradient: H, 0°±22.5°
    directions[(angles >= np.pi / 8) & (angles < 3 * np.pi / 8)] = DF      # Gradient: DR, 45°±22.5°
    directions[(angles >= 3 * np.pi / 8) & (angles < 5 * np.pi / 8)] = H   # Gradient: V, 90°±22.5°
    directions[(angles >= 5 * np.pi / 8) & (angles < 7 * np.pi / 8)] = DR  # Gradient: DF, 135°±22.5°

    return strengths, directions

########################################################################################################################

# Applies Non-Maximum Suppression on the edge strength of an image using their directions given as arrays
def non_maximum_suppression(strengths, directions):
    h, w = strengths.shape
    result = strengths.copy()

    # Offsets for all directions
    offsets = {
        H:  [(-1,  0), (1,  0)],
        V:  [( 0, -1), (0,  1)],
        DR: [(-1, -1), (1,  1)],
        DF: [(-1,  1), (1, -1)]
    }

    for y in range(h):
        for x in range(w):
            direction = directions[y, x]
            strength = strengths[y, x]

            for dy, dx in offsets[direction]:
                ny, nx = y + dy, x + dx
                # Check if neighbor is in bounds
                if 0 <= ny < h and 0 <= nx < w:
                    if strength < strengths[ny, nx]:
                        result[y, x] = 0
                        break

    return result

########################################################################################################################

def eight_neighborhood(y, x, ymax, xmax):
    candidates = [(y - 1, x - 1),
                 (y - 1, x),
                 (y - 1, x + 1),
                 (y, x - 1),
                 (y, x + 1),
                 (y + 1, x - 1),
                 (y + 1, x),
                 (y + 1, x + 1)]

    neighbors = []
    for w, v in candidates:
        if (0 <= w < ymax) and (0 <= v < xmax):
            neighbors.append((w, v))

    return neighbors

########################################################################################################################

# Compute hysteresis iteratively until convergence
def hysteresis(img, upper, lower):
    h, w = img.shape
    result = (img >= upper).astype(int)

    while True:
        prev = result.copy()

        for y in range(h):
            for x in range(w):
                if result[y, x] == 1:
                    for ny, nx in eight_neighborhood(y, x, h, w):
                        if result[ny, nx] == 0 and img[ny, nx] >= lower:
                            result[ny, nx] = 1

        if np.array_equal(result, prev):
            break

    return result

########################################################################################################################

# Apply Canny edge detector on an image represented as an array
def canny(img, upper, lower, std=None):
    if std is not None:
        # Smooth image, convert to grayscale, and turn into array
        img = gaussian_smooth(img, std)
        img = img.convert('L')
    else:
        # Convert to grayscale and turn into array
        img = img.convert('L')

    strengths, directions = sobel(img)
    img = non_maximum_suppression(strengths, directions)
    img = hysteresis(img, upper, lower)

    return img


########################################################################################################################

# Read ground truth data for an image and convert it to an array
def read_ground_truth(name):
    name = name.replace("images", "groundTruth")
    name = name.replace(".jpg", ".mat")

    data = scipy.io.loadmat(name)
    data = data["groundTruth"][0]

    results = []

    for annotator in range(len(data)):
        results.append(data[annotator][0][0])

    return results

########################################################################################################################

# Takes a two-dimensional array and turns it into a grayscale image
def image_from_array(array):
    array = array.astype(int)
    img = (array - array.min()) / (array.max() - array.min()) * 255
    img = img.astype(np.uint8)
    img = Image.fromarray(img, mode='L')

    return img

########################################################################################################################

# Compute different performance metrics for model predictions with tolerance for pixel overlap

def compute_confusion_matrix(predicted, ground_truth, tolerance):

    # Dilate both boundary masks with a tolerance
    structure = np.ones((2 * tolerance + 1, 2 * tolerance + 1))

    pred_dilated = binary_dilation(predicted.astype(bool), structure=structure)
    gt_dilated = binary_dilation(ground_truth.astype(bool), structure=structure)

    tp = np.sum(predicted & gt_dilated)
    fp = np.sum(predicted & ~gt_dilated)
    fn = np.sum(ground_truth & ~pred_dilated)

    return tp, fp, fn

def compute_metrics(predicted, ground_truth, tolerance=2):
    tp, fp, fn = compute_confusion_matrix(predicted, ground_truth, tolerance=tolerance)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn
    }

########################################################################################################################

if __name__ == "__main__":
    # Hyperparameter
    UPPER = 200
    LOWER = 100
    SIGMA = 1

    BASE_PATH = "../../BSDS500/BSDS500/data/images/test"
    files = os.listdir(BASE_PATH)
    count = sum(1 for file in files if file.lower().endswith(".jpg"))

    summary = []
    i = 1
    for file in files:
        if file.lower().endswith(".jpg"):
            print(f"Verarbeite Bild {i} von {count}.")
            i += 1

            image_id = os.path.splitext(file)[0]

            path = os.path.join(BASE_PATH, file)
            img = Image.open(path)
            gt = read_ground_truth(path)
            edge_masks = [annotator[BOUNDARIES] for annotator in gt]

            # Most images in the dataset are horizontal. Make the vertical ones horizontal too for consistency.
            width, height = img.size
            if height > width:
                img = img.rotate(90, expand=True)

                # Also rotate ground truths
                edge_masks = [image_from_array(mask).rotate(90, expand=True) for mask in edge_masks]
                edge_masks = [(np.array(mask) > 0).astype(int) for mask in edge_masks]

            result = canny(img, UPPER, LOWER, SIGMA)

            row = {"id": image_id}

            for idx, edge_mask in enumerate(edge_masks):
                metrics = compute_metrics(result, edge_mask)

                # Save metrics for all annotators
                row[f"annotator_{idx}_precision"] = metrics["precision"]
                row[f"annotator_{idx}_recall"] = metrics["recall"]
                row[f"annotator_{idx}_f1"] = metrics["f1"]
                row[f"annotator_{idx}_tp"] = metrics["tp"]
                row[f"annotator_{idx}_fp"] = metrics["fp"]
                row[f"annotator_{idx}_fn"] = metrics["fn"]

            # Best and average metrics
            all_precisions = [row[f"annotator_{idx}_precision"] for idx in range(len(edge_masks))]
            row["best_precision"] = max(all_precisions)
            row["mean_precision"] = np.mean(all_precisions)

            all_recalls = [row[f"annotator_{idx}_recall"] for idx in range(len(edge_masks))]
            row["best_recall"] = max(all_recalls)
            row["mean_recall"] = np.mean(all_recalls)

            all_f1s = [row[f"annotator_{idx}_f1"] for idx in range(len(edge_masks))]
            row["best_f1"] = max(all_f1s)
            row["mean_f1"] = np.mean(all_f1s)

            summary.append(row)

    summary = pd.DataFrame(summary)
    cols_to_front = [col for col in summary.columns if "best" in col or "mean" in col]
    cols_to_front.insert(0, "id")
    summary = summary[cols_to_front + [col for col in summary.columns if col not in cols_to_front]]
    summary.to_csv(f"summary_{UPPER}_{LOWER}_{SIGMA}_final.csv")