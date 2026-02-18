import os
import sys

from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import convolve
from tqdm import tqdm
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from exercise_1.solution import compute_metrics, compute_confusion_matrix
from exercise_1.solution import image_from_array
########################################################################################################################
# Constants/Hyperparameters
BASE_PATH = "../../BSDS500/BSDS500/data/"
PIXEL_BATCH_SIZE = 2048
N_EPOCHS = 10
LEARNING_RATE = 0.001
########################################################################################################################
class MLP(nn.Module):
    def __init__(self, input_size, name):
        super(MLP, self).__init__()
        self.name = "MLP_" + name
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ImageDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features = self.features[idx]
        labels = self.labels[idx]
        return features, labels

def load_data(split):
    images = sorted(os.listdir(os.path.join(BASE_PATH, "images", split)))
    images = [image for image in images if image.lower().endswith((".jpg", ".jpeg"))]
    count = len(images)
    print(f"Loaded {count} images from {split}.")
    return images

def load_ground_truth(split):
    ground_truths = sorted(os.listdir(os.path.join(BASE_PATH, "groundTruth", split)))
    ground_truths = [file for file in ground_truths if file.lower().endswith((".mat"))]
    count = len(ground_truths)
    print(f"Loaded {count} ground truth files from {split}.")
    return ground_truths

def preprocess(images, ground_truths, split):
    image_names = [image.replace(".jpg", "") for image in images]
    ground_truth_names = [ground_truth.replace(".mat", "") for ground_truth in ground_truths]
    assert image_names == ground_truth_names, "Image and ground truth names do not match"
    images = [Image.open(os.path.join(BASE_PATH, "images", split, image)) for image in images]
    ground_truths = [loadmat(os.path.join(BASE_PATH, "groundTruth", split, file))["groundTruth"][0][0][0][0][1] for file in ground_truths]
    images = [img.rotate(90, expand=True) if img.size[1] > img.size[0] else img for img in images]
    ground_truths = [image_from_array(gt).rotate(90, expand=True) if gt.shape[0] > gt.shape[1] else gt for gt in ground_truths]
    ground_truths = [(np.array(mask, dtype=np.float32) * (255 if np.array(mask).max() <= 1 else 1) >= 128).astype(np.float32) for mask in ground_truths]
    return images, ground_truths, image_names

def extract_features(image):
    x_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    y_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    
    img = np.array(image)
    gray = np.array(image.convert('L'), dtype=np.float32)
    dx = convolve(gray, x_kernel, mode='reflect')
    dy = convolve(gray, y_kernel, mode='reflect')
    magnitude = np.hypot(dx, dy)
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    features = np.stack([r, g, b, gray, dx, dy, magnitude], axis=-1)
    return features.reshape(-1, 7)
    
def select_pixels(features, labels):
    num_pixels = features.shape[0]
    num_edges = np.where(labels == 1)[0].shape[0]    
    total_training_pixels = int(0.1 * num_pixels)
    if num_edges < total_training_pixels // 2:
        num_training_edges = num_edges
        num_training_non_edges = total_training_pixels - num_training_edges
    else:
        num_training_edges = total_training_pixels // 2
        num_training_non_edges = total_training_pixels - num_training_edges
    edges = np.random.choice(np.where(labels == 1)[0], size=num_training_edges, replace=False)
    non_edges = np.random.choice(np.where(labels == 0)[0], size=num_training_non_edges, replace=False)
    training_pixels = np.concatenate([edges, non_edges])
    training_features = features[training_pixels]
    training_labels = labels[training_pixels]
    pos_weight = torch.tensor([num_training_non_edges / max(num_training_edges, 1)], dtype=torch.float32) 
    return training_pixels, training_features, training_labels, pos_weight

def normalize_features(features):
    return (features - features.mean(axis=0)) / features.std(axis=0)
    
def create_comparison_image(original, edge_mask, ground_truth):
    # Resize to common height (optional: preserves aspect ratio)
    def resize_to_height(img, h):
        ratio = h / img.height
        new_w = int(img.width * ratio)
        return img.resize((new_w, h), Image.Resampling.LANCZOS)
    
    h = 321
    orig_resized = resize_to_height(original.convert("RGB"), h)
    gt_resized = resize_to_height(ground_truth.convert("RGB"), h)
    edge_resized = resize_to_height(edge_mask.convert("RGB"), h)  # grayscale -> RGB
    
    total_width = orig_resized.width + edge_resized.width + gt_resized.width
    combined = Image.new("RGB", (total_width, h), color=(255, 255, 255))
    
    x = 0
    for img in [orig_resized, gt_resized, edge_resized]:
        combined.paste(img, (x, 0))
        x += img.width
    
    return combined

if __name__ == "__main__":
    # Training
    train_images = load_data("train")
    train_ground_truth = load_ground_truth("train")
    val_images = load_data("val")
    val_ground_truth = load_ground_truth("val")
    test_images = load_data("test")
    test_ground_truth = load_ground_truth("test")
    train_images, train_ground_truth, image_names = preprocess(train_images, train_ground_truth, "train")
    val_images, val_ground_truth, val_image_names = preprocess(val_images, val_ground_truth, "val")
    test_images, test_ground_truth, test_image_names = preprocess(test_images, test_ground_truth, "test")
    images = train_images + val_images + test_images
    ground_truths = train_ground_truth + val_ground_truth + test_ground_truth
    image_names = image_names + val_image_names + test_image_names
    # Extract features
    training_data = []
    for img, gt, img_name in zip(images, ground_truths, image_names):
        features = extract_features(img)
        labels = gt.flatten()
        training_data.append({"features": features, "labels": labels, "image_name": img_name})
    # Loop over all images 
    os.makedirs("edge_masks", exist_ok=True)
    summary = []
    for i, image in enumerate(tqdm(training_data)):
        all_pixels = np.arange(image["features"].shape[0])
        all_features = image["features"]
        all_labels = image["labels"]
        image_name = image["image_name"]
        # Select 10% of pixels and compute pos_weight
        training_pixels, training_features, training_labels, pos_weight = select_pixels(all_features, all_labels)
        # Normalize features
        training_features = normalize_features(training_features)
        training_dataset = ImageDataset(training_features, training_labels)
        training_loader = DataLoader(training_dataset, batch_size=1024, shuffle=True)
        for epoch in range(N_EPOCHS):
            for features, labels in training_loader:
                model = MLP(input_size=features.shape[1], name=image_name)
                optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs.squeeze(-1), labels)
                loss.backward()
                optimizer.step()

        # Inference on other 90% of pixels
        inference_pixels = all_pixels[~np.isin(all_pixels, training_pixels)]
        inference_features = torch.tensor(all_features[inference_pixels])
        inference_labels = torch.tensor(all_labels[inference_pixels])
        predictions = model(inference_features)
        pred_bin = (torch.sigmoid(predictions) >= 0.5).int().squeeze(-1)

        # Combine training pixels and inference pixels into a single edge mask
        edge_mask = np.zeros(all_labels.shape)
        edge_mask[training_pixels] = training_labels
        edge_mask[inference_pixels] = pred_bin
        h, w = images[i].size[1], images[i].size[0]
        edge_mask = edge_mask.reshape(h, w)
        ground_truth = all_labels.reshape(h, w)

        # Compute metrics
        row = {"id": image_name}
        metrics = compute_metrics(edge_mask.astype(bool), ground_truth.astype(bool))
        row["precision"] = metrics["precision"]
        row["recall"] = metrics["recall"]
        row["f1"] = metrics["f1"]
        row["tp"] = metrics["tp"]
        row["fp"] = metrics["fp"]
        row["fn"] = metrics["fn"]
        summary.append(row)
        # Save edge mask and ground truth in one file
        # edge_mask_img = image_from_array(edge_mask)
        # ground_truth_img = image_from_array(all_labels.reshape(h, w))
        # combined_img = create_comparison_image(images[i], edge_mask_img, ground_truth_img)
        # combined_img.save(f"edge_masks/comparison_{image_name}.png")

    summary = pd.DataFrame(summary)
    summary.to_csv("summary.csv")