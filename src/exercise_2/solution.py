import os
import sys
import argparse
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import convolve
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from exercise_1.solution import compute_metrics, image_from_array

# Constants
BASE_PATH = "../../BSDS500/BSDS500/data/"

# Kernels for edge detection
X_KERNEL = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
Y_KERNEL = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

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
    img = np.array(image)
    gray = np.array(image.convert('L'), dtype=np.float32)
    dx = convolve(gray, X_KERNEL, mode='reflect')
    dy = convolve(gray, Y_KERNEL, mode='reflect')
    magnitude = np.hypot(dx, dy)
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    features = np.stack([r, g, b, gray, dx, dy, magnitude], axis=-1)
    return features.reshape(-1, 7)
    
def select_pixels(features, labels, mode="all_edges"):
    num_pixels = features.shape[0]
    num_edges = np.where(labels == 1)[0].shape[0]    
    total_training_pixels = int(0.1 * num_pixels)
    if mode=="all_edges":
        num_training_edges = num_edges
    elif mode=="half_edges":
        num_training_edges = num_edges // 2
    elif mode=="random":
        num_training_edges = np.random.randint(0, num_edges+1)
    elif mode=="no_edges":
        num_training_edges = 0
    num_training_non_edges = total_training_pixels - num_training_edges
    training_edges = np.random.choice(np.where(labels == 1)[0], size=num_training_edges, replace=False)
    training_non_edges = np.random.choice(np.where(labels == 0)[0], size=num_training_non_edges, replace=False)
    training_pixels = np.concatenate([training_edges, training_non_edges]) 
    training_features = features[training_pixels]
    training_labels = labels[training_pixels]
    pos_weight = torch.tensor([num_training_non_edges / max(num_training_edges, 1)], dtype=torch.float32) 
    return training_pixels, training_features, training_labels, pos_weight, num_training_edges, num_edges
    
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

def plot_loss_history(loss_history):
    average_loss = np.mean([row["loss_history"] for row in loss_history], axis=0)
    max_loss = np.max([row["loss_history"] for row in loss_history], axis=0)
    min_loss = np.min([row["loss_history"] for row in loss_history], axis=0)
    p75_loss = np.percentile([row["loss_history"] for row in loss_history], 75, axis=0)
    p25_loss = np.percentile([row["loss_history"] for row in loss_history], 25, axis=0)
    epochs = range(len(average_loss))
    plt.plot(epochs, average_loss, label="Average Loss")
    plt.plot(epochs, max_loss, label="Max Loss")
    plt.plot(epochs, min_loss, label="Min Loss")
    # Plot shaded region between 25th and 75th percentiles
    plt.fill_between(epochs, p25_loss, p75_loss, color='gray', alpha=0.3, label='25-75% Range')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss History for mode {SELECT_MODE}")
    plt.legend()
    plt.savefig(f"results/{SELECT_MODE}_{N_EPOCHS}_epochs/loss_history.png")
    plt.close()  

def load_all_ground_truths(mat_path):
    data = loadmat(mat_path)["groundTruth"][0]
    gts = []
    for i in range(len(data)):
        gt = data[i][0][0][1]
        gt = image_from_array(gt).rotate(90, expand=True) if gt.shape[0] > gt.shape[1] else np.array(image_from_array(gt))
        if isinstance(gt, np.ndarray) and gt.dtype != np.uint8:
            gt = np.array(gt)
        gt = (np.array(gt, dtype=np.float32) * (255 if np.array(gt).max() <= 1 else 1) >= 128).astype(bool)
        gts.append(gt)
    return gts

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="all_edges", choices=["all_edges", "half_edges", "random", "no_edges"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    args = parser.parse_args()
    SELECT_MODE = args.mode
    N_EPOCHS = args.epochs
    PIXEL_BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    # Load and preprocess data
    images = load_data("test")
    ground_truths = load_ground_truth("test")
    images, ground_truths, image_names = preprocess(images, ground_truths, "test")
    # Extract features
    data = []
    for img, gt, img_name in zip(images, ground_truths, image_names):
        features = extract_features(img)
        labels = gt.flatten()
        data.append({"features": features, "labels": labels, "image_name": img_name})
    # Loop over all images 
    os.makedirs(f"results/{SELECT_MODE}_{N_EPOCHS}_epochs/edge_masks", exist_ok=True)
    summary = []
    loss_history = []   
    print(f"Training {len(data)} images")
    print(f"Using selection mode {SELECT_MODE}, training for {N_EPOCHS} epochs with pixel batch size: {PIXEL_BATCH_SIZE} and learning rate {LEARNING_RATE}")
    for i, image in enumerate(tqdm(data)):
        all_pixels = np.arange(image["features"].shape[0])
        all_features = image["features"]
        all_labels = image["labels"]
        image_name = image["image_name"]
        # Select 10% of pixels and compute pos_weight
        training_pixels, training_features, training_labels, pos_weight, num_training_edges, num_edges = select_pixels(all_features, all_labels, SELECT_MODE)
        # Normalize features
        training_mean = training_features.mean(axis=0)
        training_std = training_features.std(axis=0)
        training_features = (training_features - training_mean) / (training_std + 1e-6)
        # Create training dataset and loader
        training_dataset = ImageDataset(training_features, training_labels)
        training_loader = DataLoader(training_dataset, batch_size=PIXEL_BATCH_SIZE, shuffle=True)
        model = MLP(input_size=training_features.shape[1], name=image_name)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss_history.append({"image_name": image_name, "loss_history": []})
        for epoch in range(N_EPOCHS):
            epoch_loss = 0
            for features, labels in training_loader:
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs.squeeze(-1), labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            loss_history[-1]["loss_history"].append(epoch_loss / len(training_loader))

        # Inference on other 90% of pixels
        inference_pixels = all_pixels[~np.isin(all_pixels, training_pixels)]
        inference_features = (all_features[inference_pixels])
        inference_features = (inference_features - training_mean) / (training_std + 1e-6)
        inference_features = torch.tensor(inference_features)
        inference_labels = torch.tensor(all_labels[inference_pixels])
        with torch.no_grad():
            predictions = model(inference_features)
            pred_bin = (torch.sigmoid(predictions) >= 0.5).int().squeeze(-1)
        # Combine training pixels and inference pixels into a single edge mask
        edge_mask = np.zeros(all_labels.shape)
        edge_mask[training_pixels] = training_labels
        edge_mask[inference_pixels] = pred_bin
        h, w = images[i].size[1], images[i].size[0]
        edge_mask = edge_mask.reshape(h, w)
        ground_truth = all_labels.reshape(h, w)

        # Compute metrics against all annotators
        mat_path = os.path.join(BASE_PATH, "groundTruth", "test", image_name + ".mat")
        all_gts = load_all_ground_truths(mat_path)
        
        row = {"id": image_name}
        row["num_training_edges"] = num_training_edges
        row["num_edges"] = num_edges
        for j, gt in enumerate(all_gts):
            metrics = compute_metrics(edge_mask.astype(bool), gt.astype(bool))
            row[f"annotator_{j}_precision"] = metrics["precision"]
            row[f"annotator_{j}_recall"] = metrics["recall"]
            row[f"annotator_{j}_f1"] = metrics["f1"]
            row[f"annotator_{j}_tp"] = metrics["tp"]
            row[f"annotator_{j}_fp"] = metrics["fp"]
            row[f"annotator_{j}_fn"] = metrics["fn"]

        # Best and average metrics
        all_precisions = [row[f"annotator_{j}_precision"] for j in range(len(all_gts))]
        row["best_precision"] = max(all_precisions)
        row["mean_precision"] = np.mean(all_precisions)

        all_recalls = [row[f"annotator_{j}_recall"] for j in range(len(all_gts))]
        row["best_recall"] = max(all_recalls)
        row["mean_recall"] = np.mean(all_recalls)

        all_f1s = [row[f"annotator_{j}_f1"] for j in range(len(all_gts))]
        row["best_f1"] = max(all_f1s)
        row["mean_f1"] = np.mean(all_f1s)

        summary.append(row)
        # Save edge mask and ground truth in one file every 50 images
        if i % 20 == 0:
            edge_mask_img = image_from_array(edge_mask)
            ground_truth_img = image_from_array(all_labels.reshape(h, w))
            combined_img = create_comparison_image(images[i], edge_mask_img, ground_truth_img)
            combined_img.save(f"results/{SELECT_MODE}_{N_EPOCHS}_epochs/edge_masks/comparison_{image_name}.png")

    # Save summary to csv
    avg_row= {
        "id": "average",
        "num_training_edges": np.mean([float(row["num_training_edges"]) for row in summary]),
        "num_edges": np.mean([float(row["num_edges"]) for row in summary]),
        "best_precision": np.mean([float(row["best_precision"]) for row in summary]),
        "mean_precision": np.mean([float(row["mean_precision"]) for row in summary]),
        "best_recall": np.mean([float(row["best_recall"]) for row in summary]),
        "mean_recall": np.mean([float(row["mean_recall"]) for row in summary]),
        "best_f1": np.mean([float(row["best_f1"]) for row in summary]),
        "mean_f1": np.mean([float(row["mean_f1"]) for row in summary]),
    }
    summary.insert(0, avg_row)
    summary = pd.DataFrame(summary)
    summary.to_csv(f"results/{SELECT_MODE}_{N_EPOCHS}_epochs/summary.csv")
    # Plot loss history 
    plot_loss_history(loss_history)  
