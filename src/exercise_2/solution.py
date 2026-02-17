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
    ground_truths = [image_from_array(gt).rotate(90, expand=True) if gt.shape[1] > gt.shape[0] else gt for gt in ground_truths]
    ground_truths = [(np.array(mask) >= 128).astype(int).astype(np.float32) for mask in ground_truths]
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
    if num_edges < total_training_pixels / 2:
        num_training_edges = num_edges
        num_training_non_edges = total_training_pixels - num_training_edges
    else:
        num_training_edges = total_training_pixels / 2
        num_training_non_edges = total_training_pixels - num_training_edges
    edges = np.random.choice(np.where(labels == 1)[0], size=num_training_edges, replace=False)
    non_edges = np.random.choice(np.where(labels == 0)[0], size=num_training_non_edges, replace=False)
    pixels = np.concatenate([edges, non_edges])
    training_features = features[pixels]
    training_labels = labels[pixels]
    pos_weight = torch.tensor([num_training_non_edges / max(num_training_edges, 1)], dtype=torch.float32) 
    return training_features, training_labels, pos_weight

def normalize_features(features):
    return (features - features.mean(axis=0)) / features.std(axis=0)
    

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
    for i, image in enumerate(tqdm(training_data)):
        features = image["features"]
        labels = image["labels"]
        image_name = image["image_name"]
        # Select 10% of pixels and compute pos_weight
        training_features, training_labels, pos_weight = select_pixels(features, labels)
        # Normalize features
        training_features = normalize_features(training_features)
        training_dataset = ImageDataset(training_features, training_labels)
        training_loader = DataLoader(training_dataset, batch_size=1024, shuffle=True)
        loss_total = 0
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
                loss_total += loss.item()
        print(f"Average loss: {loss_total / len(training_loader)}")