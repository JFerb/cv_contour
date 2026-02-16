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
IMAGE_BATCH_SIZE = 8
PIXEL_BATCH_SIZE = 2048
N_EPOCHS = 10
LEARNING_RATE = 0.0001
########################################################################################################################
class BSDS500Dataset(Dataset):
    def __init__(self, image_data):
        self.image_data = image_data

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        features, labels = self.image_data[idx]
        return features, labels
    
class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

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
    images = [Image.open(os.path.join(BASE_PATH, "images", split, image)) for image in images]
    ground_truths = [loadmat(os.path.join(BASE_PATH, "groundTruth", split, file))["groundTruth"][0][0][0][0][1] for file in ground_truths]
    images = [img.rotate(90, expand=True) if img.size[1] > img.size[0] else img for img in images]
    ground_truths = [image_from_array(gt).rotate(90, expand=True) if gt.shape[1] > gt.shape[0] else gt for gt in ground_truths]
    ground_truths = [(np.array(mask) >= 128).astype(int) for mask in ground_truths]
    return images, ground_truths

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
    
if __name__ == "__main__":
    # Training
    train_images = load_data("train")
    train_ground_truth = load_ground_truth("train")
    train_images, train_ground_truth = preprocess(train_images, train_ground_truth, "train")
    # Extract features
    training_data = []
    for img, gt in zip(train_images, train_ground_truth):
        features = extract_features(img)
        labels = gt.flatten()
        training_data.append((features, labels))
    # Normalize features
    all_features = np.vstack([f for f,_ in training_data])
    mean = np.mean(all_features, axis=0)
    std = np.std(all_features, axis=0)
    training_data = [((f - mean) / (std + 1e-8), l) for f, l in training_data]
    print("Extracted features for", len(training_data), "images, each with", training_data[0][0].shape[0], "pixels.")
    print("Each pixel has", training_data[0][0].shape[1], "features.")
    # Create dataset and loader
    train_dataset = BSDS500Dataset(training_data)
    train_loader = DataLoader(train_dataset, batch_size=IMAGE_BATCH_SIZE, shuffle=True)
    model = MLP(training_data[0][0].shape[1])
    # Create criterion and optimizer
    all_labels = np.concatenate([labels for _, labels in training_data])
    num_positives = all_labels.sum()
    num_negatives = len(all_labels) - num_positives
    pos_weight = torch.tensor([num_negatives / (2 * max(num_positives, 1))], dtype=torch.float32)
    print(f"Class balance: {num_positives:.0f} edges, {num_negatives:.0f} non-edges, pos_weight={pos_weight.item():.2f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Train model
    for epoch in range(N_EPOCHS):
        for features, labels in tqdm(train_loader):
            features_flat = features.reshape(-1, features.shape[-1]) # (H*W, 7)
            labels_flat = labels.reshape(-1).float().unsqueeze(-1) # (H*W,)
            num_pixels = features.shape[0] * features.shape[1] # H*W
            loss_sum = 0
            for i in range(0,num_pixels, PIXEL_BATCH_SIZE):
                batch_f = features_flat[i:i+PIXEL_BATCH_SIZE] # (B*H*W, 7)
                batch_l = labels_flat[i:i+PIXEL_BATCH_SIZE] # (B*H*W,)
                output = model.forward(batch_f)
                optimizer.zero_grad()
                loss = criterion(output, batch_l)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
            print(f"Epoch {epoch + 1} / {N_EPOCHS}, Batch Loss: {loss_sum / PIXEL_BATCH_SIZE}")

    # Test model
    test_filenames = load_data("test")
    test_ground_truth = load_ground_truth("test")
    test_images, test_ground_truth = preprocess(test_filenames, test_ground_truth, "test")
    # Extract features
    test_data = []
    for img, gt in zip(test_images, test_ground_truth):
        features = extract_features(img)
        labels = gt.flatten()
        test_data.append((features, labels))
    # Normalize features
    all_features = np.vstack([f for f,_ in test_data])
    mean = np.mean(all_features, axis=0)
    std = np.std(all_features, axis=0)
    test_data = [((f - mean) / (std + 1e-8), l) for f, l in test_data]
    # Create dataset and loader
    test_dataset = BSDS500Dataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    # Test model
    summary = []
    for i, (features, labels) in enumerate(tqdm(test_loader)):
        image_id = os.path.splitext(test_filenames[i])[0]
        features_flat = features.reshape(-1, features.shape[-1]) # (H*W, 7)
        labels_flat = labels.reshape(-1).float().unsqueeze(-1) # (H*W,)
        predictions = model.forward(features_flat)
        pred_bin = (torch.sigmoid(predictions) >= 0.5).long().cpu().numpy()
        pred_2d = pred_bin.reshape(321, 481)
        labels_2d = labels_flat.int().cpu().numpy().reshape(321, 481)
        metrics = compute_metrics(pred_2d, labels_2d, tolerance=2)
        row = {"id": image_id}
        for key, value in metrics.items():
            row[key] = value
        summary.append(row)
    summary = pd.DataFrame(summary)
    summary.to_csv("summary.csv")
        