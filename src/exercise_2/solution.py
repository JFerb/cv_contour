import os
import sys

from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from exercise_1.solution import compute_metrics, compute_confusion_matrix

########################################################################################################################
# Constants
BASE_PATH = "../../BSDS500/BSDS500/data/"


########################################################################################################################

def load_data(split):
    files = os.listdir(os.path.join(BASE_PATH, "images", split))
    count = sum(1 for file in files if file.lower().endswith((".jpg", ".jpeg")))
    print(f"Loaded {count} images from {split}.")
    return files

def load_ground_truth(split):
    files = os.listdir(os.path.join(BASE_PATH, "groundTruth", split))
    count = sum(1 for file in files if file.lower().endswith((".mat")))
    results = []
    for file in files:
        file = os.path.join(BASE_PATH, "groundTruth", split, file)
        data = loadmat(file)
        # Nehme nur den ersten Annotator für das Training
        results.append(data["groundTruth"][0][0][0][0][1])
    print(f"Loaded {count} ground truth files from {split}.")
    return results

class BSDS500Dataset(Dataset):
    def __init__(self, files, ground_truth):
        self.files = files
        self.ground_truth = ground_truth

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        return self.files[idx], self.ground_truth[idx]

if __name__ == "__main__":
    train_files = load_data("train")
    train_ground_truth = load_ground_truth("train")
    train_dataset = BSDS500Dataset(train_files, train_ground_truth)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    pass