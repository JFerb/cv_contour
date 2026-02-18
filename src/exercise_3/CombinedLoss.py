import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss

from DiceLoss import DiceLoss
from Utils import GetBestDevice


class CombinedLoss(nn.Module):

    def __init__(self, bce_factor, pos_weight):
        super().__init__()

        self.bce = BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(GetBestDevice()))
        self.dice = DiceLoss()

        self.bce_factor = bce_factor
        self.dice_factor = 1 - bce_factor

    def forward(self, logits, target):
        return self.bce_factor * self.bce(logits, target) + self.dice_factor * self.dice(logits, target)