import torch
import torch.nn as nn

class DiceLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, logits, target):
        probs = torch.sigmoid(logits)

        flatten_probs = probs.view(probs.size(0), -1)
        flatten_target = target.view(target.size(0), -1)

        intersection = (flatten_probs * flatten_target).sum(dim=1)
        inclusion = flatten_probs.sum(dim=1) + flatten_target.sum(dim=1)

        dice = (2. * intersection + 1e-5) / (inclusion + 1e-5)    # Numerische Stabilität
        return 1 - dice.mean()