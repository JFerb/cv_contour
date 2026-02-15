import torch
import torch.nn as nn

# ANMERKUNG: In den Conv2d-Layern wird ein Padding genutzt, um zu garantieren, dass die Bildgrößen erhalten bleiben
# und die finale Ausgabe des Modells die gleiche Größe wie seine ursprüngliche Eingabe hat.

class Initializer(nn.Module):

    def __init__(self, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(3, out_channels, 3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)

        carry = x

        x = self.pool(x)

        return x, carry

class Downsampler(nn.Module):

    def __init__(self, in_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 2 * in_channels, 3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(2 * in_channels, 2 * in_channels, 3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)

        carry = x

        x = self.pool(x)

        return x, carry

class Upsampler(nn.Module):

    def __init__(self, in_channels):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, 2, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels // 2, 3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels // 2, in_channels // 2, 3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU()

    def forward(self, carry, x):
        x = self.upsample(x)
        x = self.conv1(x)

        # Schneide den Carry zentral so zurecht, dass er die gleichen Dimensionen wie x hat
        carry_h, carry_w = carry.shape[2], carry.shape[3]
        x_h, x_w = x.shape[2], x.shape[3]

        h_offset = (carry_h - x_h) // 2
        w_offset = (carry_w - x_w) // 2

        carry = carry[:, :, h_offset:h_offset + x_h, w_offset:w_offset + x_w]

        x = torch.cat([carry, x], dim=1)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)

        return x

class Summarizer(nn.Module):

    def __init__(self, in_channels):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, 1, 1, stride=1, padding=1, bias=True)

    def forward(self, x):
        x = self.conv(x)

        return x