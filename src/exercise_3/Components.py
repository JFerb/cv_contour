import torch
import torch.nn as nn

# ANMERKUNG: In den Conv2d-Layern wird ein Padding genutzt, um zu garantieren, dass die Bildgrößen erhalten bleiben
# und die finale Ausgabe des Modells die gleiche Größe wie seine ursprüngliche Eingabe hat.

class Initializer(nn.Module):

    def __init__(self, out_channels, size):
        super().__init__()

        padding = (size - 1) // 2

        self.conv1 = nn.Conv2d(3, out_channels, size, stride=1, padding=padding, bias=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, size, stride=1, padding=padding, bias=True)
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

    def __init__(self, in_channels, size):
        super().__init__()

        padding = (size - 1) // 2

        self.conv1 = nn.Conv2d(in_channels, 2 * in_channels, size, stride=1, padding=padding, bias=True)
        self.conv2 = nn.Conv2d(2 * in_channels, 2 * in_channels, size, stride=1, padding=padding, bias=True)
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

class Bottleneck(nn.Module):

    def __init__(self, in_channels, size):
        super().__init__()

        padding = (size - 1) // 2

        self.conv1 = nn.Conv2d(in_channels, 2 * in_channels, size, padding=padding, bias=True)
        self.conv2 = nn.Conv2d(2 * in_channels, 2 * in_channels, size, padding=padding, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)

        return x

class Upsampler(nn.Module):

    def __init__(self, in_channels, size):
        super().__init__()

        padding = (size - 1) // 2

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, 2, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels // 2, size, stride=1, padding=padding, bias=True)
        self.conv3 = nn.Conv2d(in_channels // 2, in_channels // 2, size, stride=1, padding=padding, bias=True)
        self.relu = nn.ReLU()

    def forward(self, carry, x):
        x = self.upsample(x)
        x = self.conv1(x)

        # Mit kernel_size=2 und padding=1 wird die Ausgabe jeweils ein Pixel größer, obwohl sie gleich groß bleiben soll
        x = x[:, :, :-1, :-1]

        x = torch.cat([carry, x], dim=1)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)

        return x

class Summarizer(nn.Module):

    def __init__(self, in_channels):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, 1, 1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x = self.conv(x)

        return x