import numpy as np
import skimage

from ModelManager import ModelManager
from BSDS500Dataset import BSDS500Dataset
from Components import *
from Utils import InitRNG

class UNet(nn.Module):

    def __init__(self, HYPERPARAMETERS):
        super().__init__()

        self.HYPERPARAMETERS = HYPERPARAMETERS
        depth = self.HYPERPARAMETERS["DEPTH"]
        initial_channels = self.HYPERPARAMETERS["INITIAL_CHANNELS"]
        convolution_size = self.HYPERPARAMETERS["CONVOLUTION_SIZE"]

        self.initializer = Initializer(initial_channels, convolution_size)

        self.downsamplers = nn.ModuleList()
        for i in range(depth - 1):
            self.downsamplers.append(Downsampler(initial_channels * (2 ** i), convolution_size))

        self.bottleneck = Bottleneck(initial_channels * (2 ** (depth - 1)), convolution_size)

        self.upsamplers = nn.ModuleList()
        for i in range(depth):
            self.upsamplers.append(Upsampler(initial_channels * (2 ** (i + 1)), convolution_size))
        self.upsamplers = nn.ModuleList(reversed(self.upsamplers))

        self.summarizer = Summarizer(initial_channels)

    def forward(self, x):
        # Speichere die zu übertragenen Feature Maps
        carries = []

        # Führe die erste Doppelfaltung durch
        x, carry = self.initializer(x)
        carries.append(carry)

        # Führe alle weiteren Downsamplings durch
        for i in range(len(self.downsamplers)):
            x, carry = self.downsamplers[i](x)
            carries.append(carry)

        # Berechne das Bottleneck
        x = self.bottleneck(x)

        # Für einfacheren Zugriff wird die Reihenfolge der Carries umgekehrt
        carries = list(reversed(carries))

        # Führe alle Upsamplings durch
        for i in range(len(self.upsamplers)):
            x = self.upsamplers[i](carries[i], x)

        # Berechne die finale Ausgabe
        x = self.summarizer(x)

        return x

    def TrainAndTest(self, datasets):
        mgr = ModelManager(self, datasets, self.HYPERPARAMETERS)
        mgr.TrainAndTest()

    def predict(self, x):
        with torch.no_grad():
            output = self.forward(x)
            output = torch.sigmoid(output).squeeze().cpu().numpy()
            output = (output >= 0.5)
            output = skimage.morphology.thin(output)
            output = output.astype(np.uint8) * 255

        return output

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    # Berechnetes POS_WEIGHT: 55.2793
    HYPERPARAMETERS = {"SEED":                      42,

                       "DEPTH":                      3,
                       "INITIAL_CHANNELS":           8,
                       "CONVOLUTION_SIZE":           7,

                       "EPOCHS":                   100,
                       "BATCH_SIZE":                16,
                       "LEARNING_RATE":           1e-4,
                       "EARLY_STOPPING_PATIENCE":   10,
                       "EARLY_STOPPING_DELTA":       0,

                       "LOSS_FUNCTION":        "dice",
                       "POS_WEIGHT":             None,
                       "BCE_FACTOR":             None,

                       "TOLERANCE":                  2}

    InitRNG(HYPERPARAMETERS["SEED"])

    unet = UNet(HYPERPARAMETERS)

    unet.TrainAndTest((BSDS500Dataset("train"),
                       BSDS500Dataset("val"),
                       BSDS500Dataset("test")))