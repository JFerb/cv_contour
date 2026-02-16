from ModelManager import ModelManager
from BSDS500Dataset import BSDS500Dataset
from Components import *
from Utils import InitRNG

class UNet(nn.Module):

    def __init__(self, HYPERPARAMETERS):
        super().__init__()

        self.HYPERPARAMETERS = HYPERPARAMETERS

        self.initializer = Initializer(self.HYPERPARAMETERS["INITIAL_CHANNELS"])

        self.downsamplers = nn.ModuleList()
        for i in range(self.HYPERPARAMETERS["DEPTH"] - 1):
            self.downsamplers.append(Downsampler(self.HYPERPARAMETERS["INITIAL_CHANNELS"] * (2 ** i)))

        self.bottleneck = Bottleneck(self.HYPERPARAMETERS["INITIAL_CHANNELS"] * (2 ** (self.HYPERPARAMETERS["DEPTH"] - 1)))

        self.upsamplers = nn.ModuleList()
        for i in range(self.HYPERPARAMETERS["DEPTH"]):
            self.upsamplers.append(Upsampler(self.HYPERPARAMETERS["INITIAL_CHANNELS"] * (2 ** (i + 1))))
        self.upsamplers = nn.ModuleList(reversed(self.upsamplers))

        self.summarizer = Summarizer(self.HYPERPARAMETERS["INITIAL_CHANNELS"])

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

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    HYPERPARAMETERS = {"SEED":                      42,

                       "DEPTH":                      2,
                       "INITIAL_CHANNELS":           8,

                       "EPOCHS":                   100,
                       "BATCH_SIZE":                32,
                       "LEARNING_RATE":           1e-3,
                       "EARLY_STOPPING_PATIENCE":   10,
                       "EARLY_STOPPING_DELTA":       0,

                       "TOLERANCE":                  2}

    InitRNG(HYPERPARAMETERS["SEED"])

    unet = UNet(HYPERPARAMETERS)

    unet.TrainAndTest((BSDS500Dataset("train"),
                       BSDS500Dataset("val"),
                       BSDS500Dataset("test")))
    """

    from torchinfo import summary
    summary(unet, input_size=(32, 3, 320, 480))
    """