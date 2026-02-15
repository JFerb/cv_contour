import random

import numpy as np
import torch

def InitRNG(seed):
    # Reguläre Zufallskomponenten der Bibliotheken.
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Falls CuDNN genutzt wird, müssen für die Reproduzierbarkeit zwei weitere Optionen eingestellt werden.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Metal soll ebenfalls deterministisch sein.
    torch.mps.manual_seed(seed)

    # Für alle sonstigen Fälle.
    torch.use_deterministic_algorithms(True)

def GetBestDevice():
    # Bevorzuge CUDA für effizientes Training.
    if torch.cuda.is_available():
        return "cuda"
    # Ist CUDA nicht verfügbar, wird aber ein Mac genutzt, läuft das Training über Metal.
    elif torch.backends.mps.is_available():
        return "mps"
    # Ist keine bessere Hardware verfügbar, wird mit der CPU trainiert.
    else:
        return "cpu"