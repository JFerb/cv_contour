import numpy as np

class EarlyStopping:

    def __init__(self, patience, delta):
        # Speichere die Hyperparameter
        self.patience = patience
        self.delta = delta

        # Speichere die Anzahl an konsekutiven Epochen ohne Verbesserung, ob ein EarlyStop bereits eingetreten ist, ob in der letzten Epoche eine Verbesserung erzielt wurde und die beste bisher erreichte Leistung.
        self.counter = 0
        self.early_stop = False
        self.improved = False
        self.best_score = -np.inf

    # Speichere die Leistung der neusten Epoche.
    def update(self, score):
        # Bestimme, ob eine Verbesserung stattgefunden hat.
        self.improved = (score > self.best_score + self.delta)

        # Hat eine Verbesserung stattgefunden, wird der Zähler zurückgesetzt und der Wert der Metrik gespeichert.
        if self.improved:
            old_val = None if not np.isfinite(self.best_score) else self.best_score
            msg = f"Metrik verbessert ({old_val:.3f} --> {score:.3f})." if old_val is not None else \
                  f"Metrik wurde mit {score:.3f} initialisiert."

            print(msg)

            self.counter = 0
            self.best_score = score
        # Hat sich das Modell nicht verbessert, wird der Zähler erhöht und geprüft, ob ein EarlyStop erreicht wurde
        else:
            self.counter += 1

            print(f"Metrik nicht verbessert. EarlyStopping Zähler: {self.counter}/{self.patience}")

            if self.counter >= self.patience:
                self.early_stop = True

    # Gib zurück, ob ein EarlyStop erfolgen muss.
    def shouldStop(self):
        return self.early_stop

    # Gib zurück, ob mit dem letzten Aufruf von update() eine Verbesserung erreicht wurde.
    def hasImproved(self):
        return self.improved