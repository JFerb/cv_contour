import json
import os

import pandas as pd
import torch.nn as nn
from scipy.ndimage import binary_dilation
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from EarlyStopping import EarlyStopping
from Utils import *

class Bold:
    BEGIN = "\033[1m"
    END = "\033[0m"

class ModelManager:

    def __init__(self, model, datasets, HYPERPARAMETERS):
        # Übertrage das Modell an die beste verfügbare Hardware.
        self.model = model.to(GetBestDevice())

        # Entpacke die übergebenen datasets und bestimme die Anzahl an Klassen.
        self.train_dataset, self.val_dataset, self.test_dataset = datasets
        self.num_classes = self.train_dataset.num_classes

        # Speichere die Hyperparameter zur Archivierung
        self.HYPERPARAMETERS = HYPERPARAMETERS

        # Konstruiere DataLoader aus den erhaltenen Datensätzen.
        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=self.HYPERPARAMETERS["BATCH_SIZE"],
                                       shuffle=True,
                                       pin_memory=True)
        self.val_loader = DataLoader(self.val_dataset,
                                     batch_size=1, # Muss 1 sein, da die Anzahl an GT-Masken nicht immer gleich ist
                                     shuffle=False,
                                     pin_memory=True)
        self.test_loader = DataLoader(self.test_dataset,
                                      batch_size=1, # Muss 1 sein, da die Anzahl an GT-Masken nicht immer gleich ist
                                      shuffle=False,
                                      pin_memory=True)

        # Für das Training muss die Number der aktuellen Epoche und gespeichert werden.
        # Für das Testen und die Auswertung muss die beste Epoche bekannt sein.
        self.epoch = None
        self.best_epoch = None
        self.location = None

        # Da Klassifikationsprobleme betrachtet werden, wird CrossEntropyLoss verwendet.
        # Da das Verhältnis der Klassen (Kante <-> Nicht-Kante) sehr unausgeglichen ist, werden Kantenpixel mit
        # einem in calculate_metrics.py berechneten Faktor überbewertet, der in der Loss-Berechnung
        # ein Gleichgewicht herstellt.
        weight = torch.tensor([55.2793]).to(GetBestDevice())
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=weight)

        # Während des Trainings werden Metriken für Trainings- und Validierungsdatensatz gespeichert.
        self.metrics = {}

        # Während des Trainings soll EarlyStopping zum Einsatz kommen.
        self.early_stopping = EarlyStopping(self.HYPERPARAMETERS["EARLY_STOPPING_PATIENCE"],
                                            self.HYPERPARAMETERS["EARLY_STOPPING_DELTA"])

        # Als Optimizer kommt Adam zum Einsatz. Die Lernrate wird durch einen CosineAnnealingLR-Scheduler im Laufe des Trainings angepasst. Sie erreicht ihr Minimum von 1e-6 in der letzten erlaubten Epoche.
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.HYPERPARAMETERS["LEARNING_RATE"])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                    T_max=self.HYPERPARAMETERS["EPOCHS"],
                                                                    eta_min=1e-6)

    # Trainiere das Modell und teste es direkt im Abschluss.
    def TrainAndTest(self):
        # Falls der Speicherordner noch nicht existiert, wird er erstellt.
        if not os.path.exists("models"):
            os.makedirs("models")

        run_count = [d for d in os.listdir("models") if os.path.isdir(os.path.join("models", d))]
        self.location = os.path.join("models", f"run-{str(len(run_count) + 1)}")
        os.makedirs(self.location)

        self._archive_hyperparameters()

        self._train()
        self._test()

        # Speichere die Metriken des Modells.
        with open(os.path.join(self.location, "metrics.json"), "w") as f:
            json.dump(self.metrics, f, indent=4)

        self._plot()

    # Starte den Trainingsprozess.
    def _train(self):
        # Gehe alle Epochen durch.
        for epoch in range(1, self.HYPERPARAMETERS["EPOCHS"] + 1):
            # Das Training wird abgebrochen, falls ein EarlyStop erreicht worden ist.
            if self.early_stopping.shouldStop():
                print("EarlyStop wurde erreicht.")
                break

            # Aktualisiere den Epochenzähler.
            self.epoch = epoch

            # Lege im Metriken-Attribut einen neuen Eintrag für die neue Epoche an.
            self.metrics[str(self.epoch)] = {
                "training": {},
                "validation": {}
            }

            # Aktiviere den Trainingsmodus des Modells.
            print("━" * 100)
            print(Bold.BEGIN + f"Beginne Trainingsphase von Epoche {self.epoch}." + Bold.END)
            self.model.train()

            # Erstelle die Fortschrittsleiste mit tqdm und durchlaufe alle Batches.
            pbar = tqdm(self.train_loader, desc=f"Training von Epoche {self.epoch}/{self.HYPERPARAMETERS['EPOCHS']}")

            training_loss = 0
            preds = []
            targets = []
            for x, y in pbar:
                # Die Daten der Batch müssen auf die genutzte Hardware verschoben werden.
                x = x.to(GetBestDevice())
                y = y.to(GetBestDevice())

                # Berechne Modellvorhersage und berechne Loss.
                pred = self.model(x)
                loss = self.loss_fn(pred, y.unsqueeze(1).float())

                # Falls der Loss NaN sein sollte, wird angebrochen.
                if torch.isnan(loss):
                    print("Loss ist NaN.")
                    raise RuntimeError()

                training_loss += loss.item()

                # Berechne die Wahrscheinlichkeiten, um am Ende Metriken bestimmen zu können.
                pred = torch.sigmoid(pred)
                preds.append(pred.detach())
                targets.append(y.detach())

                # Führe den Backward-Pass durch.
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Nach jeder Epoche wird die Lernrate vom Scheduler angepasst.
            self.scheduler.step()

            training_loss /= len(self.train_loader)
            print(f"Training mit durchschnittlichem Loss {training_loss:.3f} beendet.")

            preds = torch.cat(preds, dim=0)      # torch.cat notwendig, da preds und targets noch Listen sind
            targets = torch.cat(targets, dim=0)
            precision, recall, f1, _, _, _ = self._compute_metrics(preds, targets)

            self.metrics[str(self.epoch)]["training"] = {
                "loss": training_loss,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }

            # Starte nach dem Training die Validierungsphase.
            self._validate()

        # Speichere die beste Epoche.
        self.metrics["best_epoch"] = self.best_epoch

    # Validiere das Modell nach jeder Epoche.
    def _validate(self):
        # Aktiviere den Evaluationsmodus des Modells.
        print(Bold.BEGIN + f"Beginne Validierungsphase von Epoche {self.epoch}." + Bold.END)
        self.model.eval()

        # Gehe vor wie in der Trainingsphase.
        validation_loss = 0
        preds = []
        targets = []
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Validierung von Epoche {self.epoch}/{self.HYPERPARAMETERS['EPOCHS']}")

            for x, y in pbar:
                x = x.to(GetBestDevice())

                pred = self.model(x)

                instance_loss = 0
                for mask in y:
                    mask = mask.to(GetBestDevice())
                    loss = self.loss_fn(pred, mask.unsqueeze(1).float())

                    if torch.isnan(loss):
                        print("Loss ist NaN.")
                        raise RuntimeError()

                    instance_loss += loss.item()

                    pred = torch.sigmoid(pred)
                    preds.append(pred)
                    targets.append(mask)

                # Mittlere den Validation Loss über alle möglichen Masken
                validation_loss += instance_loss / len(y)

            validation_loss /= len(self.val_loader)
            print(f"Validierung mit durchschnittlichem Loss {validation_loss:.3f} beendet.")

            preds = torch.cat(preds, dim=0)
            targets = torch.cat(targets, dim=0)
            precision, recall, f1, _, _, _ = self._compute_metrics(preds, targets)

            self.metrics[str(self.epoch)]["validation"] = {
                "loss": validation_loss,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }

            # Aktualisiere den EarlyStopping Zähler
            self.early_stopping.update(self.metrics[str(self.epoch)]["validation"]["f1"])

            # Das Modell wird gespeichert, falls es besser ist als das aktuell beste. Das alte Modell wird gelöscht.
            if self.early_stopping.hasImproved():
                [os.remove(os.path.join(self.location, f)) for f in os.listdir(self.location) if f.startswith("best_model")]
                torch.save(self.model, os.path.join(self.location, f"best_model_epoch_{self.epoch}.pt"))

                self.best_epoch = self.epoch

    # Teste das Modell auf dem Testdatensatz.
    def _test(self):
        # Lade das beste trainierte Modell.
        best_model = torch.load(os.path.join(self.location, f"best_model_epoch_{self.best_epoch}.pt"))
        self.model.load_state_dict(best_model.state_dict())
        self.model = self.model.to(GetBestDevice())

        # Gehe vor wie in den Validierungsphasen.
        print("━" * 100)
        print(Bold.BEGIN + "Beginne Testphase." + Bold.END)
        self.model.eval()

        testing_loss = 0
        preds = []
        targets = []
        results = []
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc=f"Testen des Modells")

            for x, y, image_id in pbar:
                x = x.to(GetBestDevice())
                image_id = image_id[0]

                row = {"id": image_id}

                pred = self.model(x)

                instance_loss = 0
                for i, mask in enumerate(y):
                    mask = mask.to(GetBestDevice())
                    loss = self.loss_fn(pred, mask.unsqueeze(1).float())

                    if torch.isnan(loss):
                        print("Loss ist NaN.")
                        raise RuntimeError()

                    instance_loss += loss.item()

                    pred = torch.sigmoid(pred)
                    preds.append(pred)
                    targets.append(mask)

                    precision, recall, f1, tp, fp, fn = self._compute_metrics(torch.cat([pred], dim=0),
                                                                              torch.cat([mask], dim=0))

                    row[f"annotator_{i}_tp"] = tp
                    row[f"annotator_{i}_fp"] = fp
                    row[f"annotator_{i}_fn"] = fn
                    row[f"annotator_{i}_precision"] = precision
                    row[f"annotator_{i}_recall"] = recall
                    row[f"annotator_{i}_f1"] = f1

                # Mittlere den Testing Loss über alle möglichen Masken
                testing_loss += instance_loss / len(y)

                # Beste und durchschnittliche Metriken
                all_precisions = [row[f"annotator_{idx}_precision"] for idx in range(len(y))]
                row["best_precision"] = max(all_precisions)
                row["mean_precision"] = np.mean(all_precisions)

                all_recalls = [row[f"annotator_{idx}_recall"] for idx in range(len(y))]
                row["best_recall"] = max(all_recalls)
                row["mean_recall"] = np.mean(all_recalls)

                all_f1s = [row[f"annotator_{idx}_f1"] for idx in range(len(y))]
                row["best_f1"] = max(all_f1s)
                row["mean_f1"] = np.mean(all_f1s)

                results.append(row)

            testing_loss /= len(self.test_loader)
            print(f"Testen mit durchschnittlichem Loss {testing_loss:.3f} beendet.")

            preds = torch.cat(preds, dim=0)
            targets = torch.cat(targets, dim=0)
            precision, recall, f1, _, _, _ = self._compute_metrics(preds, targets)

            self.metrics["testing"] = {
                "loss": testing_loss,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }

            # Speichere detaillierte Ergebnisse als CSV
            results = pd.DataFrame(results)
            cols_to_front = [col for col in results.columns if "best" in col or "mean" in col]
            cols_to_front.insert(0, "id")
            results = results[cols_to_front + [col for col in results.columns if col not in cols_to_front]]
            results.to_csv(os.path.join(self.location, "summary.csv"))

    def _plot(self):
        # Veranschauliche die Trainingsergebnisse.
        with open(os.path.join(self.location, "metrics.json"), "r") as f:
            metrics = json.load(f)

        best_epoch = metrics["best_epoch"]
        epochs = list(range(1, best_epoch + 1))

        train_loss = [metrics[str(e)]["training"]["loss"] for e in epochs]
        val_loss = [metrics[str(e)]["validation"]["loss"] for e in epochs]
        test_loss = metrics["testing"]["loss"]

        train_f1 = [metrics[str(e)]["training"]["f1"] for e in epochs]
        val_f1 = [metrics[str(e)]["validation"]["f1"] for e in epochs]
        test_f1 = metrics["testing"]["f1"]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

        # Oberer Plot: Loss
        ax1.plot(epochs, train_loss, label="Trainingsdatensatz", color="blue")
        ax1.plot(epochs, val_loss, label="Validierungsdatensatz", color="orange")
        ax1.axhline(y=test_loss, color="red", linestyle="--", linewidth=2, label="Testdatensatz")
        ax1.set_xlabel("Epoche", fontsize=12)
        ax1.set_ylabel("Loss", fontsize=12)
        ax1.set_title("Entwicklung des Loss", fontweight="bold", fontsize=15)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Unterer Plot: F1
        ax2.plot(epochs, train_f1, label="Trainingsdatensatz", color="blue")
        ax2.plot(epochs, val_f1, label="Validierungsdatensatz", color="orange")
        ax2.axhline(y=test_f1, color="red", linestyle="--", linewidth=2, label=f"Testdatensatz ({(test_f1*100):.2f}%)")
        ax2.set_xlabel("Epoche", fontsize=12)
        ax2.set_ylabel("F1", fontsize=12)
        ax2.set_title("Entwicklung des F-Maßes", fontweight="bold", fontsize=15)
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.location, "plot.png"), dpi=200)

    def _archive_hyperparameters(self):
        path = os.path.join(self.location, "hyperparameters.txt")

        with open(path, "w") as f:
            f.write(f"SEED                    = {self.HYPERPARAMETERS['SEED']}\n")

            f.write(f"\n")

            f.write(f"DEPTH                   = {self.HYPERPARAMETERS['DEPTH']}\n")
            f.write(f"INITIAL_CHANNELS        = {self.HYPERPARAMETERS['INITIAL_CHANNELS']}\n")
            f.write(f"CONVOLUTION_SIZE        = {self.HYPERPARAMETERS['CONVOLUTION_SIZE']}\n")

            f.write(f"\n")

            f.write(f"EPOCHS                  = {self.HYPERPARAMETERS['EPOCHS']}\n")
            f.write(f"BATCH_SIZE              = {self.HYPERPARAMETERS['BATCH_SIZE']}\n")
            f.write(f"LEARNING_RATE           = {self.HYPERPARAMETERS['LEARNING_RATE']}\n")
            f.write(f"EARLY_STOPPING_PATIENCE = {self.HYPERPARAMETERS['EARLY_STOPPING_PATIENCE']}\n")
            f.write(f"EARLY_STOPPING_DELTA    = {self.HYPERPARAMETERS['EARLY_STOPPING_DELTA']}\n")

            f.write(f"\n")

            f.write(f"TOLERANCE               = {self.HYPERPARAMETERS['TOLERANCE']}\n")

            f.write(f"\n")

            f.write(f"PARAMETER_COUNT         = {format(sum(p.numel() for p in self.model.parameters() if p.requires_grad), ',').replace(',', '.')}")

    def _compute_metrics(self, pred, target):
        # Die Kanaldimension wird nicht benötigt
        if pred.dim() == 4:
            pred = pred.squeeze(1)

        pred = (pred >= 0.5).cpu().numpy().astype(bool)
        target = target.cpu().numpy().astype(bool)

        batch_size = pred.shape[0]
        structure = np.ones((2*self.HYPERPARAMETERS["TOLERANCE"] + 1, 2*self.HYPERPARAMETERS["TOLERANCE"] + 1))

        tp, fp, fn = 0, 0, 0
        for i in range(batch_size):
            pred_dilated = binary_dilation(pred[i], structure=structure)
            target_dilated = binary_dilation(target[i], structure=structure)

            tp += np.sum(pred[i] & target_dilated)
            fp += np.sum(pred[i] & ~target_dilated)
            fn += np.sum(target[i] & ~pred_dilated)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return precision, recall, f1, tp, fp, fn