import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

def save_history(history, path_json):
    hist = {k: [float(x) for x in v] for k, v in history.history.items()}
    with open(path_json, "w") as f:
        json.dump(hist, f, indent=2)

def plot_training_curves(history, path_prefix):
    """Create accuracy & loss curves and save as images.
    Saves: <path_prefix>_acc.png and <path_prefix>_loss.png
    """
    hist = history.history
    epochs = range(1, len(hist["loss"]) + 1)

    # Accuracy plot
    plt.figure()
    plt.plot(epochs, hist["accuracy"], label="Train Acc")
    if "val_accuracy" in hist:
        plt.plot(epochs, hist["val_accuracy"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy")
    plt.tight_layout()
    plt.savefig(path_prefix + "_acc.png")
    plt.close()

    # Loss plot
    plt.figure()
    plt.plot(epochs, hist["loss"], label="Train Loss")
    if "val_loss" in hist:
        plt.plot(epochs, hist["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss")
    plt.tight_layout()
    plt.savefig(path_prefix + "_loss.png")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, path_png, normalize=True):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)

    fmt = ".2f" if normalize else "d"
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(path_png)
    plt.close()

def save_classification_report(y_true, y_pred, path_txt):
    report = classification_report(y_true, y_pred, digits=4)
    with open(path_txt, "w") as f:
        f.write(report)
