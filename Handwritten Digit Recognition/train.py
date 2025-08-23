import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

from model import build_cnn
from utils import plot_confusion_matrix, save_classification_report, save_history, plot_training_curves

# Reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Normalize and add channel dimension
    x_train = (x_train.astype("float32") / 255.0)[..., None]
    x_test  = (x_test.astype("float32")  / 255.0)[..., None]
    return (x_train, y_train), (x_test, y_test)

def main():
    (x_train, y_train), (x_test, y_test) = load_data()

    model = build_cnn()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)
    ]

    history = model.fit(
        x_train, y_train,
        validation_split=0.1,
        epochs=15,
        batch_size=128,
        callbacks=callbacks,
        verbose=2
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")

    # Predictions
    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Save artifacts
    model.save(os.path.join(ARTIFACTS_DIR, "model.h5"))
    with open(os.path.join(ARTIFACTS_DIR, "metrics.json"), "w") as f:
        json.dump({"test_accuracy": float(test_acc), "test_loss": float(test_loss)}, f, indent=2)

    save_history(history, os.path.join(ARTIFACTS_DIR, "history.json"))
    plot_training_curves(history, os.path.join(ARTIFACTS_DIR, "training_curves"))
    plot_confusion_matrix(y_true=y_test, y_pred=y_pred, path_png=os.path.join(ARTIFACTS_DIR, "confusion_matrix.png"))
    save_classification_report(y_test, y_pred, os.path.join(ARTIFACTS_DIR, "classification_report.txt"))

if __name__ == "__main__":
    main()
