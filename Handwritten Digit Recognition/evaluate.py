import argparse
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

from utils import plot_confusion_matrix, save_classification_report

def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_test  = (x_test.astype("float32")  / 255.0)[..., None]
    return x_test, y_test

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="artifacts/model.h5")
    args = parser.parse_args()

    x_test, y_test = load_data()
    model = tf.keras.models.load_model(args.model_path)

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Loaded model test accuracy: {test_acc:.4f}")

    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    plot_confusion_matrix(y_test, y_pred, "artifacts/confusion_matrix.png")
    save_classification_report(y_test, y_pred, "artifacts/classification_report.txt")

    with open("artifacts/metrics.json", "w") as f:
        json.dump({"test_accuracy": float(test_acc), "test_loss": float(test_loss)}, f, indent=2)

if __name__ == "__main__":
    main()
