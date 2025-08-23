# Handwritten Digit Recognition (MNIST) — CNN

A clean, beginner-friendly project that trains a Convolutional Neural Network (CNN) to recognize handwritten digits (0–9) using the MNIST dataset.

## ✨ Features
- TensorFlow/Keras CNN with ~99% test accuracy (typical run)
- One-file training script: `train.py`
- Confusion matrix and training curves saved to `artifacts/`
- Reproducible: `requirements.txt` + seed setting
- Simple, well-commented code structure

## 📊 Dataset
- **MNIST** (70,000 grayscale images of size 28×28): 60,000 for training, 10,000 for testing
- Automatically downloaded by Keras on first run

## 🧠 Model
- Small CNN: `Conv2D → Conv2D → MaxPool → Dropout → Flatten → Dense → Dropout → Dense`
- Categorical cross-entropy loss, Adam optimizer

## 🚀 Quickstart
```bash
# 1) Create and activate a virtual environment (recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Train & evaluate
python train.py

# 4) (Optional) Run only evaluation on a saved model
python evaluate.py --model_path artifacts/model.h5
```

## 🗂️ Project Structure
```
mnist-cnn-handwritten-digits/
├─ train.py                 # Train and evaluate; saves metrics & plots
├─ evaluate.py              # Load a saved model and evaluate again
├─ model.py                 # Keras model factory
├─ utils.py                 # Plotting utilities (confusion matrix, curves)
├─ requirements.txt         # Dependencies
├─ README.md                # This file
├─ LICENSE                  # MIT
├─ .gitignore
└─ artifacts/               # Auto-created outputs (model + plots + reports)
```

## 📈 Outputs
After training, you'll find in `artifacts/`:
- `model.h5` — trained Keras model
- `history.json` — training history (loss/accuracy per epoch)
- `confusion_matrix.png` — normalized confusion matrix (test set)
- `training_curves_acc.png` and `training_curves_loss.png` — accuracy & loss curves
- `classification_report.txt` — precision/recall/F1 per class
- `metrics.json` — key metrics (test accuracy, etc.)

## 🔧 Reproducibility
We set seeds for NumPy and TensorFlow. For full determinism you may also need to fix OS/CUDA versions; see TensorFlow docs.

## 📜 License
This project is released under the **MIT License** — see [LICENSE](LICENSE).

---

If you use this for your portfolio or learning, a ⭐ on GitHub would be awesome!
