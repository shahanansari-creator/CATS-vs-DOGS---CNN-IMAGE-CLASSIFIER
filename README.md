# CATS-vs-DOGS---CNN-IMAGE-CLASSIFIER

# Cats vs. Dogs Image Classification using CNN

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange) ![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red) 

A beginner-friendly end-to-end image classification project that uses a **Convolutional Neural Network (CNN)** built with TensorFlow and Keras to classify images of cats and dogs with high accuracy.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Model Architecture](#model-architecture)
- [Training Configuration](#training-configuration)
- [Results](#results)
- [Making Predictions](#making-predictions)
- [Possible Improvements](#possible-improvements)

---

## 🔍 Overview

This project covers a complete deep learning pipeline:

1. **Data Download & Organisation** — Automatically downloads the dataset and splits it 80/20 into train/test sets
2. **Data Augmentation & Preprocessing** — Applies rotation, flipping, zoom, and normalization to improve generalization
3. **CNN Model Building** — 3 convolutional blocks followed by fully connected layers
4. **Training & Evaluation** — Trained with SGD and categorical cross-entropy, evaluated on validation data each epoch
5. **Prediction System** — Predict on a single image or an entire folder of images

---

## 📦 Dataset

| Property | Details |
|---|---|
| Source | Microsoft Cats and Dogs Dataset |
| Total Images | ~25,000 |
| Classes | Cat, Dog |
| Train Split | 80% (~10,000 per class) |
| Test Split | 20% (~2,500 per class) |
| Format | JPEG / PNG |

The dataset is downloaded and extracted **automatically** when you run the script. No manual download required.

---

## 📁 Project Structure

```
cats-vs-dogs-cnn/
│
├── cats_vs_dogs_cnn.py        # Main script — full pipeline
├── cats_dogs_model.keras      # Saved model (generated after training)
├── training_history.png       # Accuracy & loss plot (generated after training)
│
├── dataset/
│   ├── train/
│   │   ├── Cat/
│   │   └── Dog/
│   └── test/
│       ├── Cat/
│       └── Dog/
│
└── README.md
```

---

## ⚙️ Installation

**1. Clone the repository**
```bash
git clone https://github.com/your-username/cats-vs-dogs-cnn.git
cd cats-vs-dogs-cnn
```

**2. Create a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install tensorflow matplotlib pillow
```

> **Requirements:** Python 3.8+, TensorFlow 2.x

---

## ▶️ How to Run

Simply run the main script:

```bash
python cats_vs_dogs_cnn.py
```

This will automatically:
- Download and extract the dataset (~800 MB)
- Organise it into train/test folders
- Build and train the CNN model
- Evaluate and plot results
- Save the trained model as `cats_dogs_model.keras`

---

## 🧠 Model Architecture

```
Input: (150 × 150 × 3)
        │
        ▼
┌─────────────────────┐
│  Conv2D (32 filters)│  ← Detects edges and basic patterns
│  MaxPooling2D       │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Conv2D (64 filters)│  ← Detects shapes and textures
│  MaxPooling2D       │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│ Conv2D (128 filters)│  ← Detects complex features (ears, snouts)
│ MaxPooling2D        │
└─────────────────────┘
        │
        ▼
   Flatten Layer
        │
        ▼
   Dense (128, ReLU)
        │
        ▼
   Dropout (50%)        ← Prevents overfitting
        │
        ▼
   Dense (2, Softmax)   ← Output: [Cat, Dog]
```

---

## 🔧 Training Configuration

| Parameter | Value |
|---|---|
| Image Size | 150 × 150 px |
| Batch Size | 32 |
| Epochs | 15 |
| Optimizer | SGD (lr=0.01, momentum=0.9) |
| Loss Function | Categorical Cross-Entropy |
| Metric | Accuracy |
| Dropout Rate | 0.5 |

### Data Augmentation (Training Only)

| Technique | Setting |
|---|---|
| Rescaling | 1/255 (normalize to [0,1]) |
| Rotation | ±20° |
| Width / Height Shift | 20% |
| Shear | 20% |
| Zoom | 20% |
| Horizontal Flip | Yes |

---

## 📊 Results

After training, a plot is saved as `training_history.png` showing:

- Training vs Validation **Accuracy** across epochs
- Training vs Validation **Loss** across epochs

These curves help you identify whether the model is underfitting, overfitting, or generalizing well.

> **Expected accuracy:** ~80–88% on the test set with 15 epochs using this architecture.

---

## 🔮 Making Predictions

**On a single image:**
```python
from cats_vs_dogs_cnn import predict_image
import tensorflow as tf

model = tf.keras.models.load_model("cats_dogs_model.keras")
predict_image(model, "path/to/your/image.jpg")
```

**Output:**
```
Prediction : Dog  (94.3% confidence)
```

**On a folder of images:**
```python
from cats_vs_dogs_cnn import predict_batch

predict_batch(model, "path/to/your/folder/")
```

**Output:**
```
Image                          Prediction   Confidence
----------------------------------------------------
photo1.jpg                     Cat             87.6%
photo2.jpg                     Dog             95.1%
test.png                       Cat             78.3%
```

---

## 🚀 Possible Improvements

- **Transfer Learning** — Use VGG16, ResNet50, or MobileNet as a pre-trained base to push accuracy beyond 95%
- **Learning Rate Scheduling** — Gradually reduce LR during training for better fine-tuning
- **Batch Normalization** — Add after conv layers for faster and more stable training
- **Early Stopping** — Automatically stop training when validation accuracy plateaus
- **Class Activation Maps** — Visualize which parts of the image the model focuses on for interpretability

---


## 🙌 Acknowledgements

- Dataset provided by [Microsoft](https://www.microsoft.com/en-us/download/details.aspx?id=54765)
- Built with [TensorFlow](https://www.tensorflow.org/) and [Keras](https://keras.io/)

---

