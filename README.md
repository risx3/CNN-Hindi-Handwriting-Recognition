# CNN Hindi (Devanagari) Handwriting Recognition

A beginner-friendly project that uses a **Convolutional Neural Network (CNN)** to recognize handwritten Devanagari characters in real time via a webcam.

https://github.com/user-attachments/assets/Application.mp4

---

## What You'll Learn

- How to **preprocess image data** (normalization, reshaping) for a CNN
- How to build a **Sequential CNN** with Keras / TensorFlow
- How **Conv2D → MaxPooling → Dense** layers work together for image classification
- How to use **OpenCV** for real-time object tracking and contour detection
- How to connect a trained model to a **live webcam feed** for inference

---

## Project Structure

```
├── train.py          # CNN model definition + training script
├── predict.py        # Real-time webcam prediction app
├── requirements.txt  # Python dependencies
├── .gitignore        # Excludes large data/model files from git
└── README.md
```

**Not tracked by git (download separately):**

| File | Description |
|---|---|
| `data.zip` → `data.csv` | Flattened pixel data (1024 pixel cols + 1 label col) |
| `DevanagariHandwrittenCharacterDataset.zip` | Raw character images organized by folder |
| `devanagari.h5` | Pre-trained model weights |

---

## Dataset

**Devanagari Handwritten Character Dataset (DHCD)**

- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Devanagari+Handwritten+Character+Dataset)
- **Classes**: 37 (36 Devanagari consonants + 1 placeholder)
- **Image size**: 32 × 32 pixels, grayscale
- **Samples**: ~92,000 images
- **Format**: CSV — each row has 1024 pixel values followed by a class label

### Devanagari Consonants (36 characters)

| # | Romanized | # | Romanized | # | Romanized |
|---|-----------|---|-----------|---|-----------|
| 1 | ka | 13 | daa | 25 | ma |
| 2 | kha | 14 | dhaa | 26 | yaw |
| 3 | ga | 15 | adna | 27 | ra |
| 4 | gha | 16 | ta | 28 | la |
| 5 | kna | 17 | tha | 29 | waw |
| 6 | cha | 18 | da | 30 | saw |
| 7 | chha | 19 | dha | 31 | petchiryakha |
| 8 | ja | 20 | na | 32 | patalosaw |
| 9 | jha | 21 | pa | 33 | ha |
| 10 | yna | 22 | pha | 34 | chhya |
| 11 | taa | 23 | ba | 35 | tra |
| 12 | thaa | 24 | bha | 36 | gya |

---

## CNN Architecture

```
Layer                Output Shape       Parameters
──────────────────────────────────────────────────
Conv2D (32, 5×5)     (28, 28, 32)       832
MaxPooling2D (2×2)   (14, 14, 32)       0
Conv2D (64, 5×5)     (10, 10, 64)       51,264
MaxPooling2D (5×5)   (2, 2, 64)         0
Flatten              (256)              0
Dense (37, softmax)  (37)               9,509
──────────────────────────────────────────────────
Total params: 61,605
```

### Why this architecture works

1. **Conv2D layers** detect local patterns (edges, curves, strokes) — essential for character shapes.
2. **MaxPooling** reduces spatial dimensions, making the model translation-invariant and faster.
3. **Flatten + Dense(softmax)** converts spatial features into class probabilities.
4. **Adam optimizer** adapts learning rates per-parameter for faster convergence.
5. **Categorical cross-entropy** is the standard loss for multi-class classification.

---

## Quick Start

### 1. Setup

```bash
# Clone the repository
git clone https://github.com/YourUsername/CNN-Hindi-Handwriting-Recognition.git
cd CNN-Hindi-Handwriting-Recognition

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare the data

Download the dataset and extract `data.csv` into the project root:

```bash
unzip data.zip
```

### 3. Train the model

```bash
python train.py
```

This will:
- Load `data.csv` and split into training (70,000) and test sets
- Train for 8 epochs with batch size 64
- Save the best model to `devanagari.h5`
- Print test accuracy on completion

### 4. Run the real-time predictor

```bash
python predict.py
```

- Hold a **blue object** (pen cap, marker) in front of your webcam
- Draw a Devanagari character in the air
- Move the object away — the model predicts the character
- Press **ESC** to quit

> **Tip:** If detection is poor, adjust `LOWER_HSV` and `UPPER_HSV` in `predict.py` to match your object's color in HSV space.

---

## Key Concepts for Learners

### Image as data
Each 32×32 image is 1024 pixels. Pixel values range 0–255 and are normalized to 0–1 for training (helps gradient descent converge).

### Convolution
A small filter (kernel) slides across the image, computing dot products to produce a feature map. Different filters learn different patterns (horizontal edges, curves, etc.).

### Pooling
MaxPooling takes the maximum value in each window, reducing the feature map size while keeping the strongest activations.

### Softmax
Converts raw model outputs into probabilities that sum to 1. The class with the highest probability is the prediction.

### One-Hot Encoding
Label `5` becomes `[0, 0, 0, 0, 0, 1, 0, ...]` — a vector where only the true class position is 1. Required for categorical cross-entropy loss.

---

## Things to Try (Exercises)

1. **Add more layers** — Insert a `Dense(128, activation='relu')` before the final layer. Does accuracy improve?
2. **Add Dropout** — Add `Dropout(0.25)` after pooling layers to reduce overfitting.
3. **Data augmentation** — Use `ImageDataGenerator` to rotate/shift training images.
4. **Increase epochs** — Train for 20+ epochs. Watch for overfitting (validation accuracy drops while training accuracy rises).
5. **Try different optimizers** — Replace `adam` with `sgd` or `rmsprop` and compare.
6. **Visualize filters** — Plot what the Conv2D filters learn using `model.layers[0].get_weights()`.

---

## Requirements

- Python 3.9+
- TensorFlow 2.15+
- NumPy
- Pandas
- OpenCV (opencv-python)

---

## License

This project is intended for educational purposes.
