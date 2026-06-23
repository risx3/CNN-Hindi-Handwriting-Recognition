"""
train.py — Train a CNN to recognize handwritten Devanagari characters.

Dataset : Devanagari Handwritten Character Dataset (DHCD)
          46 classes (36 consonants + 10 digits), 32×32 grayscale images.
          https://archive.ics.uci.edu/ml/datasets/Devanagari+Handwritten+Character+Dataset

Architecture:
    Input (32×32×1)
      → Conv2D(32, 5×5) + ReLU
      → MaxPool(2×2)
      → Conv2D(64, 5×5) + ReLU
      → MaxPool(5×5)
      → Flatten
      → Dense(num_classes, softmax)

Usage:
    1. Extract data.zip so that data.csv is in the project root.
    2. Run:  python train.py
    3. The best model weights are saved to devanagari.h5
"""

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

# ── Hyperparameters ──────────────────────────────────────────────────────────
IMAGE_SIZE = 32
NUM_CLASSES = 37
EPOCHS = 8
BATCH_SIZE = 64
TRAIN_SPLIT = 70_000
MODEL_PATH = "devanagari.h5"

# ── 1. Load & shuffle ───────────────────────────────────────────────────────
data = pd.read_csv("data.csv")
dataset = np.array(data)
np.random.shuffle(dataset)

# First 1024 columns are pixel values (32×32), last column is the label
X = dataset[:, :1024]
Y = dataset[:, 1024]

# ── 2. Train / test split ───────────────────────────────────────────────────
X_train = X[:TRAIN_SPLIT] / 255.0
X_test = X[TRAIN_SPLIT:] / 255.0

Y_train = Y[:TRAIN_SPLIT]
Y_test = Y[TRAIN_SPLIT:]

# One-hot encode labels
Y_train = to_categorical(Y_train, NUM_CLASSES)
Y_test = to_categorical(Y_test, NUM_CLASSES)

# Reshape flat pixels → (samples, 32, 32, 1) for Conv2D
X_train = X_train.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
X_test = X_test.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)

print(f"Training samples : {X_train.shape[0]}")
print(f"Test samples     : {X_test.shape[0]}")
print(f"Input shape      : {X_train.shape[1:]}")
print(f"Number of classes : {NUM_CLASSES}")

# ── 3. Build CNN ─────────────────────────────────────────────────────────────
model = Sequential([
    Conv2D(32, (5, 5), activation="relu", input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),
    Conv2D(64, (5, 5), activation="relu"),
    MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding="same"),
    Flatten(),
    Dense(NUM_CLASSES, activation="softmax"),
])

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

# ── 4. Train ─────────────────────────────────────────────────────────────────
checkpoint = ModelCheckpoint(
    MODEL_PATH, monitor="val_accuracy", verbose=1, save_best_only=True, mode="max"
)

model.fit(
    X_train, Y_train,
    validation_data=(X_test, Y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[checkpoint],
)

# ── 5. Evaluate ──────────────────────────────────────────────────────────────
scores = model.evaluate(X_test, Y_test, verbose=0)
print(f"\nTest accuracy : {scores[1] * 100:.2f}%")
print(f"Test loss     : {scores[0]:.4f}")

model.save(MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
