"""
predict.py — Real-time Devanagari character recognition using a webcam.

How it works:
    1. Detect a blue-colored object (pen cap, marker) in the webcam feed.
    2. Track its movement and draw the stroke on a virtual blackboard.
    3. When the object disappears (you lift it away), extract the drawn
       character and classify it with the trained CNN.

Usage:
    1. Make sure devanagari.h5 exists (run train.py first).
    2. Run:  python predict.py
    3. Hold a blue object in front of the camera and draw a character.
    4. Move the object out of frame to trigger prediction.
    5. Press ESC to quit.

Tip: Adjust LOWER_HSV / UPPER_HSV below to match your object's color.
"""

import cv2
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_PATH = "devanagari.h5"
IMAGE_SIZE = 32

# HSV range for the tracking object (default: blue)
LOWER_HSV = np.array([100, 50, 50])
UPPER_HSV = np.array([130, 255, 255])

# ── Label mapping ────────────────────────────────────────────────────────────
LABEL_MAP = {
    0: "CHECK",
    1: "ka",      2: "kha",     3: "ga",      4: "gha",     5: "kna",
    6: "cha",     7: "chha",    8: "ja",      9: "jha",    10: "yna",
    11: "taa",   12: "thaa",   13: "daa",    14: "dhaa",   15: "adna",
    16: "ta",    17: "tha",    18: "da",     19: "dha",    20: "na",
    21: "pa",    22: "pha",    23: "ba",     24: "bha",    25: "ma",
    26: "yaw",   27: "ra",     28: "la",     29: "waw",    30: "saw",
    31: "petchiryakha", 32: "patalosaw", 33: "ha",
    34: "chhya", 35: "tra",    36: "gya",
}


def preprocess(img):
    """Resize and reshape a grayscale image for the CNN."""
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = img.astype(np.float32) / 255.0
    return img.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1)


def predict_character(model, image):
    """Return (confidence, class_index) for a single grayscale image."""
    processed = preprocess(image)
    probabilities = model.predict(processed, verbose=0)[0]
    pred_class = int(np.argmax(probabilities))
    confidence = float(probabilities[pred_class])
    return confidence, pred_class


def main():
    model = load_model(MODEL_PATH)
    print(f"Loaded model from {MODEL_PATH}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    pts = deque(maxlen=512)
    blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
    pred_class = 0
    pred_confidence = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # Convert to HSV and create a mask for the colored object
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, LOWER_HSV, UPPER_HSV)
        mask = cv2.medianBlur(mask, 15)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        if contours:
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) > 250:
                ((x, y), radius) = cv2.minEnclosingCircle(contour)
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)

                M = cv2.moments(contour)
                if M["m00"] != 0:
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                    cv2.circle(frame, center, 5, (0, 0, 255), -1)
                    pts.appendleft(center)

                for i in range(1, len(pts)):
                    if pts[i - 1] is None or pts[i] is None:
                        continue
                    cv2.line(blackboard, pts[i - 1], pts[i], (255, 255, 255), 10)
                    cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), 5)

        else:
            # Object disappeared — run prediction on whatever was drawn
            if len(pts) > 0:
                board_gray = cv2.cvtColor(blackboard, cv2.COLOR_BGR2GRAY)
                board_gray = cv2.medianBlur(board_gray, 15)
                board_gray = cv2.GaussianBlur(board_gray, (5, 5), 0)
                board_thresh = cv2.threshold(
                    board_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )[1]

                board_cnts, _ = cv2.findContours(
                    board_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
                )

                if board_cnts:
                    cnt = max(board_cnts, key=cv2.contourArea)
                    if cv2.contourArea(cnt) > 2000:
                        x, y, w, h = cv2.boundingRect(cnt)
                        digit = board_gray[y : y + h, x : x + w]
                        pred_confidence, pred_class = predict_character(model, digit)
                        print(f"Predicted: {LABEL_MAP.get(pred_class, '?')} "
                              f"(class {pred_class}, confidence {pred_confidence:.2f})")

            # Reset the drawing surface
            pts = deque(maxlen=512)
            blackboard = np.zeros((480, 640, 3), dtype=np.uint8)

        # Overlay prediction text
        label = LABEL_MAP.get(pred_class, "?")
        cv2.putText(
            frame,
            f"Prediction: {label} ({pred_confidence:.0%})",
            (10, 460),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

        cv2.imshow("Frame", frame)
        cv2.imshow("Mask", thresh)

        if cv2.waitKey(10) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
