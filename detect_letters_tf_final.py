import cv2
import numpy as np
import tensorflow as tf
import time

# === Load trained TensorFlow model ===
model = tf.keras.models.load_model("letter_model.h5")
class_names = ['H', 'S', 'U']

# === Parameters ===
MIN_CONTOUR_AREA = 800
MAX_CONTOUR_AREA = 60000
MERGE_DISTANCE = 20
SHOW_DEBUG = True
CAPTURE_INTERVAL = 0.1  # seconds (100 ms)

# === Camera ===
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

# === Preprocessing ===
def preprocess(gray):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    return binary

def merge_close_contours(contours, max_dist=MERGE_DISTANCE):
    rects = [cv2.boundingRect(c) for c in contours]
    merged = []
    for (x, y, w, h) in rects:
        added = False
        for i, (mx, my, mw, mh) in enumerate(merged):
            if (abs(x - mx) < max_dist and abs(y - my) < max_dist):
                nx = min(x, mx)
                ny = min(y, my)
                nw = max(x + w, mx + mw) - nx
                nh = max(y + h, my + mh) - ny
                merged[i] = (nx, ny, nw, nh)
                added = True
                break
        if not added:
            merged.append((x, y, w, h))
    return merged

# === Color detection ===
def detect_colors(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    color_ranges = {
        "RED": [(0, 120, 70), (10, 255, 255), (170, 120, 70), (180, 255, 255)],
        "GREEN": [(35, 50, 50), (90, 255, 255)],
        "BLUE": [(90, 100, 100), (130, 255, 255)],
        "YELLOW": [(20, 100, 100), (35, 255, 255)]
    }

    detected_colors = []

    for color, bounds in color_ranges.items():
        if color == "RED":
            lower1, upper1, lower2, upper2 = [np.array(b) for b in bounds]
            mask1 = cv2.inRange(hsv, lower1, upper1)
            mask2 = cv2.inRange(hsv, lower2, upper2)
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            lower, upper = [np.array(b) for b in bounds]
            mask = cv2.inRange(hsv, lower, upper)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv2.contourArea(c) > 1000:
                x, y, w, h = cv2.boundingRect(c)
                detected_colors.append((color, (x, y, w, h)))

    return detected_colors

# === Main loop ===
print("Starting capture loop (press 'q' to quit)...")

while True:
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        print("Camera read failed.")
        break

    frame_draw = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    binary = preprocess(gray)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    merged_boxes = merge_close_contours(contours)

    detected_letter = None
    for (x, y, w, h) in merged_boxes:
        area = w * h
        if area < MIN_CONTOUR_AREA or area > MAX_CONTOUR_AREA:
            continue

        pad = 10
        x, y = max(0, x - pad), max(0, y - pad)
        w, h = min(frame.shape[1] - x, w + pad * 2), min(frame.shape[0] - y, h + pad * 2)
        roi = gray[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi, (128, 128))
        roi_normalized = roi_resized / 255.0
        roi_input = roi_normalized.reshape(1, 128, 128, 1)

        prediction = model.predict(roi_input, verbose=0)
        letter = class_names[np.argmax(prediction)]
        conf = np.max(prediction)

        cv2.rectangle(frame_draw, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame_draw, f"{letter} ({conf*100:.1f}%)", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        detected_letter = letter
        break  # assume one main letter in frame

    # === Detect colors ===
    detected_colors = detect_colors(frame)
    for color_name, (x, y, w, h) in detected_colors:
        cv2.rectangle(frame_draw, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.putText(frame_draw, color_name, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # === Display result ===
    cv2.imshow("Rescue Vision (Snapshot Mode)", frame_draw)
    if SHOW_DEBUG:
        cv2.imshow("Binary", binary)

    # Wait 100ms between captures
    elapsed = time.time() - start_time
    delay = max(0, CAPTURE_INTERVAL - elapsed)
    time.sleep(delay)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
