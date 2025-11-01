import cv2
import numpy as np
import pytesseract
import os

# --- Config ---
MIN_CONTOUR_AREA = 800
MAX_CONTOUR_AREA = 60000
MERGE_DISTANCE = 20
SHOW_DEBUG = True

# Set Tesseract config for single character detection
TESSERACT_CONFIG = r'--oem 3 --psm 10 -c tessedit_char_whitelist=HSU'

# --- Initialize camera ---
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

# --- Utility functions ---
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
            if (abs(x - mx) < max_dist and abs(y - my) < max_dist) or \
               (abs((x + w) - (mx + mw)) < max_dist and abs((y + h) - (my + mh)) < max_dist):
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

# --- Color detection helper ---
def detect_colors(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    color_ranges = {
        "RED": [(0, 120, 70), (10, 255, 255), (170, 120, 70), (180, 255, 255)],
        "GREEN": [(40, 50, 50), (90, 255, 255)],
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

        # Cleanup
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv2.contourArea(c) > 1000:
                x, y, w, h = cv2.boundingRect(c)
                detected_colors.append((color, (x, y, w, h)))

    return detected_colors

# --- Main loop ---
while True:
    ret, frame = cap.read()
    if not ret:
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

        # OCR with pytesseract
        roi_resized = cv2.resize(roi, (150, 150))
        text = pytesseract.image_to_string(roi_resized, config=TESSERACT_CONFIG)
        text = text.strip().upper()

        if len(text) == 1 and text.isalpha():
            detected_letter = text
            cv2.rectangle(frame_draw, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame_draw, f"Letter: {text}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            break

    # --- Detect colors ---
    detected_colors = detect_colors(frame)
    for color_name, (x, y, w, h) in detected_colors:
        cv2.rectangle(frame_draw, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.putText(frame_draw, color_name, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Rescue Vision OCR", frame_draw)
    if SHOW_DEBUG:
        cv2.imshow("Binary", binary)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
