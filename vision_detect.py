import argparse
import os
import sys
import glob
import json
import time
from typing import List, Tuple, Dict

import cv2
import numpy as np


# -----------------------------
# Letter template generation
# -----------------------------
def make_letter_template(letter: str, size: int = 128, font_scale: float = 3.0, thickness: int = 10) -> np.ndarray:
    img = np.full((size, size), 255, np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(letter, font, font_scale, thickness)
    x = (size - tw) // 2
    y = (size + th) // 2
    cv2.putText(img, letter, (x, y), font, font_scale, (0,), thickness, cv2.LINE_AA)
    return img


def rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return rotated


def generate_letter_templates(letters: List[str] = ["H", "S", "U"],
                              base_size: int = 128,
                              scales: List[float] = None,
                              angles: List[float] = None,
                              include_inverted: bool = True,
                              edge_mode: bool = False) -> Dict[str, List[np.ndarray]]:
    if scales is None:
        scales = [0.3, 0.45, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0]
    if angles is None:
        angles = [-12, -8, -4, 0, 4, 8, 12]
    templates: Dict[str, List[np.ndarray]] = {}
    for L in letters:
        base = make_letter_template(L, size=base_size, font_scale=3.0, thickness=10)
        variants: List[np.ndarray] = []
        for ang in angles:
            rot = rotate_image(base, ang)
            for s in scales:
                sz = max(24, int(base_size * s))
                tpl = cv2.resize(rot, (sz, sz), interpolation=cv2.INTER_AREA)
                # Ensure binary-ish template (white background, dark letter)
                tpl_bin = cv2.threshold(tpl, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                if edge_mode:
                    tpl_edges = cv2.Canny(tpl_bin, 60, 140)
                    variants.append(tpl_edges)
                    if include_inverted:
                        variants.append(cv2.Canny(255 - tpl_bin, 60, 140))
                else:
                    variants.append(tpl_bin)
                    if include_inverted:
                        variants.append(255 - tpl_bin)
        templates[L] = variants
    return templates


# -----------------------------
# Template matching + NMS
# -----------------------------
def non_max_suppression(boxes: List[Tuple[int, int, int, int]], scores: List[float], overlap_thresh: float = 0.3):
    if len(boxes) == 0:
        return []
    boxes_np = np.array(boxes, dtype=float)
    scores_np = np.array(scores, dtype=float)

    x1 = boxes_np[:, 0]
    y1 = boxes_np[:, 1]
    x2 = boxes_np[:, 0] + boxes_np[:, 2]
    y2 = boxes_np[:, 1] + boxes_np[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores_np.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= overlap_thresh)[0]
        order = order[inds + 1]
    return [int(k) for k in keep]


def match_letters(gray: np.ndarray,
                  templates: Dict[str, List[np.ndarray]],
                  threshold: float = 0.55,
                  max_detections_per_letter: int = 3,
                  edge_mode: bool = False):
    detections = []  # dicts: {label, score, box}
    src = gray
    if edge_mode:
        src = cv2.Canny(gray, 60, 140)
    for label, tpls in templates.items():
        boxes = []
        scores = []
        for tpl in tpls:
            th, tw = tpl.shape[:2]
            if src.shape[0] < th or src.shape[1] < tw:
                continue
            res = cv2.matchTemplate(src, tpl, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= threshold)
            for (y, x) in zip(*loc):
                score = float(res[y, x])
                boxes.append((x, y, tw, th))
                scores.append(score)
        if boxes:
            keep_idx = non_max_suppression(boxes, scores, overlap_thresh=0.35)
            # sort kept by score desc
            keep_idx = sorted(keep_idx, key=lambda i: scores[i], reverse=True)
            for i in keep_idx[:max_detections_per_letter]:
                detections.append({
                    "label": label,
                    "score": float(scores[i]),
                    "box": [int(v) for v in boxes[i]]
                })
    # Sort overall by score desc
    detections.sort(key=lambda d: d["score"], reverse=True)
    return detections


# -----------------------------
# Color square detection (HSV)
# -----------------------------
def detect_color_squares(frame_bgr: np.ndarray) -> List[Dict]:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    # Tuned to RCJ markers (adjust in field)
    ranges = {
        "RED": [(np.array([0, 100, 80]), np.array([10, 255, 255])),
                 (np.array([170, 100, 80]), np.array([180, 255, 255]))],
        # Slightly broadened yellow (lower S)
        "YELLOW": [(np.array([18, 60, 120]), np.array([38, 255, 255]))],
        "GREEN": [(np.array([40, 60, 60]), np.array([90, 255, 255]))],
    }

    results: List[Dict] = []
    for color, bounds_list in ranges.items():
        mask_total = None
        for lb, ub in bounds_list:
            mask = cv2.inRange(hsv, lb, ub)
            mask_total = mask if mask_total is None else cv2.bitwise_or(mask_total, mask)
        kernel = np.ones((5, 5), np.uint8)
        mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_OPEN, kernel)
        mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_CLOSE, kernel)

        cnts, _ = cv2.findContours(mask_total, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 800:
                continue
            x, y, w, h = cv2.boundingRect(c)
            ar = w / float(h)
            if ar < 0.7 or ar > 1.3:
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            if len(approx) < 4 or len(approx) > 8:
                continue
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            if hull_area <= 0:
                continue
            solidity = area / hull_area
            if solidity < 0.8:
                continue
            results.append({"color": color, "box": [int(x), int(y), int(w), int(h)], "area": float(area)})
    return results


# -----------------------------
# Visualization helpers
# -----------------------------
def draw_detections(frame_bgr: np.ndarray, letters: List[Dict], squares: List[Dict]) -> np.ndarray:
    out = frame_bgr.copy()
    for d in letters:
        x, y, w, h = d["box"]
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 200, 0), 2)
        cv2.putText(out, f"{d['label']} {d['score']:.2f}", (x, max(0, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
    for s in squares:
        x, y, w, h = s["box"]
        color_name = s["color"]
        color_map = {"RED": (0, 0, 255), "YELLOW": (0, 255, 255), "GREEN": (0, 200, 0)}
        cv2.rectangle(out, (x, y), (x + w, y + h), color_map.get(color_name, (0, 255, 255)), 2)
        cv2.putText(out, color_name, (x, max(0, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_map.get(color_name, (0, 255, 255)), 2)
    return out


# -----------------------------
# Demo sample generator (images)
# -----------------------------
def generate_demo_samples(out_dir: str, count: int = 12):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(42)
    letters = ["H", "S", "U"]
    for i in range(count):
        canvas = np.full((480, 640, 3), 255, np.uint8)
        # Randomly decide to place a letter sign
        if rng.random() < 0.8:
            L = rng.choice(letters)
            tpl = make_letter_template(L, size=180, font_scale=4.0, thickness=16)
            # Convert template to BGR with alpha-like mask
            mask = cv2.threshold(tpl, 200, 255, cv2.THRESH_BINARY_INV)[1]
            y = int(rng.integers(60, 240))
            x = int(rng.integers(60, 380))
            roi = canvas[y:y+tpl.shape[0], x:x+tpl.shape[1]]
            if roi.shape[0] == tpl.shape[0] and roi.shape[1] == tpl.shape[1]:
                inv = cv2.bitwise_not(mask)
                bg = cv2.bitwise_and(roi, roi, mask=inv)
                fg = cv2.cvtColor(tpl, cv2.COLOR_GRAY2BGR)
                fg = cv2.bitwise_and(fg, fg, mask=mask)
                roi[...] = cv2.add(bg, fg)
        # Randomly add a colored square
        for color_name, bgr in [("RED", (0, 0, 255)), ("YELLOW", (0, 255, 255)), ("GREEN", (0, 200, 0))]:
            if rng.random() < 0.5:
                sz = int(rng.integers(60, 120))
                y = int(rng.integers(40, 360))
                x = int(rng.integers(40, 520))
                cv2.rectangle(canvas, (x, y), (x+sz, y+sz), bgr, thickness=-1)
        path = os.path.join(out_dir, f"sample_{i:02d}.png")
        cv2.imwrite(path, canvas)


# -----------------------------
# Main CLI
# -----------------------------
def _open_capture(args):
    # Resolve backend constant
    backend = None
    if getattr(args, "backend", None):
        b = args.backend.lower()
        if b == "dshow":
            backend = cv2.CAP_DSHOW
        elif b == "msmf":
            backend = cv2.CAP_MSMF
        elif b == "any":
            backend = cv2.CAP_ANY

    # URL takes precedence
    if getattr(args, "camera_url", None):
        cap = cv2.VideoCapture(args.camera_url)
        return cap

    # Try device by name when DSHOW is selected and device_name provided
    device_name = getattr(args, "device_name", None)
    if device_name and backend == cv2.CAP_DSHOW:
        cap = cv2.VideoCapture(f"video={device_name}", cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap

    # Try by index with requested backend or CAP_ANY
    index = int(getattr(args, "camera_index", 0))
    if backend is not None:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            return cap

    # Fallbacks: MSMF -> DSHOW -> ANY
    for be in (cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY):
        try:
            cap = cv2.VideoCapture(index, be)
            if cap.isOpened():
                print(json.dumps({"info": "camera_open_fallback", "backend": "MSMF" if be==cv2.CAP_MSMF else ("DSHOW" if be==cv2.CAP_DSHOW else "ANY"), "index": index}))
                return cap
        except Exception:
            pass

    # Last attempt: if a device name was provided, try DSHOW string open
    if device_name:
        cap = cv2.VideoCapture(f"video={device_name}", cv2.CAP_DSHOW)
        if cap.isOpened():
            print(json.dumps({"info": "camera_open_by_name", "backend": "DSHOW", "device_name": device_name}))
            return cap

    return cv2.VideoCapture(-1)  # unopened sentinel


def run_camera(args):
    scales_str = getattr(args, 'scales', None)
    angles_str = getattr(args, 'angles', None)
    scales = [float(x) for x in scales_str.split(',')] if scales_str else None
    angles = [float(x) for x in angles_str.split(',')] if angles_str else None
    include_inverted = not getattr(args, 'no_invert', False)
    use_edges = bool(getattr(args, 'use_edges', False))
    templates = generate_letter_templates(["H", "S", "U"],
                                          scales=scales,
                                          angles=angles,
                                          include_inverted=include_inverted,
                                          edge_mode=use_edges)
    cap = _open_capture(args)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(args.width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(args.height))
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")
    print("[INFO] Camera mode: press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Camera read failed.")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if bool(getattr(args, 'clahe', False)):
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        letter_dets = match_letters(gray, templates, threshold=args.letter_threshold, edge_mode=use_edges)
        color_dets = detect_color_squares(frame)

        annotated = draw_detections(frame, letter_dets, color_dets)
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            out_path = os.path.join(args.output_dir, f"frame_{int(time.time()*1000)}.jpg")
            cv2.imwrite(out_path, annotated)

        payload = {
            "letters": letter_dets,
            "colors": color_dets,
            "timestamp": time.time(),
        }
        print(json.dumps(payload))

        if not args.headless:
            cv2.imshow("Rescue Vision - Template Matching", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()


def run_images(args):
    scales_str = getattr(args, 'scales', None)
    angles_str = getattr(args, 'angles', None)
    scales = [float(x) for x in scales_str.split(',')] if scales_str else None
    angles = [float(x) for x in angles_str.split(',')] if angles_str else None
    include_inverted = not getattr(args, 'no_invert', False)
    use_edges = bool(getattr(args, 'use_edges', False))
    templates = generate_letter_templates(["H", "S", "U"],
                                          scales=scales,
                                          angles=angles,
                                          include_inverted=include_inverted,
                                          edge_mode=use_edges)
    src_dir = args.input
    if args.demo_samples:
        print(f"[INFO] Generating demo samples into: {src_dir}")
        generate_demo_samples(src_dir, count=12)

    patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
    paths = []
    for p in patterns:
        paths.extend(glob.glob(os.path.join(src_dir, p)))
    paths.sort()
    if not paths:
        print(f"[WARN] No images found in {src_dir}")
        return

    os.makedirs(args.output_dir, exist_ok=True) if args.output_dir else None
    for p in paths:
        frame = cv2.imread(p)
        if frame is None:
            print(f"[WARN] Unable to read {p}")
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if bool(getattr(args, 'clahe', False)):
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        letter_dets = match_letters(gray, templates, threshold=args.letter_threshold, edge_mode=use_edges)
        color_dets = detect_color_squares(frame)
        annotated = draw_detections(frame, letter_dets, color_dets)

        payload = {
            "image": os.path.basename(p),
            "letters": letter_dets,
            "colors": color_dets,
        }
        print(json.dumps(payload))
        if args.output_dir:
            out_path = os.path.join(args.output_dir, os.path.basename(p))
            cv2.imwrite(out_path, annotated)
        if not args.headless:
            cv2.imshow("Rescue Vision - Template Matching (Images)", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Rescue Maze Vision (Template Matching)")
    ap.add_argument("--mode", choices=["camera", "images"], default="camera", help="Run using camera or a folder of images")
    ap.add_argument("--input", type=str, default="samples", help="Input folder for images mode")
    ap.add_argument("--output-dir", type=str, default=None, help="Optional output folder for annotated frames")
    ap.add_argument("--letter-threshold", type=float, default=0.55, help="Threshold for letter template matching (0-1)")
    ap.add_argument("--headless", action="store_true", help="Disable display windows")
    ap.add_argument("--demo-samples", action="store_true", help="Generate demo images into --input before processing (images mode only)")
    # Camera options
    ap.add_argument("--camera-index", type=int, default=0, help="Camera index for OpenCV (e.g., 0,1,2)")
    ap.add_argument("--camera-url", type=str, default=None, help="URL for IP camera stream (e.g., http://ip:port/video)")
    ap.add_argument("--backend", type=str, default=None, choices=[None, "dshow", "msmf", "any"], help="Windows capture backend")
    ap.add_argument("--width", type=int, default=640, help="Capture width")
    ap.add_argument("--height", type=int, default=480, help="Capture height")
    ap.add_argument("--list-cams", action="store_true", help="Probe camera indices 0-10 and print which open")
    ap.add_argument("--device-name", type=str, default=None, help="DirectShow device friendly name, e.g. 'Camo Studio' or 'EpocCam Camera'")
    # Matching options
    ap.add_argument("--scales", type=str, default=None, help="Comma-separated scales for templates, e.g. 0.3,0.5,0.8,1.0,1.5,2.0")
    ap.add_argument("--angles", type=str, default=None, help="Comma-separated angles in degrees, e.g. -12,-8,-4,0,4,8,12")
    ap.add_argument("--no-invert", action="store_true", help="Do not include inverted templates (for white-on-dark letters)")
    ap.add_argument("--use-edges", action="store_true", help="Match on Canny edges (robust to lighting and colors)")
    ap.add_argument("--clahe", action="store_true", help="Apply CLAHE contrast normalization before matching")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.list_cams:
        print("[INFO] Probing camera indices 0-10 ...")
        available = []
        for i in range(0, 11):
            cap = cv2.VideoCapture(i, cv2.CAP_ANY)
            ok = cap.isOpened()
            if ok:
                ret, _ = cap.read()
                if ret:
                    available.append(i)
            cap.release()
        print(json.dumps({"available_camera_indices": available}))
        if args.mode == "camera" and args.camera_index not in available and not args.camera_url:
            print("[WARN] Requested camera index not available; pass --camera-url or choose an available index.")
    
    if args.mode == "camera":
        run_camera(args)
    else:
        run_images(args)


if __name__ == "__main__":
    main()
