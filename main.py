import tkinter as tk
from tkinter import filedialog, messagebox
from ultralytics import YOLO
import cv2
import numpy as np
import csv
import os

from sort.sort import Sort
from util import get_car, read_license_plate

# ── Load Models ──────────────────────────────────────────────────────────────
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('license_plate_detector.pt')
vehicles = [2, 3, 5, 7]

CSV_PATH = './detected_plates.csv'


def write_plates_csv(plates, output_path=CSV_PATH):
    """Append unique plates to CSV with auto-incrementing serial number.
    Creates the file with headers if it doesn't exist.
    Skips plates that are already in the file."""

    # Determine the actual output path (handle PermissionError)
    actual_path = output_path
    
    # Read existing data to find last serial number and existing plates
    last_serial = 0
    existing_plates = set()
    if os.path.exists(actual_path):
        try:
            with open(actual_path, 'r', newline='') as rf:
                reader = csv.reader(rf)
                header = next(reader, None)  # skip header
                for row in reader:
                    if len(row) >= 2:
                        try:
                            last_serial = max(last_serial, int(row[0]))
                        except ValueError:
                            pass
                        existing_plates.add(row[1])
        except (PermissionError, OSError):
            pass  # will handle below when opening for write

    # Filter out plates already in the file
    new_plates = [p for p in plates if p not in existing_plates]
    if not new_plates:
        print(f"[INFO] All {len(plates)} plate(s) already exist in CSV. Nothing to add.")
        return

    # Open file in append mode (or create with headers)
    file_exists = os.path.exists(actual_path) and os.path.getsize(actual_path) > 0
    try:
        f = open(actual_path, 'a', newline='')
    except PermissionError:
        base, ext = os.path.splitext(actual_path)
        counter = 1
        while True:
            new_path = f"{base}_{counter}{ext}"
            try:
                f = open(new_path, 'a', newline='')
                file_exists = os.path.exists(new_path) and os.path.getsize(new_path) > 0
                print(f"[WARNING] Permission denied for '{actual_path}', saving to '{new_path}' instead.")
                break
            except PermissionError:
                counter += 1
                if counter > 100:
                    raise

    with f:
        writer = csv.writer(f)
        # Write header only if file is new/empty
        if not file_exists:
            writer.writerow(['serial_no', 'license_number'])
        # Append new plates with continuing serial numbers
        for i, plate in enumerate(new_plates, start=last_serial + 1):
            writer.writerow([i, plate])

    print(f"[INFO] Appended {len(new_plates)} new plate(s) to CSV (serial {last_serial + 1} to {last_serial + len(new_plates)}).")


# ═══════════════════════════════════════════════════════════════════════════════
#  DETECTION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def detect_plates_in_frame(frame, mot_tracker):
    """Run vehicle + plate detection on a single frame.

    Only draws GREEN bounding boxes around plates that are successfully read.
    No red boxes are shown.

    Returns:
        detected: list of (car_id, plate_text, plate_score)
        frame: annotated frame
    """
    detected = []

    # detect vehicles
    detections = coco_model(frame)[0]
    detections_ = []
    for det in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = det
        if int(class_id) in vehicles:
            detections_.append([x1, y1, x2, y2, score])

    track_ids = mot_tracker.update(
        np.asarray(detections_) if len(detections_) > 0 else np.empty((0, 5))
    )

    # detect license plates
    license_plates = license_plate_detector(frame)[0]
    for lp in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = lp

        # assign to car
        xcar1, ycar1, xcar2, ycar2, car_id = get_car(lp, track_ids)

        # crop with bounds checking
        x1_c, y1_c = max(0, int(x1)), max(0, int(y1))
        x2_c, y2_c = min(frame.shape[1], int(x2)), min(frame.shape[0], int(y2))

        if x2_c > x1_c and y2_c > y1_c:
            crop = frame[y1_c:y2_c, x1_c:x2_c, :]

            # Pass the BGR crop directly — util.py handles all preprocessing
            plate_text, plate_score = read_license_plate(crop)

            if plate_text is not None:
                # GREEN bounding box — only for successful reads
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                              (0, 255, 0), 2)
                cv2.putText(frame, plate_text, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Blue box around the parent vehicle
                if car_id != -1:
                    cv2.rectangle(frame, (int(xcar1), int(ycar1)),
                                  (int(xcar2), int(ycar2)), (255, 0, 0), 2)

                detected.append((car_id, plate_text, plate_score))

    return detected, frame


def resize_to_fit(frame, max_w, max_h):
    """Resizes a frame to fit within max_w and max_h, maintaining aspect ratio."""
    h, w = frame.shape[:2]
    scale = min(max_w / w, max_h / h)
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return frame


# ═══════════════════════════════════════════════════════════════════════════════
#  MODE 1 : VIDEO UPLOAD
# ═══════════════════════════════════════════════════════════════════════════════

def run_video_upload(root):
    """Let user pick a video file, process all frames, save unique plates to CSV."""
    file_path = filedialog.askopenfilename(
        title="Select a Video File",
        filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov *.wmv"), ("All Files", "*.*")]
    )
    if not file_path:
        return

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        messagebox.showerror("Error", f"Could not open video:\n{file_path}")
        return

    mot_tracker = Sort()
    car_plates = {}       # car_id -> (plate_text, score)
    fallback_plates = set()  # plates where car_id == -1
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    max_w = int(screen_width * 0.8)
    max_h = int(screen_height * 0.8)

    window_name = 'Video Processing - License Plate Detector (Press Q to Stop)'
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    window_moved = False

    print(f"[INFO] Processing video: {file_path}  ({total_frames} frames)")
    frame_nmr = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_nmr += 1

            # Resize to fit screen
            frame = resize_to_fit(frame, max_w, max_h)

            detected, frame = detect_plates_in_frame(frame, mot_tracker)

            for car_id, plate_text, plate_score in detected:
                if car_id != -1:
                    # Keep best reading per car
                    if car_id not in car_plates or plate_score > car_plates[car_id][1]:
                        car_plates[car_id] = (plate_text, plate_score)
                        print(f"[CAR {int(car_id)}] Best plate: {plate_text} (score: {plate_score:.2f})")
                else:
                    if plate_text not in fallback_plates:
                        fallback_plates.add(plate_text)
                        print(f"[UNTRACKED] {plate_text}")

            # Show progress
            current_unique = {text for text, _ in car_plates.values()} | fallback_plates
            progress_text = f"Frame: {frame_nmr}/{total_frames} | Plates: {len(current_unique)}"
            cv2.putText(frame, progress_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            if not window_moved:
                h, w = frame.shape[:2]
                x = (screen_width - w) // 2
                y = (screen_height - h) // 2
                cv2.moveWindow(window_name, max(0, x), max(0, y))
                window_moved = True

            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    unique_plates = {text for text, _ in car_plates.values()} | fallback_plates
    if unique_plates:
        write_plates_csv(sorted(unique_plates))
        messagebox.showinfo("Done", f"Detected {len(unique_plates)} unique plate(s).\nAppended to {CSV_PATH}")
    else:
        messagebox.showinfo("Done", "No plates detected in the video.")


# ═══════════════════════════════════════════════════════════════════════════════
#  MODE 2 : PHOTO UPLOAD
# ═══════════════════════════════════════════════════════════════════════════════

def run_photo_upload(root):
    """Let user pick an image, detect plates, show annotated image, save to CSV."""
    file_path = filedialog.askopenfilename(
        title="Select an Image File",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All Files", "*.*")]
    )
    if not file_path:
        return

    frame = cv2.imread(file_path)
    if frame is None:
        messagebox.showerror("Error", f"Could not read image:\n{file_path}")
        return

    mot_tracker = Sort()

    print(f"[INFO] Processing image: {file_path}")

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    max_w = int(screen_width * 0.8)
    max_h = int(screen_height * 0.8)

    # Resize to fit screen
    frame = resize_to_fit(frame, max_w, max_h)

    detected, frame = detect_plates_in_frame(frame, mot_tracker)

    unique_plates = set()
    for _, plate_text, _ in detected:
        unique_plates.add(plate_text)

    # If no plates detected by YOLO, the image might already be a cropped plate.
    # Try running OCR directly on the entire image.
    if not unique_plates:
        print("[INFO] No plates detected by YOLO. Trying direct OCR on the image (might be a cropped plate)...")
        plate_text, plate_score = read_license_plate(frame)
        if plate_text is not None:
            unique_plates.add(plate_text)
            # Draw the result on the frame
            cv2.rectangle(frame, (5, 5), (frame.shape[1] - 5, frame.shape[0] - 5), (0, 255, 0), 2)
            cv2.putText(frame, plate_text, (10, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            print(f"[DIRECT OCR] Read plate: {plate_text} (score: {plate_score:.2f})")

    window_name = 'Detected Plates (Press any key to close)'
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    h, w = frame.shape[:2]
    x = (screen_width - w) // 2
    y = (screen_height - h) // 2
    cv2.moveWindow(window_name, max(0, x), max(0, y))

    # Show annotated image
    cv2.imshow(window_name, frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if unique_plates:
        write_plates_csv(sorted(unique_plates))
        messagebox.showinfo("Done", f"Detected {len(unique_plates)} plate(s).\nAppended to {CSV_PATH}")
    else:
        messagebox.showinfo("Done", "No plates detected in the image.")


# ═══════════════════════════════════════════════════════════════════════════════
#  GUI WINDOW
# ═══════════════════════════════════════════════════════════════════════════════

def create_gui():
    root = tk.Tk()
    root.title("License Plate Detection System")
    root.resizable(False, False)
    root.configure(bg="#1e1e2f")

    # Window dimensions
    width = 420
    height = 280

    # Center on screen
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2
    root.geometry(f"{width}x{height}+{x}+{y}")

    # Title
    title_label = tk.Label(
        root,
        text="\U0001F697  License Plate Detector",
        font=("Segoe UI", 18, "bold"),
        fg="#00e5ff",
        bg="#1e1e2f"
    )
    title_label.pack(pady=(25, 5))

    subtitle = tk.Label(
        root,
        text="Choose a detection mode below",
        font=("Segoe UI", 10),
        fg="#aaaaaa",
        bg="#1e1e2f"
    )
    subtitle.pack(pady=(0, 20))

    # Button styling
    btn_style = {
        "font": ("Segoe UI", 13, "bold"),
        "width": 28,
        "height": 2,
        "bd": 0,
        "cursor": "hand2",
        "activeforeground": "white",
    }

    btn_video = tk.Button(
        root,
        text="\U0001F3AC  Upload Video",
        bg="#2979ff", fg="white",
        activebackground="#448aff",
        command=lambda: [root.withdraw(), run_video_upload(root), root.deiconify()],
        **btn_style
    )
    btn_video.pack(pady=5)

    btn_photo = tk.Button(
        root,
        text="\U0001F5BC\uFE0F  Upload Photo",
        bg="#ff6d00", fg="white",
        activebackground="#ff9100",
        command=lambda: [root.withdraw(), run_photo_upload(root), root.deiconify()],
        **btn_style
    )
    btn_photo.pack(pady=5)

    root.mainloop()


if __name__ == '__main__':
    create_gui()
