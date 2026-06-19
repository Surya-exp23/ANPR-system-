# 🚗 License Plate Detection System

A real-time Indian license plate detection and recognition system using YOLOv8 and EasyOCR.

## Features
- **📷 Live Webcam** – Detect plates in real-time from your webcam
- **🎬 Video Upload** – Process a video file and extract all license plates
- **🖼️ Photo Upload** – Detect plates from a single image

All detected plates are saved to `detected_plates.csv` with serial numbers.

---

## Setup Instructions

### Prerequisites
- Python 3.10 or higher
- A webcam (for live mode)

### Step 1: Clone or Download the Project
```bash
git clone <your-repo-url>
cd project-intern
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Make Sure Model Files are Present
The following model files must be in the project root:
- `yolov8n.pt` – YOLOv8 nano model for vehicle detection
- `license_plate_detector.pt` – Custom trained model for plate detection

### Step 4: Run the Application
```bash
python main.py
```
A GUI window will open with 3 options to choose from.

---

## Project Structure
```
project intern/
├── main.py                     # Main application with GUI
├── util.py                     # OCR, format checking, CSV writing
├── sort/
│   └── sort.py                 # SORT tracker for vehicle tracking
├── license_plate_detector.pt   # Custom YOLO model (plate detection)
├── yolov8n.pt                  # YOLOv8 nano model (vehicle detection)
├── requirements.txt            # Python dependencies
└── detected_plates.csv         # Output file (generated after detection)
```

## Output
The CSV file contains:
| serial_no | license_number |
|-----------|---------------|
| 1         | MH12AB1234    |
| 2         | KA09C5678     |

---

## Troubleshooting

**OpenCV GUI error (`imshow not implemented`)**
```bash
pip uninstall opencv-python-headless
pip install opencv-python
```

**Webcam not detected**
- Make sure no other app is using the webcam
- Try changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` in `main.py`
