# Smart Retail Analytics: AI Footfall Counter with Track Stitching 🛒

A computer vision MVP designed to count retail store customers accurately by solving the "re-identification" problem using CPU-optimized logic.

## 🚀 The Problem
Traditional motion-based counters fail in complex retail environments. Issues include:
- **Occlusion:** Customers disappear behind shelves and reappear, getting counted twice.
- **Jitter:** Stationary staff or customers flickering in detection.
- **Hardware Cost:** Requirement for expensive GPUs for real-time processing.

## 💡 The Solution (My Approach)
This tool uses **YOLOv8** for detection and **ByteTrack** for tracking, enhanced with a custom **"Stitching Algorithm"**. 

### Key Features:
1.  **Ghost Tracking (Memory):** When a customer disappears (occlusion), the system remembers their last position as a "Ghost". If they reappear within a specific timeframe and radius, the system "stitches" the new ID to the old one.
2.  **Virtual Gate:** Counts only when a validated track crosses a user-defined vector.
3.  **Noise Filtering:** Ignores objects that don't persist for a minimum number of frames.
4.  **CPU Optimized:** Runs smoothly on standard laptops using lightweight models (`yolov8n`).

## 🛠 Tech Stack
- **Python 3.9+**
- **Computer Vision:** `Ultralytics YOLOv8`, `OpenCV`
- **UI/Dashboard:** `Streamlit`
- **Logic:** Custom Python heuristics for vector intersection and trajectory stitching.

## 📸 How to Run
1. Clone the repo.
2. Install requirements:
   ```pip install ultralytics streamlit opencv-python-headless'''

Run the app:

streamlit run app.py
Upload a CCTV footage sample and adjust the "Stitch Distance" based on camera angle.

Built as a functional MVP to demonstrate No-Code/Low-Code architecture potential in Retail Tech.

