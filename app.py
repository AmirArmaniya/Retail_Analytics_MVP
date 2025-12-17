import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import math
import os
from collections import deque

# Page Config
st.set_page_config(page_title="Retail Analytics MVP", layout="wide")

# --- Language & Text Management ---
LANGUAGES = {
    "English": {
        "dir": "ltr",
        "title": "🧵 Smart Retail Analytics (Track Stitching)",
        "caption": "Solves 'double counting' by intelligently stitching broken tracks.",
        "sidebar_input": "Input Settings",
        "upload_label": "Upload Video File",
        "gate_settings": "🛠 Gate Line Settings",
        "start_x": "Start X (%)",
        "start_y": "Start Y (%)",
        "end_x": "End X (%)",
        "end_y": "End Y (%)",
        "ai_settings": "🧠 AI & Stitching Settings",
        "stitch_dist": "Stitch Distance (px)",
        "stitch_dist_help": "Max distance to reconnect a lost person.",
        "stitch_memory": "Stitch Memory (frames)",
        "stitch_memory_help": "How long to wait for a lost person to reappear?",
        "confidence": "Detection Confidence",
        "stop_btn": "⛔ Stop Processing",
        "stat_total": "👥 Total Visitors",
        "stat_stitched": "🔧 Path Repairs (Stitches)",
        "stat_stitched_help": "Number of times AI realized a new person is actually a returning customer.",
        "error_model": "Error loading YOLO model.",
        "info_upload": "Please upload a video to start analysis."
    },
    "Farsi": {
        "dir": "rtl",
        "title": "🧵 سامانه هوشمند تحلیل تردد (ترمیم مسیر)",
        "caption": "حل مشکل 'شمارش تکراری' با وصل کردن هوشمند مسیرهای قطع شده.",
        "sidebar_input": "تنظیمات ورودی",
        "upload_label": "آپلود فایل ویدیویی",
        "gate_settings": "🛠 تنظیمات خط گیت",
        "start_x": "شروع X (%)",
        "start_y": "شروع Y (%)",
        "end_x": "پایان X (%)",
        "end_y": "پایان Y (%)",
        "ai_settings": "🧠 تنظیمات هوش مصنوعی و ترمیم",
        "stitch_dist": "فاصله بخیه (پیکسل)",
        "stitch_dist_help": "حداکثر فاصله مجاز برای وصل کردن فرد گم شده به جدید.",
        "stitch_memory": "حافظه بخیه (فریم)",
        "stitch_memory_help": "تا چند فریم منتظر بازگشت فرد گم شده بمانم؟",
        "confidence": "دقت تشخیص (Confidence)",
        "stop_btn": "⛔ توقف پردازش",
        "stat_total": "👥 کل بازدیدکنندگان",
        "stat_stitched": "🔧 تعمیر مسیر (Stitches)",
        "stat_stitched_help": "تعداد دفعاتی که هوش مصنوعی فهمید مشتری جدید، همان قبلی است.",
        "error_model": "خطا در بارگذاری مدل YOLO.",
        "info_upload": "لطفاً برای شروع تحلیل، یک ویدیو آپلود کنید."
    }
}

# Language Selector
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3043/3043888.png", width=50)
selected_lang = st.sidebar.selectbox("Language / زبان", ["English", "Farsi"])
t = LANGUAGES[selected_lang]

# Apply RTL for Farsi
if selected_lang == "Farsi":
    st.markdown("""
    <style>
        .stApp { direction: rtl; }
        div[data-testid="stMetricValue"] { text-align: right; }
        div[data-testid="stMetricLabel"] { text-align: right; }
    </style>
    """, unsafe_allow_html=True)

# Main Title
st.title(t["title"])
st.caption(t["caption"])

# --- Sidebar Settings ---
with st.sidebar:
    st.header(t["sidebar_input"])
    uploaded_file = st.file_uploader(t["upload_label"], type=["mp4", "avi", "mov"])
    
    st.divider()
    st.header(t["gate_settings"])
    col_a1, col_a2 = st.columns(2)
    with col_a1: line_x1 = st.slider(t["start_x"], 0, 100, 10)
    with col_a2: line_y1 = st.slider(t["start_y"], 0, 100, 50)
    col_b1, col_b2 = st.columns(2)
    with col_b1: line_x2 = st.slider(t["end_x"], 0, 100, 90)
    with col_b2: line_y2 = st.slider(t["end_y"], 0, 100, 50)

    st.divider()
    st.header(t["ai_settings"])
    stitch_dist = st.slider(t["stitch_dist"], 10, 200, 100, help=t["stitch_dist_help"])
    stitch_time = st.slider(t["stitch_memory"], 5, 100, 45, help=t["stitch_memory_help"])
    confidence = st.slider(t["confidence"], 0.1, 0.9, 0.25)

# Load Model Safely
@st.cache_resource
def load_model():
    # Set env to prevent download attempts if file exists locally
    #os.environ['YOLO_DISABLE_MODEL_DOWNLOAD'] = '1'
    return YOLO('yolov8n.pt')

try:
    model = load_model()
except Exception as e:
    st.error(f"{t['error_model']} {e}")

# Helper Functions
def ccw(A, B, C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def intersect(A, B, C, D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def get_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# --- Main Logic ---
col1, col2 = st.columns([3, 1])
with col2:
    st.markdown(f"### {t['stat_total']}")
    kpi_total = st.empty()
    st.divider()
    kpi_stitched = st.empty()

image_placeholder = col1.empty()

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    tfile.close()

    cap = cv2.VideoCapture(video_path)
    stop = st.button(t["stop_btn"], type="primary")

    # Tracking Variables
    active_tracks = {}
    ghost_tracks = {}
    counted_ids = set()
    total_count = 0
    stitched_count = 0
    id_map = {}

    while cap.isOpened():
        if stop: break
        ret, frame = cap.read()
        if not ret: break
        
        h, w, _ = frame.shape
        
        # Gate Line
        G1 = (int(w * line_x1 / 100), int(h * line_y1 / 100))
        G2 = (int(w * line_x2 / 100), int(h * line_y2 / 100))
        
        cv2.line(frame, G1, G2, (255, 100, 0), 2)

        # AI Processing
        results = model.track(frame, persist=True, classes=[0], conf=confidence, tracker="bytetrack.yaml", verbose=False, device='cpu')
        
        current_frame_raw_ids = set()

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, raw_id in zip(boxes, track_ids):
                real_id = id_map.get(raw_id, raw_id)
                current_frame_raw_ids.add(raw_id)
                
                x, y, wb, hb = box
                center = (int(x), int(y + hb/2)) 

                # 1. Stitching Logic
                if real_id not in active_tracks:
                    best_match = None
                    min_dist = float('inf')
                    
                    for ghost_id, ghost_data in ghost_tracks.items():
                        dist = get_distance(center, ghost_data['last_pos'])
                        if dist < stitch_dist:
                            if dist < min_dist:
                                min_dist = dist
                                best_match = ghost_id
                    
                    if best_match is not None:
                        id_map[raw_id] = best_match
                        real_id = best_match
                        del ghost_tracks[best_match]
                        stitched_count += 1
                        cv2.circle(frame, center, 15, (255, 255, 255), 3) 
                
                # 2. Update Path
                if real_id not in active_tracks:
                    active_tracks[real_id] = []
                
                active_tracks[real_id].append(center)
                if len(active_tracks[real_id]) > 30:
                    active_tracks[real_id].pop(0)
                
                # 3. Counting Logic (Intersect)
                if len(active_tracks[real_id]) >= 2:
                    prev_pos = active_tracks[real_id][-2]
                    curr_pos = center
                    
                    if real_id not in counted_ids:
                        if intersect(G1, G2, prev_pos, curr_pos):
                            total_count += 1
                            counted_ids.add(real_id)
                            cv2.line(frame, G1, G2, (0, 255, 0), 4)

                # Draw
                color = (0, 255, 0) if real_id in counted_ids else (0, 165, 255)
                cv2.circle(frame, center, 5, color, -1)

        # Ghost Management
        current_active_real_ids = set([id_map.get(rid, rid) for rid in current_frame_raw_ids])
        
        for tid in list(active_tracks.keys()):
            if tid not in current_active_real_ids:
                last_known_pos = active_tracks[tid][-1]
                ghost_tracks[tid] = {'last_pos': last_known_pos, 'frames_lost': 0}
                del active_tracks[tid]

        # Cleanup Ghosts
        dead_ghosts = []
        for gid in ghost_tracks:
            ghost_tracks[gid]['frames_lost'] += 1
            if ghost_tracks[gid]['frames_lost'] > stitch_time:
                dead_ghosts.append(gid)
        for gid in dead_ghosts:
            del ghost_tracks[gid]

        # Display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)
        
        kpi_total.metric(t["stat_total"], total_count)
        kpi_stitched.metric(t["stat_stitched"], stitched_count, help=t["stat_stitched_help"])

    cap.release()
    try: os.unlink(video_path)
    except: pass

else:
    image_placeholder.info(t["info_upload"])
