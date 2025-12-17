import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import math
from collections import deque

st.set_page_config(page_title="Stitched Counter", layout="wide")
st.title("🧵 شمارنده هوشمند با قابلیت ترمیم مسیر (Track Stitching)")
st.caption("حل مشکل 'دوباره شمردن' با وصل کردن هوشمند مسیرهای قطع شده.")

# --- استایل CSS برای متریک‌ها ---
st.markdown("""
<style>
    div[data-testid="stMetricValue"] { font-size: 30px; color: #00FF00; }
</style>
""", unsafe_allow_html=True)

# --- تنظیمات سایدبار ---
with st.sidebar:
    st.header("تنظیمات ورودی")
    uploaded_file = st.file_uploader("فایل ویدیو", type=["mp4", "avi", "mov"])
    
    st.divider()
    st.header("🛠 تنظیمات خط گیت")
    # تنظیمات خط
    col_a1, col_a2 = st.columns(2)
    with col_a1: line_x1 = st.slider("X شروع (%)", 0, 100, 10)
    with col_a2: line_y1 = st.slider("Y شروع (%)", 0, 100, 50)
    col_b1, col_b2 = st.columns(2)
    with col_b1: line_x2 = st.slider("X پایان (%)", 0, 100, 90)
    with col_b2: line_y2 = st.slider("Y پایان (%)", 0, 100, 50)

    st.divider()
    st.header("🧠 تنظیمات مغز (Stitcher)")
    st.info("این بخش جادوی اصلی است:")
    # فاصله مجاز برای وصل کردن دو مسیر
    stitch_dist = st.slider("فاصله بخیه (پیکسل)", 10, 200, 100, help="اگر فرد جدید در این فاصله از آخرین مکان فرد گم شده ظاهر شد، وصلش کن.")
    # زمان مجاز برای وصل کردن
    stitch_time = st.slider("حافظه بخیه (فریم)", 5, 100, 45, help="تا چند فریم منتظر بازگشت فرد گم شده بمانم؟")
    confidence = st.slider("دقت تشخیص", 0.1, 0.9, 0.25)

# لود مدل
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

try:
    model = load_model()
except:
    st.error("Error loading model")

# توابع ریاضی
def ccw(A, B, C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def intersect(A, B, C, D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def get_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# --- بدنه اصلی ---
col1, col2 = st.columns([3, 1])
with col2:
    st.markdown("### آمار نهایی")
    kpi_total = st.empty()
    st.divider()
    kpi_stitched = st.empty() # نمایش تعداد دفعاتی که هوش مصنوعی مسیر را اصلاح کرد

image_placeholder = col1.empty()

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    tfile.close()

    cap = cv2.VideoCapture(video_path)
    stop = st.button("⛔ توقف")

    # --- متغیرهای پیشرفته ---
    # تاریخچه مسیر آیدی‌های فعال: {id: [pos1, pos2, ...]}
    active_tracks = {}
    
    # لیست ارواح (افراد گم شده): {old_id: {'last_pos': (x,y), 'frames_lost': 0}}
    ghost_tracks = {}
    
    # آیدی‌هایی که شمارش شده‌اند
    counted_ids = set()
    
    total_count = 0
    stitched_count = 0 # چند بار آیدی‌ها را اصلاح کردیم؟
    
    # مپ تبدیل آیدی جدید به قدیم: {new_id: old_id}
    id_map = {}

    frame_idx = 0
    while cap.isOpened():
        if stop: break
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        
        h, w, _ = frame.shape
        
        # خط گیت (تبدیل درصد به پیکسل)
        G1 = (int(w * line_x1 / 100), int(h * line_y1 / 100))
        G2 = (int(w * line_x2 / 100), int(h * line_y2 / 100))
        
        cv2.line(frame, G1, G2, (255, 0, 0), 2) # خط آبی

        # پردازش YOLO
        # استفاده از bytetrack چون پایدار است، ما خودمان stitch را انجام می‌دهیم
        results = model.track(frame, persist=True, classes=[0], conf=confidence, tracker="bytetrack.yaml", verbose=False, device='cpu')
        
        current_frame_raw_ids = set()

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, raw_id in zip(boxes, track_ids):
                # 1. آیا این آیدی در مپ ما هست؟ (یعنی قبلا بخیه خورده؟)
                real_id = id_map.get(raw_id, raw_id)
                current_frame_raw_ids.add(raw_id)
                
                x, y, wb, hb = box
                center = (int(x), int(y + hb/2)) # پاها

                # 2. اگر آیدی کاملا جدید است، چک کن ببین شبیه ارواح هست؟
                if real_id not in active_tracks:
                    # جستجو در ارواح
                    best_match = None
                    min_dist = float('inf')
                    
                    for ghost_id, ghost_data in ghost_tracks.items():
                        dist = get_distance(center, ghost_data['last_pos'])
                        if dist < stitch_dist: # اگر نزدیک بود
                            if dist < min_dist:
                                min_dist = dist
                                best_match = ghost_id
                    
                    if best_match is not None:
                        # یافت شد! بخیه بزن
                        id_map[raw_id] = best_match # از این به بعد هر وقت raw_id اومد، بکنش best_match
                        real_id = best_match
                        del ghost_tracks[best_match] # زنده شد، از ارواح پاکش کن
                        stitched_count += 1
                        # افکت بصری بخیه
                        cv2.circle(frame, center, 20, (255, 255, 255), 3) 
                
                # 3. آپدیت مسیر
                if real_id not in active_tracks:
                    active_tracks[real_id] = []
                
                # اضافه کردن نقطه جدید
                active_tracks[real_id].append(center)
                if len(active_tracks[real_id]) > 30: # فقط ۳۰ نقطه آخر
                    active_tracks[real_id].pop(0)
                
                # 4. بررسی شمارش (تقاطع بردار)
                if len(active_tracks[real_id]) >= 2:
                    prev_pos = active_tracks[real_id][-2]
                    curr_pos = center
                    
                    # شرط شمارش: اگر قبلا شمرده نشده و خط را قطع کرده
                    if real_id not in counted_ids:
                        if intersect(G1, G2, prev_pos, curr_pos):
                            total_count += 1
                            counted_ids.add(real_id)
                            # افکت سبز عبور
                            cv2.line(frame, G1, G2, (0, 255, 0), 4)

                # رسم گرافیک
                color = (0, 255, 0) if real_id in counted_ids else (0, 165, 255)
                cv2.circle(frame, center, 5, color, -1)
                # نمایش ID برای دیباگ (آیدی واقعی)
                # cv2.putText(frame, str(real_id), center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 5. مدیریت ارواح (کسانی که در این فریم غیب شدند)
        # آیدی‌های فعلی (بعد از مپ شدن)
        current_active_real_ids = set([id_map.get(rid, rid) for rid in current_frame_raw_ids])
        
        # چک کن چه کسی قبلا بوده ولی الان نیست
        for tid in list(active_tracks.keys()):
            if tid not in current_active_real_ids:
                # این فرد غیب شد -> تبدیل به روح شود
                last_known_pos = active_tracks[tid][-1]
                ghost_tracks[tid] = {'last_pos': last_known_pos, 'frames_lost': 0}
                del active_tracks[tid] # از لیست فعال حذف کن

        # 6. کاهش عمر ارواح
        dead_ghosts = []
        for gid in ghost_tracks:
            ghost_tracks[gid]['frames_lost'] += 1
            if ghost_tracks[gid]['frames_lost'] > stitch_time:
                dead_ghosts.append(gid)
        
        for gid in dead_ghosts:
            del ghost_tracks[gid]

        # نمایش
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)
        
        kpi_total.metric("👥 تعداد کل مشتریان", total_count)
        kpi_stitched.metric("🔧 تعداد تعمیر مسیر (Stitches)", stitched_count, help="تعداد دفعاتی که هوش مصنوعی فهمید مشتری جدید، همان مشتری قبلی است.")

    cap.release()
    try: os.unlink(video_path)
    except: pass