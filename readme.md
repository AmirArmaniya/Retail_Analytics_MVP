# 🛒 Smart Retail Analytics MVP (Track Stitching Enabled)

A smart AI-powered footfall counter designed for retail environments. This system runs purely on CPU (No GPU required) and features an advanced **Track Stitching** logic to solve the common issue of double-counting customers who temporarily disappear behind shelves.

## 🚀 Features
- **Accurate Footfall Counting:** Uses a virtual gate line intersection algorithm.
- **Track Stitching Engine:** Intelligently reconnects broken paths (e.g., occlusion by shelves) to prevent re-identification errors.
- **Privacy First:** Analyzes video locally without sending data to the cloud.
- **Resource Efficient:** Optimized for laptop CPUs using YOLOv8 Nano and ByteTrack.
- **Multi-Language Support:** English and Persian (Farsi) interface.

## 🛠 Installation
1. Install Python 3.9+
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
Run the application:

streamlit run app.py


⚙️ How to Calibrate
Gate Line: Adjust the Start X/Y and End X/Y sliders to draw a blue line at the store entrance.

Stitch Distance: If customers are split into two IDs (double counted), increase this value.

Stitch Memory: If customers stay behind shelves for a long time, increase this value.

<div dir="rtl">

![output](https://github.com/user-attachments/assets/0b3eb6fe-0d3c-4bd9-aeff-8cc36b560e55)



🛒 سامانه هوشمند تحلیل تردد (نسخه MVP با ترمیم مسیر)
یک شمارنده تردد هوشمند برای فروشگاه‌ها که بدون نیاز به سرورهای گرافیکی گران‌قیمت، روی لپ‌تاپ معمولی اجرا می‌شود. این سیستم به موتور ترمیم مسیر (Track Stitching) مجهز است که مشکل "شمارش تکراری" (وقتی مشتری پشت قفسه می‌رود و برمی‌گردد) را حل می‌کند.

🚀 ویژگی‌ها
شمارش دقیق ورودی: استفاده از الگوریتم تقاطع خط (Virtual Gate) برای دقت بالا.

موتور بخیه زن (Stitcher): تشخیص هوشمند مشتریانی که غیب می‌شوند و بازمی‌گردند (Re-ID بدون نیاز به GPU).

حریم خصوصی: پردازش کاملاً لوکال (آفلاین) انجام می‌شود.

بهینه: استفاده از مدل YOLOv8 Nano برای اجرای روان روی پردازنده‌های معمولی.

دو زبانه: پشتیبانی کامل از محیط فارسی و انگلیسی.

🛠 نصب و اجرا
۱. پایتون نسخه ۳.۹ به بالا را نصب کنید. ۲. کتابخانه‌ها را نصب کنید:

pip install -r requirements.txt

۳. برنامه را اجرا کنید:

streamlit run app.py

⚙️ راهنمای تنظیم (کالیبراسیون)
خط گیت (Gate Line): با اسلایدرها خط آبی را دقیقاً پایین تصویر (ورودی فروشگاه) تنظیم کنید.

فاصله بخیه (Stitch Distance): اگر سیستم یک نفر را دو بار می‌شمارد، این عدد را زیاد کنید.

حافظه بخیه (Stitch Memory): اگر مشتریان مدت زیادی پشت قفسه می‌مانند، این عدد را زیاد کنید.

</div>
