import os

# Set YOLO_CONFIG_DIR before any ultralytics module imports
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"
os.makedirs("/tmp/Ultralytics", exist_ok=True)

import streamlit as st
import io
import cv2
import numpy as np
from PIL import Image, ImageOps
import time

# Imports from local modules
from config import (
    APP_TITLE, APP_SUBTITLE, AUTHOR, VERSION,
    ALLOWED_IMAGE_EXTENSIONS, ALLOWED_VIDEO_EXTENSIONS,
    UPLOAD_DIR, RESULT_DIR, MODEL_PATH
)
from utils.detector import get_detector
import database as db

# Initialize database
db.init_db()

# Page configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🪖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern Custom CSS Styling
st.markdown("""
<style>
    /* Dark AI Theme & Font Customization */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #0b0f19 0%, #111827 50%, #0d1322 100%);
        color: #f3f4f6;
    }

    /* Card Styles */
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 20px;
    }

    .metric-card {
        background: linear-gradient(145deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 25px -5px rgba(59, 130, 246, 0.4);
    }

    .metric-title {
        color: #94a3b8;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .metric-value {
        color: #60a5fa;
        font-size: 2.2rem;
        font-weight: 700;
        margin-top: 5px;
    }

    /* Custom Badges */
    .badge-safe {
        background-color: #065f46;
        color: #34d399;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1rem;
        display: inline-block;
    }

    .badge-danger {
        background-color: #991b1b;
        color: #fca5a5;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1rem;
        display: inline-block;
    }

    /* Header Styling */
    .app-header {
        text-align: center;
        padding: 20px 0 30px 0;
    }

    .app-header h1 {
        background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
    }

    .app-header p {
        color: #94a3b8;
        font-size: 1.1rem;
    }

    /* Streamlit Upload Container */
    .stUploader {
        border: 2px dashed #3b82f6 !important;
        border-radius: 12px !important;
        padding: 10px !important;
    }
</style>
""", unsafe_allow_html=True)

# Application Header Component
def render_header():
    st.markdown(
        f"""<div class="app-header">
<h1>🪖 {APP_TITLE}</h1>
<p>{APP_SUBTITLE}</p>
</div>""",
        unsafe_allow_html=True,
    )

# Detector initialization with Streamlit caching
@st.cache_resource
def load_cached_detector():
    return get_detector()

detector = load_cached_detector()

# Model status checks
if not os.path.exists(MODEL_PATH):
    st.error(f"MODEL NOT FOUND: {MODEL_PATH} does not exist. Please place your trained model file in the models/ directory.")
    st.stop()

if detector.load_error:
    st.error(f"MODEL LOAD ERROR: {detector.load_error}")
    st.stop()

if not detector.model_loaded:
    st.error("MODEL NOT LOADED: The YOLO model failed to initialize. Check logs for details.")
    st.stop()

# Check if model appears to be a proper custom detection model
model_names_count = len(detector.class_names)
if model_names_count > 10:
    st.warning(
        "MODEL ACCURACY ISSUE: The loaded model appears to be a general-purpose COCO-pretrained model "
        f"({model_names_count} classes) rather than a custom helmet detection model. "
        "Detections will show COCO classes (person, car, etc.) instead of helmet classes. "
        "Train a dedicated helmet detection model and replace models/best.pt for accurate results."
    )

# Sidebar Navigation Setup
st.sidebar.image("https://img.icons8.com/isometric/100/construction-worker-helmet.png", width=70)
st.sidebar.title("Navigation Menu")

selected_page = st.sidebar.radio(
    "Select Module",
    [
        "Home",
        "Image Detection",
        "Video Detection",
        "Webcam Detection",
        "Dashboard",
        "Detection History",
        "About"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Version:** {VERSION}")
st.sidebar.markdown(f"**Author:** {AUTHOR}")
st.sidebar.markdown("**Engine:** YOLOv8 (Ultralytics)")

# ==========================================
# PAGE 1: HOME
# ==========================================
if selected_page == "Home":
    render_header()

    col1, col2 = st.columns([12, 8])

    with col1:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color:#60a5fa; margin-top:0;">🚀 AI Powered Safety Compliance System</h3>
            <p>Welcome to the <b>AI Helmet Detection System</b>. This application leverages cutting-edge <b>YOLOv8 Deep Learning Architecture</b> to monitor, detect, and enforce safety helmet compliance across high-risk environments like construction sites, industrial plants, and traffic surveillance networks.</p>
            <ul>
                <li>⚡ <b>Real-Time Frame Inferencing</b> with ultra-high accuracy and low latency.</li>
                <li>🖼️ <b>Multi-Format Support</b>: Analyze images, recorded video streams, and live webcams.</li>
                <li>📊 <b>Automated Analytics</b>: Track compliance rates, safety violations, and confidence scores.</li>
                <li>💾 <b>SQLite Database Integration</b>: Permanent logging of detection logs for audit trails.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🛠️ Quick System Capabilities")
        cap1, cap2, cap3 = st.columns(3)
        with cap1:
            st.info("📷 **Image Scanner**\n\nUpload static images to detect helmet compliance instantly.")
        with cap2:
            st.success("🎥 **Video Analysis**\n\nProcess CCTV or recorded surveillance video footage frame-by-frame.")
        with cap3:
            st.warning("📹 **Live Webcam**\n\nStream real-time webcam video feed for live monitoring.")

    with col2:
        st.markdown("### 📈 Live Safety Overview")
        stats = db.get_analytics_summary()

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Compliance Rate</div>
            <div class="metric-value" style="color:#34d399;">{stats['compliance_rate']}%</div>
        </div>
        <br>
        <div class="metric-card">
            <div class="metric-title">Total Processed Detections</div>
            <div class="metric-value">{stats['total_detections']}</div>
        </div>
        <br>
        <div class="metric-card">
            <div class="metric-title">Helmets Detected</div>
            <div class="metric-value" style="color:#38bdf8;">{stats['helmet_count']}</div>
        </div>
        <br>
        <div class="metric-card">
            <div class="metric-title">Violations (No Helmet)</div>
            <div class="metric-value" style="color:#f87171;">{stats['without_helmet_count']}</div>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# PAGE 2: IMAGE DETECTION
# ==========================================
elif selected_page == "Image Detection":
    render_header()
    st.subheader("🖼️ Image Helmet Detection")
    st.caption("Upload a JPG, JPEG, PNG, or WebP image to detect helmet compliance.")

    # Confidence threshold slider
    conf_threshold = st.slider("Detection Confidence Threshold", min_value=0.10, max_value=0.90, value=0.50, step=0.05)

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "webp"],
        key="image_uploader"
    )

    if uploaded_file is not None:
        # Display basic file info
        st.caption(f"**Selected file:** {uploaded_file.name}")
        st.write(f"**File type:** {uploaded_file.type}")
        st.write(f"**File size:** {uploaded_file.size} bytes")

        # Load image safely
        try:
            image_bytes = uploaded_file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            st.error(f"Unable to open image: {e}")
            image = None

        if image is not None:
            # Show original image
            st.markdown("---")
            st.markdown("### ORIGINAL IMAGE")
            st.image(image, caption=uploaded_file.name, width=700)

            # Run detection
            with st.spinner("🔍 Running YOLOv8 Detection Model..."):
                try:
                    start_time = time.time()
                    annotated_img, detections, counts, avg_conf = detector.detect_image(
                        image, conf_threshold=conf_threshold
                    )
                    proc_time = round(time.time() - start_time, 3)
                except Exception as e:
                    st.error(f"Detection failed: {e}")
                    annotated_img = None

            if annotated_img is not None:
                st.markdown("---")
                st.markdown("### YOLO DETECTION RESULT")
                st.image(annotated_img, caption="Annotated Result", width=700)

                # Model information
                with st.expander("Model Information"):
                    st.json({
                        "Model Path": MODEL_PATH,
                        "Model Classes": detector.class_names,
                        "Number of Detections": len(detections),
                        "Confidence Threshold": conf_threshold,
                        "Processing Time (s)": proc_time
                    })

                # Summary statistics
                helmet_count = counts.get("With Helmet", 0)
                no_helmet_count = counts.get("Without Helmet", 0)
                if no_helmet_count > 0:
                    result_str = "Violation"
                elif helmet_count > 0:
                    result_str = "Safe"
                else:
                    result_str = "No Detection"

                st.markdown("---")
                if result_str == "No Detection":
                    st.markdown("### Status: **No Helmet Detection Found**")
                else:
                    st.markdown(f"### Status: **{result_str}**")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("With Helmet", helmet_count)
                m2.metric("Without Helmet", no_helmet_count)
                m3.metric("Average Confidence", f"{round(avg_conf * 100, 1)}%")
                m4.metric("Processing Time", f"{proc_time}s")

                # Save to history button
                if st.button("💾 Save Detection to History Database", type="primary"):
                    rec_id = db.save_detection(
                        filename=uploaded_file.name,
                        file_type="Image",
                        helmet_count=helmet_count,
                        without_helmet_count=no_helmet_count,
                        confidence=avg_conf,
                        result=result_str
                    )
                    if rec_id:
                        st.success(f"✅ Record #{rec_id} successfully saved to SQLite Database!")
                    else:
                        st.error("Failed to save detection log.")

# ==========================================
# PAGE 3: VIDEO DETECTION
# ==========================================
elif selected_page == "Video Detection":
    render_header()
    st.subheader("🎥 Video Helmet Detection")
    st.caption("Upload a video file (MP4, AVI) to process surveillance footage frame-by-frame.")

    conf_threshold = st.slider("Video Confidence Threshold", min_value=0.10, max_value=0.90, value=0.50, step=0.05)
    uploaded_video = st.file_uploader("Upload Video File...", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        temp_input_path = os.path.join(UPLOAD_DIR, uploaded_video.name)
        temp_output_path = os.path.join(RESULT_DIR, f"detected_{uploaded_video.name}")

        with open(temp_input_path, "wb") as f:
            f.write(uploaded_video.read())

        st.video(temp_input_path)

        if st.button("🚀 Process Video with YOLOv8", type="primary"):
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def update_progress(percent):
                progress_bar.progress(percent)
                status_text.text(f"Processing Video Frame... {int(percent * 100)}% Completed")

            with st.spinner("Analyzing Video Stream..."):
                results_summary = detector.detect_video(
                    input_video_path=temp_input_path,
                    output_video_path=temp_output_path,
                    conf_threshold=conf_threshold,
                    progress_callback=update_progress
                )

            st.success("🎉 Video Processing Completed Successfully!")

            st.markdown("### 📊 Video Analysis Metrics")
            v1, v2, v3, v4 = st.columns(4)
            v1.metric("Total Frames Processed", results_summary["total_frames"])
            v2.metric("Total Helmet Detections", results_summary["helmet_count"])
            v3.metric("Total Violation Detections", results_summary["without_helmet_count"])
            v4.metric("Avg Video Confidence", f"{round(results_summary['avg_confidence'] * 100, 1)}%")

            v_result = "Violation" if results_summary["without_helmet_count"] > 0 else "Safe"
            db.save_detection(
                filename=uploaded_video.name,
                file_type="Video",
                helmet_count=results_summary["helmet_count"],
                without_helmet_count=results_summary["without_helmet_count"],
                confidence=results_summary["avg_confidence"],
                result=v_result
            )

            if os.path.exists(temp_output_path):
                st.markdown("##### 🎬 Annotated Output Video")
                st.video(temp_output_path)
                with open(temp_output_path, "rb") as file:
                    st.download_button(
                        label="📥 Download Annotated Video",
                        data=file,
                        file_name=f"detected_{uploaded_video.name}",
                        mime="video/mp4"
                    )

# ==========================================
# PAGE 4: WEBCAM DETECTION
# ==========================================
elif selected_page == "Webcam Detection":
    render_header()
    st.subheader("📹 Real-Time Webcam Detection")
    st.caption("Capture snapshot frames from your live camera to evaluate helmet compliance.")

    conf_threshold = st.slider("Webcam Detection Threshold", min_value=0.10, max_value=0.90, value=0.50, step=0.05)

    img_file_buffer = st.camera_input("Take a snapshot for real-time safety inspection")

    if img_file_buffer is not None:
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        with st.spinner("Analyzing Live Snapshot..."):
            annotated_bgr, detections, counts, avg_conf = detector.process_frame(cv2_img, conf_threshold=conf_threshold)
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### 📷 Raw Camera Capture")
            st.image(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB), width=700)
        with col2:
            st.markdown("##### 🎯 Live AI Detection Overlay")
            st.image(annotated_rgb, width=700)

        helmet_count = counts.get("With Helmet", 0)
        no_helmet_count = counts.get("Without Helmet", 0)
        result_str = "Violation" if no_helmet_count > 0 else ("Safe" if helmet_count > 0 else "No Detection")

        st.markdown("### 📊 Snapshot Analysis")
        w1, w2, w3, w4 = st.columns(4)
        w1.metric("Result", result_str)
        w2.metric("Helmets Detected", helmet_count)
        w3.metric("No Helmet Violations", no_helmet_count)
        w4.metric("Avg Confidence", f"{round(avg_conf * 100, 1)}%")

        if st.button("💾 Save Webcam Snapshot to History"):
            rec_id = db.save_detection(
                filename="Webcam_Snapshot.jpg",
                file_type="Webcam",
                helmet_count=helmet_count,
                without_helmet_count=no_helmet_count,
                confidence=avg_conf,
                result=result_str
            )
            st.success(f"Snapshot recorded to SQLite history database (ID: #{rec_id}).")

# ==========================================
# PAGE 5: DASHBOARD ANALYTICS
# ==========================================
elif selected_page == "Dashboard":
    render_header()
    st.subheader("📊 Safety Analytics & Intelligence Dashboard")
    st.caption("Interactive charts visualizing helmet compliance data.")
    
    # Lazy import heavy visualization libraries
    import pandas as pd
    import plotly.express as px

    stats = db.get_analytics_summary()
    df = db.get_history()

    # Metric Cards Row
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Detections", stats["total_detections"])
    m2.metric("Helmet Count", stats["helmet_count"])
    m3.metric("Violation Count", stats["without_helmet_count"])
    m4.metric("Avg Confidence", f"{stats['avg_confidence']}%")
    m5.metric("Compliance Rate", f"{stats['compliance_rate']}%")

    st.markdown("---")

    if not df.empty:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🥧 Helmet vs Violation Distribution")
            pie_data = pd.DataFrame({
                "Category": ["With Helmet", "Without Helmet"],
                "Count": [stats["helmet_count"], stats["without_helmet_count"]]
            })
            fig_pie = px.pie(
                pie_data,
                values="Count",
                names="Category",
                color="Category",
                color_discrete_map={"With Helmet": "#10b981", "Without Helmet": "#ef4444"},
                hole=0.4
            )
            fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f3f4f6")
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            st.markdown("#### 📈 Detection Trend Over Time")
            df["timestamp_dt"] = pd.to_datetime(df["timestamp"])
            fig_line = px.line(
                df,
                x="timestamp_dt",
                y=["helmet_count", "without_helmet_count"],
                labels={"value": "Count", "timestamp_dt": "Timestamp"},
                color_discrete_sequence=["#38bdf8", "#f87171"]
            )
            fig_line.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f3f4f6")
            st.plotly_chart(fig_line, use_container_width=True)

        st.markdown("#### 📁 File Type Distribution")
        fig_bar = px.bar(
            df["file_type"].value_counts().reset_index(),
            x="file_type",
            y="count",
            labels={"file_type": "Source Type", "count": "Total Detections"},
            color="file_type",
            color_discrete_sequence=["#60a5fa", "#c084fc", "#f472b6"]
        )
        fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f3f4f6")
        st.plotly_chart(fig_bar, use_container_width=True)

    else:
        st.info("ℹ️ No detection data available in database yet. Run an Image, Video, or Webcam detection to populate analytics charts.")

# ==========================================
# PAGE 6: DETECTION HISTORY
# ==========================================
elif selected_page == "Detection History":
    render_header()
    st.subheader("📜 Detection Audit History")

    df = db.get_history()

    if not df.empty:
        col1, col2 = st.columns([8, 2])
        with col1:
            st.markdown(f"Displaying **{len(df)}** historical records from SQLite Database.")
        with col2:
            if st.button("🗑️ Clear History", type="secondary"):
                if db.delete_history():
                    st.success("History database cleared.")
                    st.rerun()

        # Display Data Table
        st.dataframe(
            df,
            column_config={
                "id": "Log ID",
                "filename": "File Name",
                "file_type": "Source",
                "helmet_count": "Helmets Worn",
                "without_helmet_count": "Violations",
                "confidence": st.column_config.NumberColumn("Avg Confidence", format="%.2f"),
                "result": "Status Result",
                "timestamp": "Date & Time"
            },
            use_container_width=True,
            hide_index=True
        )

        # Export CSV Button
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Audit History to CSV",
            data=csv_data,
            file_name="helmet_detection_history.csv",
            mime="text/csv"
        )
    else:
        st.info("ℹ️ Detection history is empty. Process images or videos to store history logs.")

# ==========================================
# PAGE 7: ABOUT
# ==========================================
elif selected_page == "About":
    render_header()
    st.markdown("""
    <div class="glass-card">
        <h3>ℹ️ About AI Helmet Detection System</h3>
        <p>This web application is a complete production-grade computer vision solution built to monitor and enforce safety helmet usage across industrial, construction, and traffic environments.</p>

        <h4>⚙️ Technology Stack</h4>
        <ul>
            <li><b>AI Architecture:</b> YOLOv8 (Ultralytics) Deep Learning Neural Network</li>
            <li><b>Frontend Framework:</b> Streamlit (Python 3.11)</li>
            <li><b>Computer Vision:</b> OpenCV & Pillow</li>
            <li><b>Database:</b> SQLite3</li>
            <li><b>Data Visualization:</b> Plotly & Pandas</li>
            <li><b>Deployment:</b> Render & GitHub</li>
        </ul>

        <h4>🏷️ Object Classes</h4>
        <ul>
            <li><span style="color:#10b981; font-weight:bold;">Class 0: With Helmet</span> - Person verified wearing safety helmet.</li>
            <li><span style="color:#ef4444; font-weight:bold;">Class 1: Without Helmet</span> - Safety violation detected (Person not wearing helmet).</li>
        </ul>

        <h4>👨‍💻 Developer & Author</h4>
        <p>Developed by <b>rithikesh18-rk</b></p>
    </div>
    """, unsafe_allow_html=True)