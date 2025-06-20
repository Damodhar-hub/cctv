import streamlit as st
import tempfile
import cv2
import os
from ultralytics import YOLO
import numpy as np

st.title("Forensic CCTV Analysis")

uploaded_file = st.file_uploader("Upload CCTV video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    st.video(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        temp_video.write(uploaded_file.read())
        video_path = temp_video.name

    st.success("✅ Video successfully loaded. Starting processing...")

    # Load YOLO model
    yolo = YOLO('yolov8n.pt')
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    st.write(f"📹 FPS: {fps}, Total Frames: {frame_count}, Resolution: {int(width)}x{int(height)}")

    frame_num = 0
    detections = 0
    progress_bar = st.progress(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        if frame_num % int(fps * 2) == 0:
            results = yolo(frame, conf=0.4, verbose=False)
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        cls_id = int(box.cls[0].cpu().numpy())
                        label = yolo.names[cls_id]
                        if label == "person":
                            detections += 1
                            break

        if frame_num % 50 == 0:
            progress = int((frame_num / frame_count) * 100)
            progress_bar.progress(min(progress, 100))

    cap.release()
    st.success(f"✅ Processing complete. Total person detections (approx): {detections}")
else:
    st.info("📂 Please upload a CCTV video file to start analysis.")