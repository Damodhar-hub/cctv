import streamlit as st
import tempfile
import cv2
import os
from ultralytics import YOLO
import numpy as np
import pandas as pd
import shutil
from datetime import datetime

st.title("📹 Forensic CCTV Person Detection & Report Generator")

uploaded_file = st.file_uploader("Upload CCTV video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    st.video(uploaded_file)

    with tempfile.TemporaryDirectory() as temp_dir:
        video_path = os.path.join(temp_dir, uploaded_file.name)
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())

        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        crops_dir = os.path.join(output_dir, "crops")
        os.makedirs(crops_dir, exist_ok=True)

        st.success("✅ Video successfully loaded. Starting processing...")

        # Load YOLO model
        yolo = YOLO('yolov8n.pt')
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        st.write(f"🎞️ FPS: {fps:.2f}, Total Frames: {frame_count}, Resolution: {int(width)}x{int(height)}")

        frame_num = 0
        person_detections = []
        annotated_path = os.path.join(output_dir, "annotated.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_vid = cv2.VideoWriter(annotated_path, fourcc, fps, (int(width), int(height)))

        process_interval = int(fps * 2)
        progress_bar = st.progress(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_num += 1

            if frame_num % process_interval == 0:
                results = yolo(frame, conf=0.4, verbose=False)
                for result in results:
                    if result.boxes is not None:
                        for i, box in enumerate(result.boxes):
                            cls_id = int(box.cls[0].cpu().numpy())
                            label = yolo.names[cls_id]
                            if label == "person":
                                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                                crop = frame[y1:y2, x1:x2]
                                if crop.size > 0:
                                    crop_path = os.path.join(crops_dir, f"person_{frame_num}_{i}.jpg")
                                    cv2.imwrite(crop_path, crop)
                                    person_detections.append({
                                        "Frame": frame_num,
                                        "X1": x1, "Y1": y1, "X2": x2, "Y2": y2,
                                        "Width": x2 - x1,
                                        "Height": y2 - y1
                                    })
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                                    cv2.putText(frame, "Person", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            out_vid.write(frame)
            if frame_num % 50 == 0:
                progress = int((frame_num / frame_count) * 100)
                progress_bar.progress(min(progress, 100))

        cap.release()
        out_vid.release()

        st.success(f"✅ Processing complete. Total person detections (approx): {len(person_detections)}")

        # Create report CSV
        report_path = os.path.join(output_dir, "person_detections.xlsx")
        pd.DataFrame(person_detections).to_excel(report_path, index=False)

        # Create summary text
        summary_path = os.path.join(output_dir, "summary.txt")
        with open(summary_path, "w") as f:
            f.write("FORENSIC CCTV REPORT\n")
            f.write(f"Processed on: {datetime.now()}\n")
            f.write(f"Total Detections: {len(person_detections)}\n")
            f.write(f"Video FPS: {fps:.2f}, Total Frames: {frame_count}\n")
            f.write(f"Resolution: {int(width)}x{int(height)}\n")
            f.write(f"Process Interval: {process_interval} frames\n")

        # Zip everything
        zip_path = os.path.join(temp_dir, "forensic_output.zip")
        shutil.make_archive(zip_path.replace(".zip", ""), 'zip', output_dir)

        with open(zip_path, "rb") as f:
            st.download_button(
                label="📦 Download Full Report Package (ZIP)",
                data=f,
                file_name="forensic_output.zip",
                mime="application/zip"
            )