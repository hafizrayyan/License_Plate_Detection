import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import os
import subprocess  # Added for web video conversion

# =========================
# LOAD MODELS
# =========================

print("Loading YOLO model...")
model = YOLO("best.pt")

print("Loading EasyOCR...")
reader = easyocr.Reader(['en'], gpu=True)

print("Models loaded successfully!")

# =========================
# MAIN DETECTION FUNCTION
# =========================

def detect_license_plate(video_file):

    try:
        # We save OpenCV's raw output to a temp file first
        temp_output_path = "temp_processed.mp4"
        final_output_path = "processed_output.mp4"

        cap = cv2.VideoCapture(video_file)

        # Check video opened
        if not cap.isOpened():
            raise ValueError("Cannot open uploaded video.")

        # Video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Fix FPS issue
        if fps == 0:
            fps = 30

        print(f"Video Loaded: {width}x{height} @ {fps} FPS")

        # Codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # Write to temp path initially
        out = cv2.VideoWriter(
            temp_output_path,
            fourcc,
            fps,
            (width, height)
        )

        frame_count = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            print(f"Processing Frame: {frame_count}")

            # Resize for performance
            frame = cv2.resize(frame, (width, height))

            # YOLO inference
            results = model(frame, verbose=False)

            for result in results:

                if result.boxes is None:
                    continue

                for box in result.boxes:

                    class_id = int(box.cls.cpu().numpy())

                    # COCO car class = 2
                    if class_id != 2:
                        continue

                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy.cpu().numpy()[0]
                    )

                    # Bounds safety
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(width, x2)
                    y2 = min(height, y2)

                    if x2 <= x1 or y2 <= y1:
                        continue

                    # Draw car box
                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (255, 0, 0),
                        2
                    )

                    # Crop car
                    car_img = frame[y1:y2, x1:x2]

                    if car_img.size == 0:
                        continue

                    car_height = y2 - y1

                    # Lower region
                    plate_start = int(0.65 * car_height)

                    plate_region = car_img[plate_start:, :]

                    if plate_region.size == 0:
                        continue

                    try:

                        # OCR
                        ocr_results = reader.readtext(
                            plate_region,
                            detail=1
                        )

                        all_boxes = []
                        all_texts = []

                        for (bbox, text, conf) in ocr_results:

                            if conf > 0.25:

                                all_boxes.append(bbox)
                                all_texts.append(text)

                        if not all_boxes:
                            continue

                        plate_text = "".join(all_texts).strip()

                        points = np.concatenate(all_boxes, axis=0)

                        px1 = int(np.min(points[:, 0])) - 3
                        py1 = int(np.min(points[:, 1])) - 3

                        px2 = int(np.max(points[:, 0])) + 3
                        py2 = int(np.max(points[:, 1])) + 3

                        frame_x1 = x1 + px1
                        frame_y1 = y1 + plate_start + py1

                        frame_x2 = x1 + px2
                        frame_y2 = y1 + plate_start + py2

                        # Draw plate box
                        cv2.rectangle(
                            frame,
                            (frame_x1, frame_y1),
                            (frame_x2, frame_y2),
                            (0, 255, 0),
                            3
                        )

                        # Draw text background
                        text_size = cv2.getTextSize(
                            plate_text,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            2
                        )[0]

                        cv2.rectangle(
                            frame,
                            (frame_x1, frame_y1 - 30),
                            (
                                frame_x1 + text_size[0] + 10,
                                frame_y1
                            ),
                            (0, 255, 0),
                            -1
                        )

                        # Put text
                        cv2.putText(
                            frame,
                            plate_text,
                            (frame_x1 + 5, frame_y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 0),
                            2
                        )

                        print(f"Detected Plate: {plate_text}")

                    except Exception as ocr_error:
                        print("OCR ERROR:", ocr_error)

            out.write(frame)

        cap.release()
        out.release()

        # ---------------------------------------------------------
        # NEW: Convert Video to H.264 for Browser Compatibility
        # ---------------------------------------------------------
        print("Converting video format for browser compatibility...")
        if os.path.exists(final_output_path):
            os.remove(final_output_path)

        # Run FFmpeg transcode
        subprocess.run([
            'ffmpeg', '-y', '-i', temp_output_path,
            '-vcodec', 'libx264', '-crf', '23', final_output_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Clean up the original OpenCV temporary video file
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
        # ---------------------------------------------------------

        print("Processing complete!")

        # Verify final output exists
        if not os.path.exists(final_output_path):
            raise ValueError("Output video was not created.")

        return final_output_path

    except Exception as e:

        print("MAIN ERROR:", e)

        return None


# =========================
# GRADIO UI
# =========================

demo = gr.Interface(
    fn=detect_license_plate,
    inputs=gr.Video(label="Upload Video"),
    outputs=gr.Video(label="Processed Video"),
    title="AI License Plate Detection",
    description="Upload a video to detect license plates using YOLO + EasyOCR . Built by Hafiz Rayyan",
    theme=gr.themes.Soft()
)

# IMPORTANT FOR HUGGING FACE
demo.launch()
