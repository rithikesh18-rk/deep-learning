import os
import cv2
import numpy as np
from PIL import Image
import logging

# Ensure YOLO_CONFIG_DIR is set to /tmp/Ultralytics for cloud deployments (Render, etc.)
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"
os.makedirs("/tmp/Ultralytics", exist_ok=True)

from ultralytics import YOLO
from config import (
    MODEL_PATH, CLASS_NAMES, CLASS_COLORS,
    DEFAULT_CONFIDENCE, DEFAULT_IOU, TARGET_IMAGE_SIZE, RESULT_DIR
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HelmetDetector:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = os.path.abspath(model_path)
        self.model = None
        self.model_loaded = False
        self.load_error = None
        self.class_names = {}
        self.load_model()

    def load_model(self):
        """Loads the YOLOv8 model from specified weights path."""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"Model file not found at {self.model_path}. Place your trained best.pt in the models/ directory."
                )

            self.model = YOLO(self.model_path)
            logger.info(f"Loaded YOLOv8 model from {self.model_path}")

            if hasattr(self.model, 'names') and self.model.names:
                self.class_names = dict(self.model.names)
                logger.info(f"Model class names: {self.class_names}")
            else:
                self.class_names = dict(CLASS_NAMES)
                logger.warning("Model has no names attribute, using config CLASS_NAMES.")

            self.model_loaded = True
            self.load_error = None
        except Exception as e:
            self.load_error = str(e)
            self.model_loaded = False
            logger.error(f"Error loading YOLO model: {e}")

    def draw_boxes(self, frame, detections):
        """
        Draw bounding boxes, labels, and confidence tags on the frame.
        detections: list of dicts with keys: box, class_id, class_name, conf
        """
        annotated_frame = frame.copy()
        h, w = annotated_frame.shape[:2]

        for det in detections:
            x1, y1, x2, y2 = det['box']
            cls_id = det['class_id']
            conf = det['conf']
            class_name = det.get('class_name', self.class_names.get(cls_id, f"Class {cls_id}"))

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            color = CLASS_COLORS.get(cls_id, (0, 225, 100) if cls_id == 0 else (40, 40, 255))

            thickness = max(2, int(min(h, w) / 250))
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

            corner_len = max(10, int(min(x2 - x1, y2 - y1) * 0.15))
            cv2.line(annotated_frame, (x1, y1), (x1 + corner_len, y1), (255, 255, 255), thickness + 1)
            cv2.line(annotated_frame, (x1, y1), (x1, y1 + corner_len), (255, 255, 255), thickness + 1)
            cv2.line(annotated_frame, (x2, y2), (x2 - corner_len, y2), (255, 255, 255), thickness + 1)
            cv2.line(annotated_frame, (x2, y2), (x2, y2 - corner_len), (255, 255, 255), thickness + 1)

            label = f"{class_name} {int(conf * 100)}%"
            font_scale = max(0.5, min(w, h) / 1000.0)
            font_thickness = max(1, int(font_scale * 2))

            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

            banner_y1 = max(0, y1 - text_h - 12)
            banner_y2 = y1

            cv2.rectangle(annotated_frame, (x1, banner_y1), (x1 + text_w + 12, banner_y2), color, -1)
            cv2.putText(annotated_frame, label, (x1 + 6, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

        return annotated_frame

    def process_frame(self, frame: np.ndarray, conf_threshold: float = DEFAULT_CONFIDENCE, iou_threshold: float = DEFAULT_IOU):
        """
        Process a single image frame (BGR format numpy array) with YOLOv8.
        Returns:
            annotated_frame (np.ndarray BGR)
            detections (list)
            counts (dict)
            avg_conf (float)
        """
        if not self.model_loaded or self.model is None:
            raise RuntimeError(f"Model not loaded: {self.load_error}")

        results = self.model(
            frame,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=TARGET_IMAGE_SIZE,
            verbose=False
        )

        detections = []
        counts = {}
        total_conf = 0.0

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = self.class_names.get(cls_id, f"Class {cls_id}")

                counts[class_name] = counts.get(class_name, 0) + 1
                total_conf += conf

                detections.append({
                    "box": [x1, y1, x2, y2],
                    "class_id": cls_id,
                    "class_name": class_name,
                    "conf": conf
                })

        avg_conf = (total_conf / len(detections)) if len(detections) > 0 else 0.0
        annotated_frame = self.draw_boxes(frame, detections)

        return annotated_frame, detections, counts, avg_conf

    def detect_image(self, image_input, conf_threshold: float = DEFAULT_CONFIDENCE, iou_threshold: float = DEFAULT_IOU):
        """
        Detect helmets in an input image (PIL Image or NumPy array or path string).
        Returns:
            annotated_image (np.ndarray - RGB format)
            detections (list)
            counts (dict)
            avg_conf (float)
        """
        if isinstance(image_input, str):
            frame = cv2.imread(image_input)
            if frame is None:
                raise ValueError(f"Could not read image from path: {image_input}")
        elif isinstance(image_input, Image.Image):
            frame = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            frame = image_input.copy()
            if frame is None or frame.size == 0:
                raise ValueError("Empty numpy array provided as image")
        else:
            raise ValueError(f"Unsupported image format: {type(image_input)}")

        annotated_bgr, detections, counts, avg_conf = self.process_frame(
            frame,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold
        )
        del frame
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        del annotated_bgr
        return annotated_rgb, detections, counts, avg_conf

    def detect_video(self, input_video_path: str, output_video_path: str, conf_threshold: float = DEFAULT_CONFIDENCE, iou_threshold: float = DEFAULT_IOU, progress_callback=None):
        """Processes a video file frame-by-frame and saves the annotated video."""
        if not self.model_loaded or self.model is None:
            raise RuntimeError(f"Model not loaded: {self.load_error}")

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {input_video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        total_helmets = 0
        total_without_helmets = 0
        conf_accumulator = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            annotated_frame, detections, counts, avg_conf = self.process_frame(
                frame,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold
            )
            out.write(annotated_frame)

            total_helmets += counts.get("With Helmet", 0)
            total_without_helmets += counts.get("Without Helmet", 0)
            if avg_conf > 0:
                conf_accumulator.append(avg_conf)

            frame_idx += 1
            if progress_callback:
                progress_callback(frame_idx / total_frames)

        cap.release()
        out.release()

        overall_avg_conf = float(np.mean(conf_accumulator)) if conf_accumulator else 0.0
        return {
            "output_path": output_video_path,
            "total_frames": frame_idx,
            "helmet_count": total_helmets,
            "without_helmet_count": total_without_helmets,
            "avg_confidence": overall_avg_conf
        }

    def detect_webcam(self, frame, conf_threshold: float = DEFAULT_CONFIDENCE, iou_threshold: float = DEFAULT_IOU):
        """Webcam real-time frame processing interface."""
        return self.detect_image(frame, conf_threshold=conf_threshold, iou_threshold=iou_threshold)


# Global singleton detector instance
_detector_instance = None


def get_detector():
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = HelmetDetector()
    return _detector_instance


if __name__ == "__main__":
    detector = HelmetDetector()
    print("HelmetDetector initialized. Loaded:", detector.model_loaded)
    print("Class Names:", detector.class_names)
