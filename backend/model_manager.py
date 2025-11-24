# model_manager.py
import os
import time
import base64
import cv2
import logging
import threading
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from ultralytics import YOLO  # type: ignore
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self,
                 model_paths: Optional[List[str]] = None,
                 conf_threshold: float = 0.45,
                 iou_threshold: float = 0.45,
                 imgsz: int = 1280,
                 max_det: int = 500,
                 enable_tta: bool = True,
                 use_half_if_cuda: bool = True):
        self.lock = threading.Lock()
        self.model = None
        self.class_names: Dict[int, str] = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_half = use_half_if_cuda and (self.device == "cuda")
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.max_det = max_det
        self.enable_tta = enable_tta
        self.warmup_done = False
        self.model_paths = model_paths or [
            "yolov8n.pt",  # Nano - fastest
            "yolov8s.pt",  # Small - balanced
            "yolov8m.pt",  # Medium - good accuracy
            "yolov8l.pt",  # Large - better accuracy
            "yolov8x.pt",  # Extra large - best accuracy
            "models/best.pt",
            "runs/hazard_model/train/weights/best.pt",
            "models/garbage_detection_model.pt"
        ]
        self.hazard_severity = {
            'fire': 'critical', 'smoke': 'critical', 'gas_leak': 'critical',
            'chemical_spill': 'high', 'electrical_hazard': 'high',
            'exposed_wires': 'high', 'sharp_object': 'high', 'broken_glass': 'high',
            'biohazard_waste': 'high', 'medical_waste': 'medium',
            'overflowing_trash': 'medium', 'wet_floor': 'medium',
            'slippery_surface': 'medium', 'oil_spill': 'medium',
            'fallen_debris': 'low', 'plastic_bag': 'high', 'plastic_wrap': 'medium'
        }
        self.waste_keywords = [
            'trash', 'garbage', 'waste', 'litter', 'debris', 'rubbish'
        ]
        self.safe_keywords = [
            # Foods
            'banana', 'apple', 'orange', 'carrot', 'broccoli', 'pizza', 'donut',
            'cake', 'sandwich', 'hot dog', 'hamburger', 'fruit', 'vegetable',
            # People & Animals
            'person', 'people', 'dog', 'cat', 'bird', 'horse', 'sheep', 'cow',
            # Furniture & Objects
            'chair', 'table', 'couch', 'bed', 'desk', 'bench', 'plant', 'vase',
            'book', 'clock', 'laptop', 'keyboard', 'mouse', 'phone', 'tv',
            # Vehicles
            'car', 'truck', 'bus', 'bicycle', 'motorcycle', 'train',
            # Nature
            'tree', 'flower', 'grass', 'sky', 'cloud'
        ]
        self._load_model()
        logger.info(f"ModelManager ready on device={self.device} half={self.use_half}")

    def _load_model(self):
        with self.lock:
            for p in self.model_paths:
                try:
                    if os.path.exists(p):
                        logger.info(f"Loading model -> {p}")
                        self.model = YOLO(p)
                        break
                except Exception:
                    continue
            if self.model is None:
                logger.info("No local model found, attempting to download yolov8x-seg")
                self.model = YOLO("yolov8x-seg.pt")
            if hasattr(self.model, "names"):
                self.class_names = {int(k): v for k, v in self.model.names.items()} if isinstance(self.model.names, dict) else {i: n for i, n in enumerate(self.model.names)}
            else:
                self.class_names = {}
            try:
                if self.device == "cuda":
                    self.model.to("cuda")
                if self.use_half and hasattr(self.model, "model") and hasattr(self.model.model, "half"):
                    try:
                        self.model.model.half()  # type: ignore
                    except Exception:
                        pass
            except Exception:
                pass
            self._warmup()

    def _warmup(self):
        if self.warmup_done or self.model is None:
            return
        try:
            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            start = time.time()
            self.model.predict(source=dummy, imgsz=self.imgsz, conf=0.001, iou=0.5, max_det=1)
            self.warmup_time = time.time() - start
            self.warmup_done = True
            logger.info(f"Model warmup done in {self.warmup_time:.3f}s")
        except Exception:
            self.warmup_time = 0.0
            self.warmup_done = True

    def _preprocess(self, image_path: str) -> Tuple[np.ndarray, Tuple[int, int]]:
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR) if os.path.isfile(image_path) else None
        if img is None:
            img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Unable to read image")
        h, w = img.shape[:2]
        max_side = max(h, w)
        scale = 1.0
        if max_side > self.imgsz:
            scale = self.imgsz / max_side
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        img = cv2.bilateralFilter(img, d=5, sigmaColor=75, sigmaSpace=75)
        return img, (h, w)

    def _size_boost_confidence(self, box_xyxy: List[float], conf: float, orig_shape: Tuple[int, int]) -> float:
        x1, y1, x2, y2 = box_xyxy
        area = (x2 - x1) * (y2 - y1)
        h, w = orig_shape
        img_area = h * w
        rel_area = (area / img_area) if img_area > 0 else 0
        if rel_area > 0.25:
            return min(1.0, conf + 0.12)
        if rel_area > 0.05:
            return min(1.0, conf + 0.05)
        if rel_area < 0.001:
            return max(0.0, conf - 0.12)
        return conf

    def detect_and_categorize(self, image_path: str, min_conf: Optional[float] = None) -> Dict:
        if self.model is None:
            raise RuntimeError("Model not loaded")
        start_total = time.time()
        img, orig_shape = self._preprocess(image_path)
        preprocess_time = time.time() - start_total
        conf = self.conf_threshold if min_conf is None else max(min_conf, self.conf_threshold)
        predict_kwargs = {
            "source": img,
            "imgsz": self.imgsz,
            "conf": conf,
            "iou": self.iou_threshold,
            "max_det": self.max_det,
            "augment": self.enable_tta,
            "verbose": False
        }
        if self.device == "cuda" and self.use_half:
            predict_kwargs["half"] = True
        with self.lock:
            t0 = time.time()
            results = self.model.predict(**predict_kwargs)
            infer_time = time.time() - t0
        detections = []
        annotated = img.copy()
        severity_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        detected_classes = []
        for res in results:
            boxes = getattr(res, "boxes", None)
            masks = getattr(res, "masks", None)
            if boxes is None or len(boxes) == 0:
                continue
            for i, box in enumerate(boxes):
                try:
                    cls_id = int(box.cls.cpu().numpy()[0]) if hasattr(box, "cls") else int(box.cls[0])
                except Exception:
                    cls_id = int(box.cls[0]) if hasattr(box, "cls") else 0
                class_name = self.class_names.get(cls_id, str(cls_id)).lower()
                detected_classes.append(class_name)
                conf_raw = float(box.conf.cpu().numpy()[0]) if hasattr(box, "conf") else float(box.conf[0])
                xyxy = box.xyxy.cpu().numpy()[0].tolist() if hasattr(box, "xyxy") else box.xyxy[0].tolist()
                conf_adj = self._size_boost_confidence(xyxy, conf_raw, orig_shape)
                if conf_adj < conf:
                    continue
                x1, y1, x2, y2 = map(int, xyxy)
                
                # Check if item is explicitly safe
                is_safe = any(k in class_name for k in self.safe_keywords)
                
                # Check if item is waste/trash
                is_waste = any(k in class_name for k in self.waste_keywords)
                
                # Check for critical hazards
                is_critical_hazard = any(k in class_name for k in ['fire', 'smoke', 'flame', 'gas'])
                
                # Check for high hazards
                is_high_hazard = any(k in class_name for k in [
                    'chemical', 'oil', 'battery', 'broken', 'sharp', 'glass',
                    'needle', 'syringe', 'biohazard', 'electrical', 'wire'
                ])
                
                # Classify the detection
                if is_safe:
                    sev = 'safe'
                    health_status = "Safe"
                elif is_critical_hazard:
                    sev = 'critical'
                    health_status = "Hazardous"
                    severity_counts[sev] = severity_counts.get(sev, 0) + 1
                elif is_high_hazard:
                    sev = 'high'
                    health_status = "Hazardous"
                    severity_counts[sev] = severity_counts.get(sev, 0) + 1
                elif is_waste:
                    sev = 'medium'
                    health_status = "Hazardous"
                    severity_counts[sev] = severity_counts.get(sev, 0) + 1
                else:
                    # Default to safe for unrecognized common objects
                    sev = 'safe'
                    health_status = "Safe"
                color = (0, 255, 0) if health_status == "Safe" else ((0, 0, 255) if sev in ['critical', 'high'] else (0, 165, 255))
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name} {conf_adj:.2f}"
                cv2.putText(annotated, label, (x1, max(y1 - 6, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                area = (x2 - x1) * (y2 - y1)
                mask_poly = None
                if masks is not None and hasattr(masks, "data") and len(masks.data) > i:
                    try:
                        mask = masks.data[i].cpu().numpy()
                        mask_poly = mask
                    except Exception:
                        mask_poly = None
                detections.append({
                    "class": class_name,
                    "confidence": round(conf_adj, 4),
                    "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                    "bbox_area": float(area),
                    "health_status": health_status,
                    "severity": sev,
                    "mask_present": mask_poly is not None
                })
        total_detections = len(detections)
        hazardous_count = sum(1 for d in detections if d["health_status"] == "Hazardous")
        safe_count = total_detections - hazardous_count
        avg_confidence = round((sum(d["confidence"] for d in detections) / total_detections) if total_detections else 0.0, 4)
        high_conf = sum(1 for d in detections if d["confidence"] >= 0.70)
        med_conf = sum(1 for d in detections if 0.50 <= d["confidence"] < 0.70)
        low_conf = sum(1 for d in detections if d["confidence"] < 0.50)
        weighted_score = (high_conf * 1.0) + (med_conf * 0.85) + (low_conf * 0.60) if total_detections else 0.0
        model_accuracy = round((weighted_score / total_detections) * 100, 2) if total_detections else 0.0
        overall_risk = "safe"
        if severity_counts.get('critical', 0) > 0:
            overall_risk = "critical"
        elif severity_counts.get('high', 0) > 0:
            overall_risk = "high"
        elif severity_counts.get('medium', 0) > 0:
            overall_risk = "medium"
        else:
            overall_risk = "safe"
        _, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        img_base64 = base64.b64encode(buf).decode("utf-8")
        elapsed = time.time() - start_total
        fps = round((1.0 / elapsed) if elapsed > 0 else 0.0, 2)
        return {
            "total_detections": total_detections,
            "hazardous_count": hazardous_count,
            "safe_count": safe_count,
            "overall_assessment": "Safe" if hazardous_count == 0 else "Hazardous",
            "overall_risk_level": overall_risk,
            "severity_breakdown": severity_counts,
            "average_confidence": avg_confidence,
            "model_accuracy": model_accuracy,
            "high_confidence_detections": high_conf,
            "medium_confidence_detections": med_conf,
            "detections": detections,
            "annotated_image": img_base64,
            "performance": {
                "device": self.device,
                "use_half": self.use_half,
                "imgsz": self.imgsz,
                "inference_time_s": round(infer_time, 4),
                "total_elapsed_s": round(elapsed, 4),
                "fps": fps,
                "warmup_time_s": round(getattr(self, "warmup_time", 0.0), 4),
                "preprocess_time_s": round(preprocess_time, 4)
            },
            "model_settings": {
                "model_path": str(getattr(self.model, "ckpt_path", "unknown")),
                "confidence_threshold": self.conf_threshold,
                "iou_threshold": self.iou_threshold,
                "max_det": self.max_det,
                "tta_enabled": self.enable_tta
            }
        }

    def update_weights(self, new_model_path: str):
        with self.lock:
            if not os.path.exists(new_model_path):
                raise FileNotFoundError(f"Model not found: {new_model_path}")
            logger.info(f"Updating model to {new_model_path}")
            self.model = YOLO(new_model_path)
            if self.device == "cuda":
                try:
                    self.model.to("cuda")
                except Exception:
                    pass
            if self.use_half and hasattr(self.model, "model") and hasattr(self.model.model, "half"):
                try:
                    self.model.model.half()  # type: ignore
                except Exception:
                    pass
            if hasattr(self.model, "names"):
                self.class_names = {int(k): v for k, v in self.model.names.items()} if isinstance(self.model.names, dict) else {i: n for i, n in enumerate(self.model.names)}
            self.warmup_done = False
            self._warmup()
            logger.info("Model updated and warmed up")

    def update_inference_settings(self, conf: Optional[float] = None, iou: Optional[float] = None, imgsz: Optional[int] = None, tta: Optional[bool] = None):
        if conf is not None:
            self.conf_threshold = float(min(max(conf, 0.0), 1.0))
        if iou is not None:
            self.iou_threshold = float(min(max(iou, 0.0), 1.0))
        if imgsz is not None:
            self.imgsz = int(imgsz)
            self.warmup_done = False
        if tta is not None:
            self.enable_tta = bool(tta)

    def get_class_names(self) -> Dict[int, str]:
        return self.class_names

    def get_model_info(self) -> Dict:
        return {
            "loaded": self.model is not None,
            "num_classes": len(self.class_names),
            "device": self.device,
            "use_half": self.use_half,
            "imgsz": self.imgsz,
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "max_det": self.max_det,
            "warmup_done": self.warmup_done
        }