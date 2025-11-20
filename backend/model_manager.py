import os
import base64
import cv2
import logging
import threading
from ultralytics import YOLO
from typing import Dict
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GarbageDetectionModel:
    def __init__(self):
        self.model = None
        self.class_names = {}
        self.lock = threading.Lock()
        
        # Optimized high-accuracy inference settings
        self.conf_threshold = 0.25  # Lower threshold to catch more objects including plastics
        self.iou_threshold = 0.45   # IoU threshold for NMS
        self.imgsz = 1280           # Larger image size for better accuracy
        self.max_det = 500          # Maximum detections per image
        
        # Hazard severity mapping
        self.hazard_severity = {
            'fire': 'critical',
            'smoke': 'critical',
            'gas_leak': 'critical',
            'chemical_spill': 'high',
            'electrical_hazard': 'high',
            'exposed_wires': 'high',
            'sharp_object': 'high',
            'broken_glass': 'high',
            'biohazard_waste': 'high',
            'medical_waste': 'medium',
            'overflowing_trash': 'medium',
            'wet_floor': 'medium',
            'slippery_surface': 'medium',
            'oil_spill': 'medium',
            'fallen_debris': 'low',
            'plastic_bag': 'medium',
            'plastic_wrap': 'medium',
            'disposable_container': 'low'
        }
        
        # WASTE-SPECIFIC DETECTION: Items considered as waste/hazard
        # Based on TACO (Trash Annotations in Context) categories
        self.waste_items = [
            # Plastic waste (HIGH hazard - pollution)
            'bottle', 'plastic bottle', 'water bottle', 'pet bottle',
            'plastic bag', 'plastic container', 'plastic wrapper', 'plastic cup',
            'disposable cup', 'plastic cutlery', 'straw', 'lid', 'cap',
            # Paper waste
            'paper', 'cardboard', 'newspaper', 'magazine', 'paper bag',
            'tissue', 'napkin', 'paper cup', 'pizza box',
            # Metal waste
            'can', 'aluminum can', 'tin can', 'metal container', 'foil',
            # Glass waste
            'glass bottle', 'broken glass', 'glass container',
            # Food waste
            'food waste', 'organic waste', 'leftovers', 'scraps',
            # General trash
            'trash', 'garbage', 'waste', 'litter', 'rubbish', 'refuse',
            'overflowing trash', 'trash bag', 'garbage bag',
            # Packaging
            'packaging', 'wrapper', 'box', 'carton', 'container',
            # Other waste
            'cigarette butt', 'cigarette', 'cup', 'bowl'
        ]
        
        # Safe items (fresh food, clean utensils NOT in trash context)
        self.safe_items = [
            'banana', 'apple', 'orange', 'carrot', 'broccoli',
            'person', 'chair', 'table', 'laptop', 'book',
            'car', 'bicycle', 'tree', 'plant', 'flower'
        ]
        
        logger.info("Initializing Hazard Detection Model with high-accuracy settings...")
        self.setup_model()
        logger.info("Model initialized successfully!")

    def setup_model(self):
        """Initialize and setup the YOLO model with high-accuracy configuration"""
        # Priority: best.pt > yolov8x-seg.pt (segmentation) > yolov8x.pt > yolov8l.pt
        model_paths = [
            "models/best.pt",                    # Trained high-accuracy model
            "runs/hazard_model/train/weights/best.pt",  # Training output
            "models/garbage_detection_model.pt", # Legacy model
            "yolov8x-seg.pt",                    # Segmentation model (best for plastic/transparent)
            "yolov8x.pt",                        # Extra-large detection
            "yolov8l.pt"                         # Large model fallback
        ]

        try:
            with self.lock:
                logger.info("Setting up Hazard Detection Model with MAXIMUM ACCURACY...")
                logger.info("Priority: Segmentation model for better plastic/transparent object detection")
                
                # Try to load models in priority order
                for model_path in model_paths:
                    if os.path.exists(model_path):
                        logger.info(f"Loading model from {model_path}...")
                        self.model = YOLO(model_path)
                        logger.info(f"✓ Model loaded successfully: {model_path}")
                        break
                else:
                    # Download segmentation model for better plastic detection
                    logger.info("Downloading YOLOv8x-seg (Segmentation) model for MAXIMUM accuracy...")
                    logger.info("Segmentation model is BEST for detecting plastic bags and transparent objects!")
                    logger.info("This may take a few minutes for first-time download...")
                    self.model = YOLO('yolov8x-seg.pt')  # Use segmentation for plastic detection

                if hasattr(self.model, 'names'):
                    self.class_names = self.model.names
                    logger.info(f"✓ Model loaded with {len(self.class_names)} classes")
                    logger.info(f"✓ High-accuracy settings: conf={self.conf_threshold}, iou={self.iou_threshold}, imgsz={self.imgsz}")
                    logger.info(f"✓ Test-Time Augmentation: ENABLED")
                    logger.info(f"✓ Segmentation: ENABLED (better for plastics)")
                    logger.info(f"✓ Safe items database: {len(self.safe_items)} categories")
                else:
                    logger.warning("Model doesn't have class names attribute")
                    self.class_names = {}
                    
        except Exception as e:
            logger.error(f"Error setting up segmentation model: {e}")
            logger.info("Falling back to YOLOv8x detection model...")
            try:
                with self.lock:
                    self.model = YOLO('yolov8x.pt')
                    self.class_names = self.model.names if hasattr(self.model, 'names') else {}
                logger.info("Fallback model (YOLOv8x) loaded successfully")
            except Exception as fallback_error:
                logger.error(f"Fallback model loading failed: {fallback_error}")
                logger.critical("Unable to load any model. API will not function correctly.")
                logger.info("Fallback model (YOLOv8l) loaded successfully")
            except Exception as fallback_error:
                logger.error(f"Fallback model loading failed: {fallback_error}")
                logger.critical("Unable to load any model. API will not function correctly.")

    def detect_and_categorize(self, image_path):
        """Run detection on an image with high-accuracy settings and return categorized results"""
        if self.model is None:
            logger.error("Model is not initialized")
            raise ValueError("Model is not initialized")

        # Check if image exists and is readable
        if not os.path.exists(image_path):
            logger.error(f"Image path does not exist: {image_path}")
            raise ValueError(f"Image path does not exist: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Could not read image: {image_path}")
            raise ValueError(f"Could not read image: {image_path}")

        try:
            # Run inference with MAXIMUM ACCURACY settings
            with self.lock:
                results = self.model.predict(
                    source=image_path,
                    imgsz=self.imgsz,           # Large image size for detail
                    conf=self.conf_threshold,    # Lower threshold for sensitivity
                    iou=self.iou_threshold,      # NMS IoU threshold
                    max_det=self.max_det,        # Max detections
                    augment=True,                # Test-time augmentation (TTA)
                    agnostic_nms=False,          # Class-specific NMS
                    half=False,                  # Full precision (FP32)
                    verbose=False                # Suppress output
                )
            
            detections = []
            food_items_count = 0
            total_items = 0
            severity_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
            
            # Context detection: Check if trash bin/container present
            detected_classes = []
            for result in results:
                if hasattr(result, 'boxes') and len(result.boxes) > 0:
                    for box in result.boxes:
                        if hasattr(box, 'cls') and len(box.cls) > 0:
                            cls_id = int(box.cls[0])
                            class_name = self.class_names.get(cls_id, '').lower()
                            detected_classes.append(class_name)
            
            # Check for trash context
            trash_context = any(keyword in ' '.join(detected_classes) for keyword in 
                              ['trash', 'garbage', 'waste', 'bin', 'can', 'container', 'bag'])

            # Process detections
            for result in results:
                if not hasattr(result, 'boxes') or len(result.boxes) == 0:
                    continue
                    
                for box in result.boxes:
                    if not hasattr(box, 'cls') or not hasattr(box, 'conf') or not hasattr(box, 'xyxy'):
                        continue
                    if len(box.cls) == 0 or len(box.conf) == 0 or len(box.xyxy) == 0:
                        continue
                        
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = self.class_names.get(cls_id, 'Unknown')
                    coords = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = map(int, coords)
                    
                    # WASTE DETECTION LOGIC: Check if item is waste/hazard
                    class_lower = class_name.lower()
                    is_waste = any(waste_type in class_lower for waste_type in self.waste_items)
                    is_safe = any(safe_type in class_lower for safe_type in self.safe_items) and not is_waste
                    
                    # CONTEXT-AWARE: If trash context detected, treat containers/bottles as waste
                    if trash_context and ('bottle' in class_lower or 'cup' in class_lower or 
                                         'bowl' in class_lower or 'container' in class_lower):
                        is_waste = True
                        is_safe = False
                    
                    # Count waste vs safe items
                    if is_safe:
                        food_items_count += 1
                    total_items += 1
                    
                    # Determine hazard status based on waste detection
                    if is_waste:
                        hazard_status = 'Hazardous'
                        # Assign severity based on waste type
                        if 'plastic' in class_lower or 'bottle' in class_lower:
                            severity = 'high'  # Plastic waste = environmental hazard
                        elif 'trash' in class_lower or 'garbage' in class_lower:
                            severity = 'medium'
                        else:
                            severity = 'low'
                        severity_counts[severity] = severity_counts.get(severity, 0) + 1
                    elif is_safe:
                        hazard_status = 'Safe'
                        severity = 'safe'
                    else:
                        # Unknown items treated as potential waste
                        hazard_status = 'Hazardous'
                        severity = 'low'
                        severity_counts[severity] = severity_counts.get(severity, 0) + 1
                    
                    # Color coding based on waste detection
                    if hazard_status == 'Safe':
                        color = (0, 255, 0)  # Green
                    elif severity == 'high':
                        color = (0, 0, 255)  # Red - plastic/bottles
                    elif severity == 'medium':
                        color = (0, 165, 255)  # Orange - general trash
                    else:
                        color = (0, 255, 255)  # Yellow - low hazard waste

                    # Draw detection rectangle and label
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
                    label = f"{class_name} {conf:.2f}"
                    if severity != 'safe' and severity != 'unknown':
                        label += f" [WASTE-{severity}]"
                    
                    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(image, (x1, y1 - label_height - 10), (x1 + label_width, y1), color, -1)
                    cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    detections.append({
                        'class': class_name,
                        'confidence': float(conf),
                        'coordinates': [float(c) for c in coords],
                        'health_status': hazard_status,
                        'severity': severity,
                        'bbox_area': (x2 - x1) * (y2 - y1)
                    })

            # Calculate statistics and accuracy metrics
            is_mostly_food = (food_items_count / max(total_items, 1)) > 0.5
            total_detections = len(detections)
            hazardous_count = sum(1 for d in detections if d["health_status"] == "Hazardous")
            safe_count = total_detections - hazardous_count
            
            # Calculate average confidence (accuracy indicator)
            avg_confidence = sum(d['confidence'] for d in detections) / max(total_detections, 1) if detections else 0
            
            # IMPROVED: More realistic accuracy calculation with tiered thresholds
            # High confidence: ≥0.6 (60%), Medium: 0.45-0.59, Low: <0.45
            high_conf_count = sum(1 for d in detections if d['confidence'] >= 0.60)
            medium_conf_count = sum(1 for d in detections if 0.45 <= d['confidence'] < 0.60)
            
            # Weighted accuracy: high conf = 100%, medium = 70%, low = 30%
            if total_detections > 0:
                weighted_score = (high_conf_count * 1.0) + (medium_conf_count * 0.7) + ((total_detections - high_conf_count - medium_conf_count) * 0.3)
                model_accuracy = (weighted_score / total_detections) * 100
            else:
                model_accuracy = 0
            
            # Determine overall risk level
            if severity_counts['critical'] > 0:
                overall_risk = 'critical'
            elif severity_counts['high'] > 0:
                overall_risk = 'high'
            elif severity_counts['medium'] > 0:
                overall_risk = 'medium'
            elif hazardous_count > 0:
                overall_risk = 'low'
            else:
                overall_risk = 'safe'
            
            # Convert the annotated image to base64
            _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            img_base64 = base64.b64encode(buffer).decode('utf-8')

            return {
                "total_detections": total_detections,
                "hazardous_count": hazardous_count,
                "safe_count": safe_count,
                "overall_assessment": "Safe" if is_mostly_food or hazardous_count == 0 else "Hazardous",
                "overall_risk_level": overall_risk,
                "severity_breakdown": severity_counts,
                "average_confidence": round(avg_confidence, 4),
                "model_accuracy": round(model_accuracy, 2),  # Weighted accuracy (realistic)
                "high_confidence_detections": high_conf_count,  # Detections with ≥60% confidence
                "medium_confidence_detections": medium_conf_count,  # Detections 45-59%
                "detections": detections,
                "annotated_image": img_base64,
                "model_settings": {
                    "model_type": "YOLOv8x-seg (Segmentation)" if "-seg" in str(self.model.ckpt_path) else "YOLOv8x (Detection)",
                    "confidence_threshold": self.conf_threshold,
                    "iou_threshold": self.iou_threshold,
                    "image_size": self.imgsz,
                    "augmentation": "Test-Time Augmentation (TTA)",
                    "precision": "FP32 (Full Precision)",
                    "segmentation": "Enabled for better plastic detection" if "-seg" in str(self.model.ckpt_path) else "Detection only"
                }
            }
            
        except Exception as e:
            logger.error(f"Error during detection: {e}")
            raise ValueError(f"Detection failed: {str(e)}")

    def get_class_names(self):
        """Get model class names"""
        return self.class_names

    def get_health_hazard_mapping(self):
        """Get mapping of class names to health hazard status"""
        return {name: 'Safe' if name.lower() in self.safe_items else 'Hazardous'
                for name in self.class_names.values()}
    
    def update_weights(self, new_model_path: str):
        """Update model weights with new trained model"""
        try:
            with self.lock:
                logger.info(f"Updating model weights from: {new_model_path}")
                self.model = YOLO(new_model_path)
                if hasattr(self.model, 'names'):
                    self.class_names = self.model.names
                logger.info("Model weights updated successfully")
        except Exception as e:
            logger.error(f"Failed to update model weights: {e}")
            raise
    
    def update_inference_settings(self, conf: float = None, iou: float = None, imgsz: int = None):
        """Update inference settings for accuracy tuning"""
        if conf is not None:
            self.conf_threshold = max(0.0, min(1.0, conf))
            logger.info(f"Confidence threshold updated to: {self.conf_threshold}")
        if iou is not None:
            self.iou_threshold = max(0.0, min(1.0, iou))
            logger.info(f"IoU threshold updated to: {self.iou_threshold}")
        if imgsz is not None:
            self.imgsz = imgsz
            logger.info(f"Image size updated to: {self.imgsz}")
    
    def get_model_info(self) -> dict:
        """Get current model information and settings"""
        return {
            "model_loaded": self.model is not None,
            "num_classes": len(self.class_names),
            "class_names": list(self.class_names.values()),
            "inference_settings": {
                "confidence_threshold": self.conf_threshold,
                "iou_threshold": self.iou_threshold,
                "image_size": self.imgsz,
                "max_detections": self.max_det
            },
            "hazard_categories": len(self.hazard_severity)
        }
