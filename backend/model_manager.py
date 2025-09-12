import os
import base64
import cv2
import logging
from ultralytics import YOLO

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional: Only import Kaggle API if credentials are available
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False
    logger.warning("Kaggle API not available. Model training via Kaggle will be disabled.")


class GarbageDetectionModel:
    def __init__(self):
        self.model = None
        self.class_names = {}
        self.safe_items = [
            'banana', 'apple', 'sandwich', 'orange', 'carrot', 'broccoli', 
            'hot dog', 'pizza', 'donut', 'cake', 'bowl', 'cucumber', 
            'tomato', 'lettuce', 'celery', 'potato', 'onion', 'bell pepper',
            'fruits', 'vegetables'
        ]
        logger.info("Initializing Garbage Detection Model...")
        self.setup_model()
        logger.info("Model initialized successfully!")

    def setup_model(self):
        """Initialize and setup the YOLO model"""
        model_path = "models/garbage_detection_model.pt"
        fallback_model_path = "yolov8n.pt"

        try:
            logger.info("Setting up Garbage Detection Model...")
            if os.path.exists(model_path):
                logger.info(f"Loading existing trained model from {model_path}...")
                self.model = YOLO(model_path)
            elif os.path.exists(fallback_model_path):
                logger.info(f"Loading pre-trained YOLOv8 model from {fallback_model_path}...")
                self.model = YOLO(fallback_model_path)
            elif KAGGLE_AVAILABLE and os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
                logger.info("Kaggle API credentials found. Attempting to download dataset...")
                self._download_dataset_and_train()
            else:
                logger.info("No existing model found. Downloading pre-trained YOLOv8 model...")
                self.model = YOLO('yolov8n.pt')

            if hasattr(self.model, 'names'):
                self.class_names = self.model.names
                logger.info(f"Model loaded with {len(self.class_names)} classes")
            else:
                logger.warning("Model doesn't have class names attribute")
                self.class_names = {}
        except Exception as e:
            logger.error(f"Error setting up model: {e}")
            logger.info("Falling back to pre-trained YOLOv8 model...")
            try:
                self.model = YOLO('yolov8n.pt')
                self.class_names = self.model.names if hasattr(self.model, 'names') else {}
                logger.info("Fallback model loaded successfully")
            except Exception as fallback_error:
                logger.error(f"Fallback model loading failed: {fallback_error}")
                logger.critical("Unable to load any model. API will not function correctly.")

    def _download_dataset_and_train(self):
        """Download Taco dataset via Kaggle API and train YOLO"""
        try:
            dataset_path = "datasets/taco_dataset"
            os.makedirs(dataset_path, exist_ok=True)
            
            if not KAGGLE_AVAILABLE:
                logger.error("Kaggle API not available for dataset download")
                raise ImportError("Kaggle API not available")
                
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            
            logger.info("Downloading TACO dataset from Kaggle...")
            api.dataset_download_files("vencerlanz09/taco-dataset-yolo-format", path=dataset_path, unzip=True)
            logger.info(f"Dataset downloaded to: {dataset_path}")

            logger.info("Initializing YOLOv8 model for training...")
            self.model = YOLO('yolov8n.pt')
            data_yaml_path = os.path.join(dataset_path, 'data.yaml')
            
            logger.info("Training model for 10 epochs...")
            self.model.train(data=data_yaml_path, epochs=10)
            
            os.makedirs("models", exist_ok=True)
            self.model.save("models/garbage_detection_model.pt")
            logger.info("Model training and saving complete.")
            
            if hasattr(self.model, 'names'):
                self.class_names = self.model.names
            else:
                logger.warning("Trained model doesn't have class names attribute")
                self.class_names = {}
                
        except Exception as e:
            logger.error(f"Error during dataset download or training: {e}")
            logger.info("Falling back to pre-trained YOLOv8 model...")
            self.model = YOLO('yolov8n.pt')
            self.class_names = self.model.names if hasattr(self.model, 'names') else {}

    def detect_and_categorize(self, image_path):
        """Run detection on an image and return categorized results with annotated image"""
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
            # Run inference with the model
            results = self.model(image_path)
            detections = []
            food_items_count = 0
            total_items = 0

            # Count food items
            for result in results:
                if not hasattr(result, 'boxes') or len(result.boxes) == 0:
                    continue
                    
                for box in result.boxes:
                    if not hasattr(box, 'cls') or len(box.cls) == 0:
                        continue
                        
                    cls_id = int(box.cls[0])
                    cls_name = self.class_names.get(cls_id, '').lower()
                    if cls_name in self.safe_items:
                        food_items_count += 1
                    total_items += 1

            is_mostly_food = (food_items_count / max(total_items, 1)) > 0.5

            # Process detections and draw boxes
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
                    is_food_item = class_name.lower() in self.safe_items
                    hazard_status = 'Safe' if (is_food_item or is_mostly_food) else 'Hazardous'
                    color = (0, 255, 0) if hazard_status == 'Safe' else (0, 0, 255)

                    # Draw detection rectangle and label
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    label = f"{class_name} ({conf:.2f})"
                    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(image, (x1, y1 - label_height - 10), (x1 + label_width, y1), color, -1)
                    cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    detections.append({
                        'class': class_name,
                        'confidence': float(conf),
                        'coordinates': [float(c) for c in coords],
                        'health_status': hazard_status
                    })

            # Prepare response
            total_detections = len(detections)
            hazardous_count = sum(1 for d in detections if d["health_status"] == "Hazardous")
            safe_count = total_detections - hazardous_count
            
            # Convert the annotated image to base64
            _, buffer = cv2.imencode('.jpg', image)
            img_base64 = base64.b64encode(buffer).decode('utf-8')

            return {
                "total_detections": total_detections,
                "hazardous_count": hazardous_count,
                "safe_count": safe_count,
                "overall_assessment": "Safe" if hazardous_count == 0 else "Hazardous",
                "detections": detections,
                "annotated_image": img_base64
            }
            
        except Exception as e:
            logger.error(f"Error during detection: {e}")
            raise ValueError(f"Detection failed: {str(e)}")

    def get_class_names(self):
        return self.class_names

    def get_health_hazard_mapping(self):
        return {name: 'Safe' if name.lower() in self.safe_items else 'Hazardous'
                for name in self.class_names.values()}
