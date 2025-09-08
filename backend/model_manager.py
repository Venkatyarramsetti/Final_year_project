import os
import kagglehub
import base64
import cv2
from io import BytesIO
from ultralytics import YOLO
from PIL import Image as PILImage
import numpy as np
import cv2
import base64
from io import BytesIO

class GarbageDetectionModel:
    def __init__(self):
        self.model = None
        self.garbage_classes_map = {}
        self.class_index_to_health_hazard = {}
        self.setup_model()
    
    def setup_model(self):
        """Initialize and setup the YOLO model"""
        try:
            print("Setting up Garbage Detection Model...")
            
            # Check if model already exists
            model_path = "models/garbage_detection_model.pt"
            
            if os.path.exists(model_path):
                print("Loading existing trained model...")
                self.model = YOLO(model_path)
            else:
                print("No existing model found. Starting fresh setup...")
                self._download_dataset_and_train()
            
            # Setup class mappings
            self._setup_class_mappings()
            print("Model setup complete!")
            
        except Exception as e:
            print(f"Error setting up model: {e}")
            # Fallback to pre-trained model
            print("Falling back to pre-trained YOLOv8 model...")
            self.model = YOLO('yolov8n.pt')
            self._setup_default_mappings()
    
    def _download_dataset_and_train(self):
        """Download dataset and train the model"""
        print("Downloading garbage object detection dataset...")
        dataset_path = kagglehub.dataset_download("vencerlanz09/taco-dataset-yolo-format")
        print(f"Dataset downloaded to: {dataset_path}")
        
        # Load a pre-trained YOLOv8 model
        print("Loading YOLOv8n model...")
        self.model = YOLO('yolov8n.pt')
        
        # Define the path to the data.yaml file
        data_yaml_path = os.path.join(dataset_path, 'data.yaml')
        print(f"Using data.yaml from: {data_yaml_path}")
        
        # Train the model (reduced epochs for faster setup)
        print("Starting model training (10 epochs for quick setup)...")
        results = self.model.train(data=data_yaml_path, epochs=10)
        print("Model training complete.")
        
        # Save the trained model
        os.makedirs("models", exist_ok=True)
        self.model.save("models/garbage_detection_model.pt")
        
        # Evaluate the model
        print("Evaluating the trained model...")
        metrics = self.model.val()
        print("Model evaluation complete.")
    
    def _setup_class_mappings(self):
        """Setup class name mappings and health/hazard categorization"""
        self.garbage_classes_map = self.model.names
        
        # Define hazardous garbage types
        hazardous_items = [
            'Aluminium foil', 'Bottle cap', 'Broken glass', 'Cigarette', 
            'Cup', 'Lid', 'Other litter', 'Other plastic', 
            'Plastic bag - wrapper', 'Pop tab', 'Straw', 
            'Styrofoam piece', 'Unlabeled litter'
        ]
        
        self.class_index_to_health_hazard = {
            index: 'Hazardous' if name in hazardous_items else 'Healthy'
            for index, name in self.garbage_classes_map.items()
        }
    
    def _setup_default_mappings(self):
        """Setup default mappings for fallback model"""
        if self.model is None:
            raise ValueError("Model is not initialized")
            
        # Store names from model
        self.class_names = self.model.names if hasattr(self.model, 'names') else {}
        
        # Define safe food items from COCO dataset
        self.safe_items = [
            'banana', 'apple', 'sandwich', 'orange', 'carrot', 'broccoli', 
            'hot dog', 'pizza', 'donut', 'cake', 'bowl', 'cucumber', 
            'tomato', 'lettuce', 'celery', 'potato', 'onion', 'bell pepper',
            'fruits', 'vegetables'
        ]
    
    def detect_and_categorize(self, image_path):
        """Run detection on an image and return categorized results with annotated image"""
        if self.model is None:
            raise ValueError("Model is not initialized")
            
        print(f"Running inference on {image_path}...")
        # Read image with OpenCV for drawing
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Could not read image")
            
        results = self.model(image_path)
        
        detections = []
        food_items_count = 0
        total_items = 0
        
        # Process all detections first to count food items
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = self.class_names[cls_id].lower()
                
                if cls_name in self.safe_items:
                    food_items_count += 1
                total_items += 1
        
        # Determine if image mostly contains food
        is_mostly_food = (food_items_count / max(total_items, 1)) > 0.5
        
        # Process detections for output and draw on image
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = self.class_names[cls_id]
                coords = box.xyxy[0].tolist()
                
                # Convert coordinates to integers for drawing
                x1, y1, x2, y2 = map(int, coords)
                
                # Mark as safe if it's a food item or if the image mostly contains food
                is_food_item = class_name.lower() in self.safe_items
                hazard_status = 'Safe' if (is_food_item or is_mostly_food) else 'Hazardous'
                
                # Choose color based on hazard status (green for safe, red for hazardous)
                color = (0, 255, 0) if hazard_status == 'Safe' else (0, 0, 255)
                
                # Draw bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                
                # Add label with class name and confidence
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
                
        # Summary statistics
        total_detections = len(detections)
        hazardous_count = sum(1 for d in detections if d["health_status"] == "Hazardous")
        safe_count = total_detections - hazardous_count
        
        # Convert the annotated image to base64
        _, buffer = cv2.imencode('.jpg', image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        result = {
            "total_detections": total_detections,
            "hazardous_count": hazardous_count,
            "safe_count": safe_count,
            "overall_assessment": "Safe" if hazardous_count == 0 else "Hazardous",
            "detections": detections,
            "annotated_image": img_base64
        }
        
        print("Inference and categorization complete.")
        return result
    
    def get_class_names(self):
        """Get all class names"""
        return self.class_names
    
    def get_health_hazard_mapping(self):
        """Get the health/hazard mapping for each class"""
        return {name: 'Safe' if name.lower() in self.safe_items else 'Hazardous' 
                for name in self.class_names.values()}
