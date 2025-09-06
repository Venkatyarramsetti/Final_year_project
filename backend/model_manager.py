import os
import kagglehub
from ultralytics import YOLO
from PIL import Image as PILImage
import numpy as np

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
        self.garbage_classes_map = self.model.names
        
        # For default YOLO model, assume all detected objects are potentially hazardous
        self.class_index_to_health_hazard = {
            index: 'Hazardous' for index in self.garbage_classes_map.keys()
        }
    
    def detect_and_categorize(self, image_path):
        """Run detection on an image and return categorized results"""
        print(f"Running inference on {image_path}...")
        results = self.model(image_path)
        
        detections = []
        
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else []
            scores = r.boxes.conf.cpu().numpy() if r.boxes is not None else []
            class_indices = r.boxes.cls.cpu().numpy().astype(int) if r.boxes is not None else []
            
            for i in range(len(boxes)):
                box = boxes[i]
                score = scores[i]
                class_index = class_indices[i]
                class_name = self.model.names[class_index]
                health_hazard_category = self.class_index_to_health_hazard.get(class_index, 'Unknown')
                
                detection = {
                    "class_name": class_name,
                    "confidence": float(score),
                    "category": health_hazard_category,
                    "bounding_box": {
                        "x1": float(box[0]),
                        "y1": float(box[1]),
                        "x2": float(box[2]),
                        "y2": float(box[3])
                    }
                }
                detections.append(detection)
        
        # Summary statistics
        total_detections = len(detections)
        hazardous_count = sum(1 for d in detections if d["category"] == "Hazardous")
        healthy_count = total_detections - hazardous_count
        
        result = {
            "total_detections": total_detections,
            "hazardous_count": hazardous_count,
            "healthy_count": healthy_count,
            "overall_assessment": "Hazardous" if hazardous_count > 0 else "Healthy",
            "detections": detections
        }
        
        print("Inference and categorization complete.")
        return result
    
    def get_class_names(self):
        """Get all class names"""
        return self.garbage_classes_map
    
    def get_health_hazard_mapping(self):
        """Get the health/hazard mapping"""
        return self.class_index_to_health_hazard
