"""
Context-Aware Postprocessing for Waste Detection
Implements spatial reasoning: objects inside bin polygon = waste
Applies per-class confidence thresholds for production deployment
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from shapely.geometry import Point, Polygon, box as Box
from shapely.ops import unary_union
from typing import List, Dict, Tuple
import logging
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ContextAwareDetector:
    def __init__(self, model_path: str, threshold_config: str = None):
        """
        Args:
            model_path: Path to trained YOLOv8 model
            threshold_config: Path to threshold_report.json (optional)
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        logger.info(f"Loading model: {model_path}")
        self.model = YOLO(str(self.model_path))
        self.class_names = self.model.names
        
        # Load per-class thresholds
        self.class_thresholds = self._load_thresholds(threshold_config)
        
        logger.info(f"Model loaded with {len(self.class_names)} classes")
    
    def _load_thresholds(self, threshold_config: str) -> Dict[str, float]:
        """Load per-class confidence thresholds from JSON"""
        if threshold_config and Path(threshold_config).exists():
            with open(threshold_config, 'r') as f:
                data = json.load(f)
                thresholds = {
                    name: info['recommended_threshold']
                    for name, info in data.get('optimal_thresholds', {}).items()
                }
            logger.info(f"Loaded {len(thresholds)} per-class thresholds")
            return thresholds
        else:
            # Default thresholds
            logger.warning("No threshold config found - using defaults")
            return {
                'hazardous': 0.60,
                'plastic_bag': 0.50,
                'plastic_bottle': 0.50,
                'glass_bottle': 0.45,
                'metal_can': 0.45,
                'food_waste': 0.45,
                'paper': 0.40,
                'cardboard': 0.40,
                'container': 0.45,
                'wrapper': 0.45,
                'general_trash': 0.40,
                'bin': 0.50
            }
    
    def detect_bins(self, results) -> List[Polygon]:
        """Extract bin/container polygons from detections"""
        bin_polygons = []
        
        for result in results:
            if not hasattr(result, 'boxes') or len(result.boxes) == 0:
                continue
            
            for idx, box in enumerate(result.boxes):
                cls_id = int(box.cls[0])
                class_name = self.class_names[cls_id].lower()
                
                # Check if this is a bin/container
                if 'bin' in class_name or 'trash' in class_name or 'container' in class_name:
                    coords = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = coords
                    
                    # Create polygon from bounding box
                    bin_poly = Box(x1, y1, x2, y2)
                    bin_polygons.append(bin_poly)
                    
                    logger.debug(f"Detected bin: {class_name} at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
        
        # Merge overlapping bins
        if bin_polygons:
            merged_bin = unary_union(bin_polygons)
            if merged_bin.geom_type == 'Polygon':
                return [merged_bin]
            elif merged_bin.geom_type == 'MultiPolygon':
                return list(merged_bin.geoms)
        
        return []
    
    def mask_to_polygon(self, mask: np.ndarray) -> Polygon:
        """Convert segmentation mask to shapely Polygon"""
        # Find contours
        mask_uint8 = (mask * 255).astype('uint8')
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(largest_contour) < 50:  # Too small
            return None
        
        # Convert to polygon
        points = [(int(p[0][0]), int(p[0][1])) for p in largest_contour]
        if len(points) < 3:
            return None
        
        try:
            poly = Polygon(points)
            return poly if poly.is_valid else poly.buffer(0)
        except:
            return None
    
    def compute_overlap_fraction(self, obj_polygon: Polygon, bin_polygons: List[Polygon]) -> float:
        """
        Compute fraction of object inside bin(s)
        Returns: 0.0 to 1.0 (0 = completely outside, 1 = completely inside)
        """
        if not bin_polygons or obj_polygon is None:
            return 0.0
        
        # Union all bin polygons
        bin_union = unary_union(bin_polygons)
        
        # Compute intersection
        try:
            intersection = obj_polygon.intersection(bin_union)
            overlap_fraction = intersection.area / obj_polygon.area if obj_polygon.area > 0 else 0.0
            return overlap_fraction
        except:
            return 0.0
    
    def postprocess_detections(self, 
                               image_path: str, 
                               in_bin_threshold: float = 0.5,
                               visualize: bool = True) -> Dict:
        """
        Run detection with context-aware postprocessing
        
        Args:
            image_path: Path to input image
            in_bin_threshold: Fraction of object inside bin to count as waste (0.5 = 50%)
            visualize: Draw results on image
            
        Returns:
            Dictionary with detections, waste classification, and metadata
        """
        logger.info(f"Processing: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Run inference
        results = self.model.predict(
            source=image_path,
            conf=0.001,  # Very low threshold - we'll apply per-class later
            iou=0.45,
            max_det=300,
            verbose=False
        )
        
        # Detect bins first
        bin_polygons = self.detect_bins(results)
        logger.info(f"Detected {len(bin_polygons)} bin(s)")
        
        # Process all detections
        detections = []
        waste_count = 0
        safe_count = 0
        
        for result in results:
            if not hasattr(result, 'boxes') or len(result.boxes) == 0:
                continue
            
            for idx, box in enumerate(result.boxes):
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = self.class_names[cls_id]
                coords = box.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, coords)
                
                # Apply per-class threshold
                class_threshold = self.class_thresholds.get(class_name, 0.25)
                if conf < class_threshold:
                    continue  # Skip low confidence for this class
                
                # Skip bin class in waste counting
                if class_name.lower() == 'bin':
                    continue
                
                # Get object polygon (from segmentation mask if available)
                obj_polygon = None
                if hasattr(result, 'masks') and result.masks is not None and idx < len(result.masks.data):
                    mask = result.masks.data[idx].cpu().numpy()
                    obj_polygon = self.mask_to_polygon(mask)
                
                # Fallback to bbox polygon
                if obj_polygon is None:
                    obj_polygon = Box(x1, y1, x2, y2)
                
                # Compute overlap with bins
                overlap_fraction = self.compute_overlap_fraction(obj_polygon, bin_polygons)
                
                # Context-aware classification
                in_bin = overlap_fraction >= in_bin_threshold
                
                # Determine if waste based on context
                is_waste = False
                reasoning = ""
                
                if in_bin:
                    # Object inside bin → likely waste
                    is_waste = True
                    reasoning = f"Inside bin ({overlap_fraction*100:.0f}% overlap)"
                    waste_count += 1
                elif 'plastic' in class_name.lower() or 'trash' in class_name.lower() or 'waste' in class_name.lower():
                    # Inherently waste items
                    is_waste = True
                    reasoning = "Waste item detected"
                    waste_count += 1
                else:
                    # Safe item (food, furniture, etc.)
                    is_waste = False
                    reasoning = "Safe item (not in bin)"
                    safe_count += 1
                
                # Store detection
                detections.append({
                    'class': class_name,
                    'confidence': float(conf),
                    'bbox': [float(c) for c in coords],
                    'is_waste': is_waste,
                    'in_bin': in_bin,
                    'overlap_fraction': float(overlap_fraction),
                    'reasoning': reasoning,
                    'class_threshold': class_threshold
                })
                
                # Visualize
                if visualize:
                    color = (0, 0, 255) if is_waste else (0, 255, 0)  # Red=waste, Green=safe
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    
                    label = f"{class_name} {conf:.2f}"
                    if in_bin:
                        label += f" [WASTE {overlap_fraction*100:.0f}%]"
                    
                    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(image, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
                    cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw bin outlines
        if visualize and bin_polygons:
            for bin_poly in bin_polygons:
                bounds = bin_poly.bounds  # (minx, miny, maxx, maxy)
                x1, y1, x2, y2 = map(int, bounds)
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 3)  # Yellow for bins
                cv2.putText(image, "BIN", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Compile results
        result_dict = {
            'image_path': image_path,
            'total_detections': len(detections),
            'waste_count': waste_count,
            'safe_count': safe_count,
            'bins_detected': len(bin_polygons),
            'overall_assessment': 'Hazardous' if waste_count > 0 else 'Safe',
            'detections': detections,
            'annotated_image': image if visualize else None
        }
        
        return result_dict
    
    def save_visualization(self, result_dict: Dict, output_path: str):
        """Save annotated image"""
        if result_dict['annotated_image'] is not None:
            cv2.imwrite(output_path, result_dict['annotated_image'])
            logger.info(f"Saved visualization: {output_path}")


def main():
    """
    Usage:
        python scripts/postprocess_detector.py \\
            outputs/waste_yolov8x_seg/weights/best.pt \\
            test_images/trash_bin.jpg \\
            --threshold-config threshold_report.json \\
            --output results/output.jpg
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Context-aware waste detection')
    parser.add_argument('model', type=str, help='Path to trained model')
    parser.add_argument('image', type=str, help='Input image path')
    parser.add_argument('--threshold-config', type=str, default=None,
                       help='Path to threshold_report.json')
    parser.add_argument('--in-bin-threshold', type=float, default=0.5,
                       help='Overlap fraction to count as in-bin (0.0-1.0)')
    parser.add_argument('--output', type=str, default='result.jpg',
                       help='Output visualization path')
    parser.add_argument('--json', type=str, default=None,
                       help='Save results JSON to path')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = ContextAwareDetector(args.model, args.threshold_config)
    
    # Run detection
    results = detector.postprocess_detections(
        args.image,
        in_bin_threshold=args.in_bin_threshold,
        visualize=True
    )
    
    # Save visualization
    detector.save_visualization(results, args.output)
    
    # Save JSON
    if args.json:
        results_json = results.copy()
        results_json.pop('annotated_image')  # Remove image for JSON
        with open(args.json, 'w') as f:
            json.dump(results_json, f, indent=2)
        logger.info(f"Saved results: {args.json}")
    
    # Print summary
    logger.info("=" * 60)
    logger.info("DETECTION SUMMARY")
    logger.info(f"Total detections: {results['total_detections']}")
    logger.info(f"Waste items: {results['waste_count']}")
    logger.info(f"Safe items: {results['safe_count']}")
    logger.info(f"Bins detected: {results['bins_detected']}")
    logger.info(f"Assessment: {results['overall_assessment']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
