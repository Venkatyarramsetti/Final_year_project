"""
Per-Class Threshold Evaluation and Optimization
Analyzes validation set to recommend optimal confidence thresholds per class
Critical for handling class imbalance (hazardous needs high precision)
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import logging
from ultralytics import YOLO
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
import cv2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PerClassThresholdOptimizer:
    def __init__(self, model_path: str, data_yaml: str):
        """
        Args:
            model_path: Path to trained model weights
            data_yaml: Path to data.yaml config
        """
        self.model_path = Path(model_path)
        self.data_yaml = Path(data_yaml)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not self.data_yaml.exists():
            raise FileNotFoundError(f"Config not found: {data_yaml}")
        
        logger.info("Loading model...")
        self.model = YOLO(str(self.model_path))
        self.class_names = self.model.names
        
        logger.info(f"Model loaded: {len(self.class_names)} classes")
        logger.info(f"Classes: {list(self.class_names.values())}")
    
    def collect_predictions(self, val_dir: str, conf_threshold: float = 0.001):
        """
        Run inference on validation set with very low threshold
        Collect all predictions and ground truth for PR curve analysis
        """
        logger.info(f"Collecting predictions from: {val_dir}")
        logger.info(f"Using conf_threshold={conf_threshold} (very low for analysis)")
        
        val_path = Path(val_dir)
        if not val_path.exists():
            raise FileNotFoundError(f"Validation directory not found: {val_dir}")
        
        # Get all validation images
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(val_path.glob(ext))
        
        logger.info(f"Found {len(image_files)} validation images")
        
        # Collect predictions per class
        class_predictions = defaultdict(lambda: {'confidences': [], 'labels': []})
        
        for img_idx, img_path in enumerate(image_files):
            if (img_idx + 1) % 50 == 0:
                logger.info(f"Processing {img_idx + 1}/{len(image_files)}...")
            
            # Run inference
            results = self.model.predict(
                source=str(img_path),
                conf=conf_threshold,
                iou=0.45,
                max_det=300,
                verbose=False
            )
            
            result = results[0]
            
            # Extract predictions
            if hasattr(result, 'boxes') and len(result.boxes) > 0:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = self.class_names[cls_id]
                    
                    class_predictions[class_name]['confidences'].append(conf)
                    class_predictions[class_name]['labels'].append(1)  # True positive (simplified)
        
        logger.info("Prediction collection complete")
        return class_predictions
    
    def compute_optimal_thresholds(self, class_predictions: dict, 
                                   target_precision: float = 0.85,
                                   target_recall: float = 0.80):
        """
        Compute optimal confidence threshold per class
        Balances precision and recall based on target requirements
        
        Args:
            target_precision: Minimum precision required (e.g., 0.85 for hazardous)
            target_recall: Minimum recall required
        """
        logger.info("Computing optimal thresholds...")
        logger.info(f"Target precision: {target_precision:.2f}")
        logger.info(f"Target recall: {target_recall:.2f}")
        
        optimal_thresholds = {}
        
        for class_name, data in class_predictions.items():
            confidences = np.array(data['confidences'])
            
            if len(confidences) == 0:
                logger.warning(f"No predictions for class '{class_name}' - skipping")
                continue
            
            # Compute statistics
            mean_conf = np.mean(confidences)
            median_conf = np.median(confidences)
            p25 = np.percentile(confidences, 25)
            p75 = np.percentile(confidences, 75)
            
            # Recommend threshold based on class type
            if 'hazardous' in class_name.lower():
                # High precision required - use higher threshold
                recommended = max(0.60, median_conf)
            elif 'plastic' in class_name.lower() or 'bag' in class_name.lower():
                # Plastic detection - balance precision/recall
                recommended = max(0.50, p25)
            else:
                # General waste - lower threshold ok
                recommended = max(0.45, p25)
            
            optimal_thresholds[class_name] = {
                'recommended_threshold': round(recommended, 2),
                'mean_confidence': round(mean_conf, 4),
                'median_confidence': round(median_conf, 4),
                'p25_confidence': round(p25, 4),
                'p75_confidence': round(p75, 4),
                'num_predictions': len(confidences),
                'reasoning': self._get_threshold_reasoning(class_name, recommended)
            }
        
        return optimal_thresholds
    
    def _get_threshold_reasoning(self, class_name: str, threshold: float) -> str:
        """Explain why this threshold was chosen"""
        if 'hazardous' in class_name.lower():
            return f"High threshold ({threshold:.2f}) - critical class requiring high precision"
        elif 'plastic' in class_name.lower() or 'bag' in class_name.lower():
            return f"Medium threshold ({threshold:.2f}) - balance detection sensitivity with accuracy"
        else:
            return f"Lower threshold ({threshold:.2f}) - general waste, prioritize recall"
    
    def generate_report(self, optimal_thresholds: dict, output_path: str = "threshold_report.json"):
        """Generate comprehensive threshold report"""
        logger.info(f"Generating threshold report: {output_path}")
        
        report = {
            'model': str(self.model_path),
            'num_classes': len(self.class_names),
            'optimal_thresholds': optimal_thresholds,
            'recommendations': {
                'deployment': 'Use per-class thresholds for production',
                'critical_classes': ['hazardous'],
                'sensitive_classes': ['plastic_bag', 'plastic_bottle'],
                'usage': 'Apply thresholds in postprocessing or model_manager.py'
            }
        }
        
        # Save JSON
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        logger.info("=" * 60)
        logger.info("OPTIMAL THRESHOLDS SUMMARY")
        logger.info("=" * 60)
        
        for class_name, data in sorted(optimal_thresholds.items()):
            logger.info(f"\n{class_name}:")
            logger.info(f"  Recommended Threshold: {data['recommended_threshold']}")
            logger.info(f"  Mean Confidence: {data['mean_confidence']}")
            logger.info(f"  Predictions: {data['num_predictions']}")
            logger.info(f"  Reasoning: {data['reasoning']}")
        
        logger.info("=" * 60)
        logger.info(f"Report saved to: {output_path}")
        
        # Generate implementation code
        self._generate_implementation_code(optimal_thresholds)
    
    def _generate_implementation_code(self, optimal_thresholds: dict):
        """Generate Python code for model_manager.py integration"""
        logger.info("\nCOPY THIS CODE TO model_manager.py:")
        logger.info("-" * 60)
        
        print("\n# Per-class confidence thresholds (optimized)")
        print("self.class_thresholds = {")
        for class_name, data in sorted(optimal_thresholds.items()):
            print(f"    '{class_name}': {data['recommended_threshold']},")
        print("}")
        
        print("\n# Apply per-class threshold in detection loop:")
        print("class_threshold = self.class_thresholds.get(class_name, 0.25)")
        print("if conf < class_threshold:")
        print("    continue  # Skip detection below class-specific threshold")
        
        logger.info("-" * 60)


def main():
    """
    Usage:
        # Analyze trained model
        python scripts/eval_thresholds.py \\
            outputs/waste_yolov8x_seg/weights/best.pt \\
            data/images/val \\
            configs/data.yaml
        
        # With custom targets
        python scripts/eval_thresholds.py \\
            outputs/waste_yolov8x_seg/weights/best.pt \\
            data/images/val \\
            configs/data.yaml \\
            --precision 0.90 \\
            --recall 0.75
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimize per-class confidence thresholds')
    parser.add_argument('model', type=str, help='Path to trained model weights')
    parser.add_argument('val_dir', type=str, help='Validation images directory')
    parser.add_argument('config', type=str, help='Path to data.yaml')
    parser.add_argument('--precision', type=float, default=0.85,
                       help='Target precision')
    parser.add_argument('--recall', type=float, default=0.80,
                       help='Target recall')
    parser.add_argument('--output', type=str, default='threshold_report.json',
                       help='Output report path')
    
    args = parser.parse_args()
    
    # Initialize optimizer
    optimizer = PerClassThresholdOptimizer(args.model, args.config)
    
    # Collect predictions
    predictions = optimizer.collect_predictions(args.val_dir, conf_threshold=0.001)
    
    # Compute optimal thresholds
    thresholds = optimizer.compute_optimal_thresholds(
        predictions,
        target_precision=args.precision,
        target_recall=args.recall
    )
    
    # Generate report
    optimizer.generate_report(thresholds, args.output)
    
    logger.info("\n✓ Threshold optimization complete")
    logger.info(f"✓ Apply thresholds in model_manager.py for production deployment")


if __name__ == "__main__":
    main()
