"""
YOLOv8 Segmentation Training Script for Waste Detection
Supports transfer learning, checkpoint resuming, and automatic hyperparameter tuning
"""

import os
import sys
from pathlib import Path
import logging
from ultralytics import YOLO
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WasteDetectionTrainer:
    def __init__(self, config_path: str, hyp_path: str):
        """
        Args:
            config_path: Path to data.yaml
            hyp_path: Path to hyp.yaml (hyperparameters)
        """
        self.config_path = Path(config_path)
        self.hyp_path = Path(hyp_path)
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        if not self.hyp_path.exists():
            raise FileNotFoundError(f"Hyperparameters not found: {hyp_path}")
        
        logger.info("=" * 60)
        logger.info("WASTE DETECTION MODEL TRAINING")
        logger.info(f"Config: {config_path}")
        logger.info(f"Hyperparameters: {hyp_path}")
        logger.info("=" * 60)
    
    def check_gpu(self):
        """Check GPU availability"""
        if torch.cuda.is_available():
            logger.info(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
            logger.info(f"✓ CUDA version: {torch.version.cuda}")
            logger.info(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            return True
        else:
            logger.warning("⚠ No GPU detected - training will be VERY slow on CPU")
            logger.warning("⚠ Consider using Google Colab (free GPU) or cloud GPU")
            return False
    
    def train(self, 
              model_path: str = 'yolov8x-seg.pt',
              epochs: int = 150,
              batch_size: int = 16,
              imgsz: int = 640,
              device: int = 0,
              resume: bool = False,
              pretrained: bool = True,
              cache: bool = False,
              workers: int = 8,
              name: str = 'waste_yolov8x_seg',
              project: str = 'outputs'):
        """
        Train YOLOv8 segmentation model
        
        Args:
            model_path: Starting model ('yolov8x-seg.pt' for COCO pretrained)
            epochs: Total training epochs
            batch_size: Batch size (-1 for AutoBatch)
            imgsz: Input image size
            device: GPU device (0, 1, 2, ...) or 'cpu'
            resume: Resume from last checkpoint
            pretrained: Use pretrained weights
            cache: Cache images to RAM (faster but needs memory)
            workers: Number of dataloader workers
            name: Experiment name
            project: Project directory
        """
        logger.info("Initializing training...")
        
        # Check GPU
        has_gpu = self.check_gpu()
        if not has_gpu and device != 'cpu':
            logger.warning("Forcing device=cpu")
            device = 'cpu'
        
        # Load model
        logger.info(f"Loading model: {model_path}")
        model = YOLO(model_path)
        
        # Show model info
        logger.info(f"Model architecture: {model_path}")
        logger.info(f"Pretrained: {pretrained}")
        
        # Training parameters
        train_params = {
            'data': str(self.config_path.absolute()),
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch_size,
            'device': device,
            'save': True,
            'save_period': -1,  # Save only best and last
            'cache': cache,
            'workers': workers,
            'pretrained': pretrained,
            'optimizer': 'AdamW',  # Better for fine-tuning
            'verbose': True,
            'seed': 42,  # Reproducibility
            'deterministic': True,
            'val': True,
            'plots': True,
            'name': name,
            'project': project,
            'exist_ok': False,
            'resume': resume,
            'amp': True,  # Automatic Mixed Precision for faster training
            'fraction': 1.0,  # Use full dataset
            'profile': False,
            'freeze': None,  # Layers to freeze (None = train all)
            'multi_scale': False,  # Multi-scale training (slower but better)
            'overlap_mask': True,  # Segmentation mask overlap
            'mask_ratio': 4,  # Segmentation mask downsample ratio
            'dropout': 0.0,  # Dropout (0 = off)
            'cos_lr': False,  # Cosine LR scheduler
            'close_mosaic': 10,  # Disable mosaic last N epochs
            'resume': resume,
            'label_smoothing': 0.05,  # From hyp.yaml
            'patience': 50,  # Early stopping patience
            'cfg': str(self.hyp_path.absolute())  # Hyperparameters
        }
        
        logger.info("Training parameters:")
        for key, value in train_params.items():
            logger.info(f"  {key}: {value}")
        
        logger.info("=" * 60)
        logger.info("STARTING TRAINING...")
        logger.info("=" * 60)
        
        # Train
        try:
            results = model.train(**train_params)
            
            logger.info("=" * 60)
            logger.info("TRAINING COMPLETE!")
            logger.info("=" * 60)
            logger.info(f"Results saved to: {project}/{name}")
            logger.info(f"Best weights: {project}/{name}/weights/best.pt")
            logger.info(f"Last weights: {project}/{name}/weights/last.pt")
            
            # Print final metrics
            if hasattr(results, 'results_dict'):
                metrics = results.results_dict
                logger.info("\nFinal Metrics:")
                logger.info(f"  mAP@0.5: {metrics.get('metrics/mAP50(B)', 0):.4f}")
                logger.info(f"  mAP@0.5:0.95: {metrics.get('metrics/mAP50-95(B)', 0):.4f}")
                logger.info(f"  Precision: {metrics.get('metrics/precision(B)', 0):.4f}")
                logger.info(f"  Recall: {metrics.get('metrics/recall(B)', 0):.4f}")
            
            return results
            
        except KeyboardInterrupt:
            logger.warning("\nTraining interrupted by user")
            logger.info("To resume: set resume=True")
            sys.exit(0)
        
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def validate(self, weights_path: str, imgsz: int = 640, device: int = 0):
        """Run validation on trained model"""
        logger.info(f"Running validation on: {weights_path}")
        
        model = YOLO(weights_path)
        results = model.val(
            data=str(self.config_path.absolute()),
            imgsz=imgsz,
            device=device,
            save_json=True,
            save_hybrid=False,
            conf=0.001,  # Low threshold for PR curve
            iou=0.6,
            max_det=300,
            half=False,
            plots=True
        )
        
        logger.info("Validation complete")
        return results
    
    def export(self, weights_path: str, format: str = 'onnx'):
        """Export model to deployment format"""
        logger.info(f"Exporting model to {format}...")
        
        model = YOLO(weights_path)
        model.export(format=format, half=False, int8=False, dynamic=False)
        
        logger.info(f"✓ Exported to {format}")


def main():
    """
    Usage:
        # Train from scratch
        python scripts/train_yolo.py
        
        # Train with custom settings
        python scripts/train_yolo.py --epochs 200 --batch 8 --imgsz 1280
        
        # Resume training
        python scripts/train_yolo.py --resume
        
        # Validate only
        python scripts/train_yolo.py --validate outputs/waste_yolov8x_seg/weights/best.pt
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Train YOLOv8 waste detection model')
    parser.add_argument('--config', type=str, 
                       default='configs/data.yaml',
                       help='Path to data.yaml')
    parser.add_argument('--hyp', type=str,
                       default='configs/hyp.yaml',
                       help='Path to hyp.yaml')
    parser.add_argument('--model', type=str,
                       default='yolov8x-seg.pt',
                       help='Starting model weights')
    parser.add_argument('--epochs', type=int, default=150,
                       help='Number of epochs')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size (-1 for AutoBatch)')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Input image size')
    parser.add_argument('--device', type=str, default='0',
                       help='Device (0, 1, 2, ... or cpu)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from last checkpoint')
    parser.add_argument('--cache', action='store_true',
                       help='Cache images to RAM')
    parser.add_argument('--workers', type=int, default=8,
                       help='Dataloader workers')
    parser.add_argument('--name', type=str,
                       default='waste_yolov8x_seg',
                       help='Experiment name')
    parser.add_argument('--project', type=str, default='outputs',
                       help='Project directory')
    parser.add_argument('--validate', type=str, default=None,
                       help='Validate weights path (skip training)')
    parser.add_argument('--export', type=str, default=None,
                       help='Export weights path to ONNX')
    
    args = parser.parse_args()
    
    # Convert device
    device = args.device if args.device == 'cpu' else int(args.device)
    
    trainer = WasteDetectionTrainer(args.config, args.hyp)
    
    if args.validate:
        trainer.validate(args.validate, args.imgsz, device)
    elif args.export:
        trainer.export(args.export, format='onnx')
    else:
        trainer.train(
            model_path=args.model,
            epochs=args.epochs,
            batch_size=args.batch,
            imgsz=args.imgsz,
            device=device,
            resume=args.resume,
            cache=args.cache,
            workers=args.workers,
            name=args.name,
            project=args.project
        )


if __name__ == "__main__":
    main()
