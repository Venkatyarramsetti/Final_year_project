"""
Generate synthetic waste images by compositing segmented objects onto backgrounds
CRITICAL for class imbalance (hazardous, plastic_bag) and transparent object detection
"""

import os
import random
import json
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from pathlib import Path
import logging
from typing import List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SyntheticWasteGenerator:
    def __init__(self, object_dir: str, background_dir: str, output_dir: str):
        """
        Args:
            object_dir: Directory with segmented waste objects (PNGs with alpha channel)
            background_dir: Directory with bin/trash background images
            output_dir: Directory to save synthetic composites
        """
        self.object_dir = Path(object_dir)
        self.background_dir = Path(background_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load object and background lists
        self.objects = self._load_objects()
        self.backgrounds = self._load_backgrounds()
        
        logger.info(f"Loaded {len(self.objects)} objects, {len(self.backgrounds)} backgrounds")
    
    def _load_objects(self) -> List[dict]:
        """Load segmented object PNGs with metadata"""
        objects = []
        for obj_path in self.object_dir.glob("*.png"):
            # Expect format: plastic_bag_001.png, bottle_045.png
            name = obj_path.stem
            parts = name.split('_')
            category = '_'.join(parts[:-1]) if len(parts) > 1 else 'unknown'
            
            objects.append({
                'path': obj_path,
                'name': name,
                'category': category
            })
        return objects
    
    def _load_backgrounds(self) -> List[Path]:
        """Load background images"""
        backgrounds = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            backgrounds.extend(self.background_dir.glob(ext))
        return backgrounds
    
    def augment_object(self, obj_img: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """Apply augmentations to object for variation"""
        # Random rotation
        angle = random.uniform(-30, 30)
        obj_img = obj_img.rotate(angle, expand=True, fillcolor=(0, 0, 0, 0))
        mask = mask.rotate(angle, expand=True, fillcolor=0)
        
        # Random brightness/contrast (simulate lighting)
        if random.random() > 0.5:
            enhancer = ImageEnhance.Brightness(obj_img)
            obj_img = enhancer.enhance(random.uniform(0.7, 1.3))
        
        if random.random() > 0.5:
            enhancer = ImageEnhance.Contrast(obj_img)
            obj_img = enhancer.enhance(random.uniform(0.8, 1.2))
        
        # Slight blur (simulate depth/motion)
        if random.random() > 0.7:
            obj_img = obj_img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        
        return obj_img, mask
    
    def paste_object(self, obj_data: dict, bg_path: Path, max_scale: float = 0.6) -> Tuple[Image.Image, dict]:
        """
        Paste object onto background with augmentation
        Returns: (composite_image, annotation_dict)
        """
        # Load object and background
        obj_img = Image.open(obj_data['path']).convert("RGBA")
        bg_img = Image.open(bg_path).convert("RGB")
        
        # Extract mask from alpha channel
        mask = obj_img.split()[3]  # Alpha channel
        
        # Augment object
        obj_img, mask = self.augment_object(obj_img, mask)
        
        # Random scale
        scale = random.uniform(0.15, max_scale)
        new_w = int(obj_img.width * scale)
        new_h = int(obj_img.height * scale)
        obj_img = obj_img.resize((new_w, new_h), Image.LANCZOS)
        mask = mask.resize((new_w, new_h), Image.LANCZOS)
        
        # Random position (bias towards lower half - bins floor)
        max_x = max(0, bg_img.width - new_w)
        max_y = max(0, bg_img.height - new_h)
        
        # Bias Y position to bottom 60% (trash settles at bottom of bins)
        y_start = int(bg_img.height * 0.4)
        x = random.randint(0, max_x) if max_x > 0 else 0
        y = random.randint(y_start, max_y) if max_y > y_start else y_start
        
        # Paste object onto background
        bg_img.paste(obj_img, (x, y), mask)
        
        # Create annotation (COCO bbox format: [x, y, width, height])
        bbox = [x, y, new_w, new_h]
        
        # Calculate segmentation polygon from mask
        mask_np = np.array(mask)
        contours, _ = cv2.findContours((mask_np > 128).astype(np.uint8), 
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        segmentation = []
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            # Offset contour by paste position
            polygon = largest_contour.reshape(-1, 2) + np.array([x, y])
            segmentation = [polygon.flatten().tolist()]
        
        annotation = {
            'category': obj_data['category'],
            'bbox': bbox,
            'segmentation': segmentation,
            'area': new_w * new_h
        }
        
        return bg_img, annotation
    
    def generate_composite(self, num_objects: int = 3, output_name: str = "synth_001.jpg") -> dict:
        """
        Generate single composite image with multiple objects
        Returns: annotation dict with image info
        """
        # Random background
        bg_path = random.choice(self.backgrounds)
        bg_img = Image.open(bg_path).convert("RGB")
        
        annotations = []
        
        # Paste multiple objects
        for _ in range(num_objects):
            obj_data = random.choice(self.objects)
            bg_img, ann = self.paste_object(obj_data, bg_path)
            annotations.append(ann)
        
        # Save composite
        output_path = self.output_dir / output_name
        bg_img.save(output_path, quality=95)
        
        return {
            'file_name': output_name,
            'width': bg_img.width,
            'height': bg_img.height,
            'annotations': annotations
        }
    
    def generate_batch(self, count: int = 1000, objects_per_image: int = 3):
        """Generate batch of synthetic images with COCO annotations"""
        logger.info(f"Generating {count} synthetic composites...")
        
        coco_data = {
            "images": [],
            "annotations": [],
            "categories": self._get_categories()
        }
        
        img_id = 1
        ann_id = 1
        
        for i in range(count):
            output_name = f"synth_{i+1:05d}.jpg"
            
            try:
                composite_data = self.generate_composite(
                    num_objects=random.randint(1, objects_per_image + 2),
                    output_name=output_name
                )
                
                # Add image to COCO
                coco_data["images"].append({
                    "id": img_id,
                    "file_name": composite_data['file_name'],
                    "width": composite_data['width'],
                    "height": composite_data['height']
                })
                
                # Add annotations to COCO
                for ann in composite_data['annotations']:
                    cat_id = self._category_name_to_id(ann['category'])
                    coco_data["annotations"].append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": cat_id,
                        "bbox": ann['bbox'],
                        "segmentation": ann['segmentation'],
                        "area": ann['area'],
                        "iscrowd": 0
                    })
                    ann_id += 1
                
                img_id += 1
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Generated {i + 1}/{count} images...")
                
            except Exception as e:
                logger.warning(f"Failed to generate {output_name}: {e}")
                continue
        
        # Save COCO annotations
        ann_path = self.output_dir / "annotations.json"
        with open(ann_path, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        logger.info(f"✓ Generated {len(coco_data['images'])} synthetic images")
        logger.info(f"✓ Annotations saved to: {ann_path}")
    
    def _get_categories(self) -> List[dict]:
        """Get category list for COCO"""
        categories = [
            {"id": 1, "name": "plastic_bag"},
            {"id": 2, "name": "plastic_bottle"},
            {"id": 3, "name": "glass_bottle"},
            {"id": 4, "name": "metal_can"},
            {"id": 5, "name": "food_waste"},
            {"id": 6, "name": "paper"},
            {"id": 7, "name": "cardboard"},
            {"id": 8, "name": "container"},
            {"id": 9, "name": "wrapper"},
            {"id": 10, "name": "hazardous"},
            {"id": 11, "name": "general_trash"}
        ]
        return categories
    
    def _category_name_to_id(self, name: str) -> int:
        """Map category name to ID"""
        mapping = {cat['name']: cat['id'] for cat in self._get_categories()}
        return mapping.get(name, 11)  # Default to general_trash


def main():
    """
    Usage: python make_synthetic.py <object_dir> <background_dir> <output_dir> <count>
    
    Example:
        python scripts/make_synthetic.py \\
            data/taco/objects \\
            data/bin_backgrounds \\
            data/synth \\
            1000
    """
    import sys
    
    if len(sys.argv) < 5:
        print("Usage: python make_synthetic.py <object_dir> <bg_dir> <output_dir> <count>")
        print("\nExample:")
        print("  python scripts/make_synthetic.py \\")
        print("      data/taco/objects \\")
        print("      data/bin_backgrounds \\")
        print("      data/synth \\")
        print("      1000")
        sys.exit(1)
    
    object_dir = sys.argv[1]
    bg_dir = sys.argv[2]
    output_dir = sys.argv[3]
    count = int(sys.argv[4])
    
    generator = SyntheticWasteGenerator(object_dir, bg_dir, output_dir)
    generator.generate_batch(count=count, objects_per_image=3)
    
    logger.info("=" * 60)
    logger.info("SYNTHETIC GENERATION COMPLETE")
    logger.info(f"Next steps:")
    logger.info("1. Merge with real data: python scripts/merge_datasets.py ...")
    logger.info("2. Split train/val (80/20)")
    logger.info("3. Train model: python scripts/train_yolo.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
