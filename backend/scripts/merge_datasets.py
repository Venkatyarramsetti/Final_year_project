"""
Merge multiple COCO-format datasets into a single training dataset
Combines TACO + local annotations + any other COCO JSON files
Handles image copying and ID remapping
"""

import json
import shutil
import os
import sys
from pathlib import Path
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class COCODatasetMerger:
    def __init__(self):
        self.merged_data = {
            "images": [],
            "annotations": [],
            "categories": []
        }
        self.next_img_id = 1
        self.next_ann_id = 1
        self.category_map = {}  # old_id -> new_id
        self.category_name_to_id = {}  # name -> new_id
        
    def merge_categories(self, categories):
        """Merge categories from dataset, avoiding duplicates by name"""
        for cat in categories:
            cat_name = cat["name"].lower()
            if cat_name not in self.category_name_to_id:
                new_id = len(self.category_name_to_id) + 1
                self.category_name_to_id[cat_name] = new_id
                self.merged_data["categories"].append({
                    "id": new_id,
                    "name": cat_name,
                    "supercategory": cat.get("supercategory", "waste")
                })
            # Map old category ID to new ID
            self.category_map[cat["id"]] = self.category_name_to_id[cat_name]
    
    def merge_images(self, images, annotations, source_dir, out_img_dir):
        """Merge images and annotations with ID remapping"""
        img_id_map = {}  # old_id -> new_id
        
        for img in images:
            old_img_id = img["id"]
            new_img_id = self.next_img_id
            img_id_map[old_img_id] = new_img_id
            
            # Copy image file
            src_img_path = Path(source_dir) / img["file_name"]
            dst_img_path = Path(out_img_dir) / f"img_{new_img_id:06d}.jpg"
            
            if src_img_path.exists():
                dst_img_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copy(src_img_path, dst_img_path)
                    logger.debug(f"Copied: {src_img_path} -> {dst_img_path}")
                except Exception as e:
                    logger.warning(f"Failed to copy {src_img_path}: {e}")
                    continue
            else:
                logger.warning(f"Image not found: {src_img_path}")
                continue
            
            # Add image to merged dataset
            self.merged_data["images"].append({
                "id": new_img_id,
                "file_name": dst_img_path.name,
                "width": img.get("width", 640),
                "height": img.get("height", 480)
            })
            
            self.next_img_id += 1
        
        # Merge annotations with remapped IDs
        for ann in annotations:
            if ann["image_id"] not in img_id_map:
                continue  # Skip annotations for missing images
            
            self.merged_data["annotations"].append({
                "id": self.next_ann_id,
                "image_id": img_id_map[ann["image_id"]],
                "category_id": self.category_map.get(ann["category_id"], 1),
                "bbox": ann.get("bbox", [0, 0, 1, 1]),
                "area": ann.get("area", 1),
                "segmentation": ann.get("segmentation", []),
                "iscrowd": ann.get("iscrowd", 0)
            })
            self.next_ann_id += 1
    
    def merge_datasets(self, coco_files, output_json, output_img_dir):
        """Main function to merge multiple COCO datasets"""
        logger.info(f"Merging {len(coco_files)} datasets...")
        
        for coco_file in coco_files:
            logger.info(f"Processing: {coco_file}")
            try:
                with open(coco_file, 'r') as f:
                    data = json.load(f)
                
                # Get source directory (where images are located)
                source_dir = Path(coco_file).parent
                
                # Merge categories first
                categories = data.get("categories", [])
                self.merge_categories(categories)
                
                # Merge images and annotations
                images = data.get("images", [])
                annotations = data.get("annotations", [])
                self.merge_images(images, annotations, source_dir, output_img_dir)
                
                logger.info(f"✓ Merged {len(images)} images, {len(annotations)} annotations")
                
            except Exception as e:
                logger.error(f"Error processing {coco_file}: {e}")
                continue
        
        # Save merged dataset
        logger.info(f"Saving merged dataset to: {output_json}")
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_json, 'w') as f:
            json.dump(self.merged_data, f, indent=2)
        
        # Print summary
        logger.info("=" * 60)
        logger.info("MERGE COMPLETE")
        logger.info(f"Total images: {len(self.merged_data['images'])}")
        logger.info(f"Total annotations: {len(self.merged_data['annotations'])}")
        logger.info(f"Total categories: {len(self.merged_data['categories'])}")
        logger.info(f"Categories: {[c['name'] for c in self.merged_data['categories']]}")
        logger.info(f"Output JSON: {output_json}")
        logger.info(f"Output images: {output_img_dir}")
        logger.info("=" * 60)


def main():
    """
    Usage: python merge_datasets.py <json1> <json2> ... <output.json> <output_img_dir>
    Example:
        python scripts/merge_datasets.py \\
            data/taco/annotations.json \\
            data/local/annotations.json \\
            data/annotations/merged.json \\
            data/images/train
    """
    if len(sys.argv) < 4:
        print("Usage: python merge_datasets.py <json1> [json2 ...] <output.json> <output_img_dir>")
        print("\nExample:")
        print("  python scripts/merge_datasets.py \\")
        print("      data/taco/annotations.json \\")
        print("      data/local/annotations.json \\")
        print("      data/annotations/merged.json \\")
        print("      data/images/train")
        sys.exit(1)
    
    input_files = sys.argv[1:-2]
    output_json = sys.argv[-2]
    output_img_dir = sys.argv[-1]
    
    merger = COCODatasetMerger()
    merger.merge_datasets(input_files, output_json, output_img_dir)


if __name__ == "__main__":
    main()
