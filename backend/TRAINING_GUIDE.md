# 🎯 Complete Waste Detection Model Training Guide

## 📋 Table of Contents
1. [Quick Start](#quick-start)
2. [Problem Analysis](#problem-analysis)
3. [Directory Structure](#directory-structure)
4. [Dataset Preparation](#dataset-preparation)
5. [Training Pipeline](#training-pipeline)
6. [Evaluation & Optimization](#evaluation--optimization)
7. [Deployment](#deployment)
8. [Expected Results](#expected-results)
9. [Troubleshooting](#troubleshooting)

---

## ⚡ Quick Start

```bash
# 1. Install dependencies
cd backend
.\hazard_env\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Prepare dataset (see detailed steps below)
# Download TACO dataset, add your local images

# 3. Generate synthetic data
python scripts/make_synthetic.py data/taco/objects data/bin_backgrounds data/synth 1000

# 4. Merge datasets
python scripts/merge_datasets.py data/taco/annotations.json data/local/annotations.json data/synth/annotations.json data/annotations/merged.json data/images/train

# 5. Train model (GPU recommended)
python scripts/train_yolo.py --epochs 150 --batch 16 --imgsz 640

# 6. Evaluate thresholds
python scripts/eval_thresholds.py outputs/waste_yolov8x_seg/weights/best.pt data/images/val configs/data.yaml

# 7. Test with context-aware detection
python scripts/postprocess_detector.py outputs/waste_yolov8x_seg/weights/best.pt test_images/trash_bin.jpg --output result.jpg
```

---

## 🔍 Problem Analysis

### Why Current Model Has Low Accuracy

**Root Causes:**
1. **Domain Mismatch**: COCO pretraining → general objects (bottle, bowl), NOT waste context
2. **Transparent Plastics**: Thin plastic bags look different from COCO training images
3. **Class Imbalance**: Hazardous/plastic_bag underrepresented, general waste overrepresented
4. **Low Threshold (0.25)**: High recall but many false positives
5. **No Context Rules**: "bottle" detected as safe even inside trash bin
6. **Insufficient Segmentation**: Bounding boxes don't capture thin/transparent edges

### Solution Strategy

**Priority Improvements:**
- ✅ **Domain Dataset**: TACO (Trash Annotations in Context) + local waste images
- ✅ **Segmentation Model**: YOLOv8x-seg for transparent plastic masks
- ✅ **Synthetic Augmentation**: 1000+ composites of plastic bags on bin backgrounds
- ✅ **Per-Class Thresholds**: hazardous=0.60, plastic=0.50, general=0.45
- ✅ **Context-Aware Logic**: Object inside bin polygon = waste
- ✅ **Class Balancing**: Upsample rare classes (hazardous x10, plastic_bag x5)

---

## 📁 Directory Structure

```
backend/
├── configs/
│   ├── data.yaml                # Dataset config (12 waste classes)
│   └── hyp.yaml                 # Hyperparameters (optimized for waste)
├── data/
│   ├── taco/                    # TACO dataset (download from tacodataset.org)
│   │   ├── images/
│   │   ├── annotations.json
│   │   └── objects/             # Segmented objects for synthesis
│   ├── local/                   # Your captured images
│   │   ├── images/
│   │   └── annotations.json
│   ├── bin_backgrounds/         # Empty bin/trash backgrounds for synthesis
│   ├── synth/                   # Generated synthetic composites
│   │   ├── synth_00001.jpg
│   │   └── annotations.json
│   ├── images/
│   │   ├── train/               # Training images (merged)
│   │   └── val/                 # Validation images (20% holdout)
│   └── annotations/
│       └── merged.json          # Merged COCO annotations
├── scripts/
│   ├── merge_datasets.py        # Merge TACO + local + synth
│   ├── make_synthetic.py        # Generate synthetic composites
│   ├── train_yolo.py            # Training script
│   ├── eval_thresholds.py       # Per-class threshold optimization
│   └── postprocess_detector.py  # Context-aware detection (in_bin)
├── models/                      # Trained model weights
│   └── best.pt                  # Production-ready model
└── outputs/                     # Training outputs
    └── waste_yolov8x_seg/
        ├── weights/
        │   ├── best.pt
        │   └── last.pt
        ├── results.csv
        └── confusion_matrix.png
```

---

## 📦 Dataset Preparation

### Step 1: Download TACO Dataset

**TACO (Trash Annotations in Context)** is the gold standard for waste detection.

```bash
# Download from: http://tacodataset.org/
# Or use Kaggle: https://www.kaggle.com/datasets/kneroma/tacotrashdataset

# Extract to backend/data/taco/
# Expected structure:
# data/taco/
#   ├── images/
#   │   ├── batch_1/
#   │   ├── batch_2/
#   │   └── ...
#   └── annotations.json
```

**TACO Statistics:**
- 1500+ images from 10+ countries
- 60+ waste categories (bottles, bags, cans, food waste)
- Segmentation masks for transparent objects
- Context annotations (in_bin, on_ground, in_water)

### Step 2: Add Your Local Images

Capture domain-specific images (cafeteria, street, office bins):

```bash
# backend/data/local/images/
# - Capture 100-500 images in YOUR environment
# - Various lighting, angles, bin types
# - Mix of full/empty bins, single/multiple items

# Annotate using:
# - CVAT (Computer Vision Annotation Tool): cvat.ai
# - LabelMe: github.com/wkentaro/labelme
# - Roboflow: roboflow.com

# Export as COCO JSON format
# Save to: data/local/annotations.json
```

**Annotation Guidelines:**
- **Plastic bags**: Draw precise segmentation masks (not just boxes)
- **Overlapping items**: Mark all visible items separately
- **Containers/bins**: Label bins separately (class: "bin")
- **Hazardous items**: High-quality annotations for batteries, chemicals

### Step 3: Create Bin Background Dataset

```bash
# Capture 50-100 images of EMPTY bins/trash backgrounds
# Save to: data/bin_backgrounds/
# - Various bin types (indoor, outdoor, metal, plastic)
# - Different lighting conditions
# - Clean backgrounds for synthetic compositing
```

### Step 4: Extract Objects for Synthesis

```bash
# From TACO or local images, extract segmented objects as PNGs with alpha
# Use this Python script:

from PIL import Image
import numpy as np
import cv2

def extract_object(image_path, mask, output_path):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = mask * 255  # Set alpha from mask
    cv2.imwrite(output_path, rgba)

# Save to: data/taco/objects/plastic_bag_001.png, bottle_045.png, etc.
```

### Step 5: Generate Synthetic Composites

```powershell
# Generate 1000 synthetic images (plastic bags on bin backgrounds)
python scripts/make_synthetic.py `
    data/taco/objects `
    data/bin_backgrounds `
    data/synth `
    1000

# Output: data/synth/ with 1000 images + annotations.json
```

**Why Synthetic Data?**
- **Class Imbalance**: Upsample rare hazardous/plastic_bag classes
- **Transparent Plastics**: Train model on various lighting through plastic
- **Cost-Effective**: 1000 synthetic images = 1 day annotation work
- **Controlled Variation**: Systematic augmentation (rotation, scale, lighting)

### Step 6: Merge All Datasets

```powershell
# Merge TACO + local + synthetic into single training set
python scripts/merge_datasets.py `
    data/taco/annotations.json `
    data/local/annotations.json `
    data/synth/annotations.json `
    data/annotations/merged.json `
    data/images/train

# This will:
# - Remap category IDs
# - Copy all images to data/images/train/
# - Create unified merged.json
```

### Step 7: Train/Val Split

```powershell
# Split merged dataset into train (80%) and val (20%)
# Manual split or use this script:

cd data/images
mkdir val
# Move 20% of images to val/ (stratified by class)
```

**Stratified Split Guidelines:**
- Ensure all 12 classes present in both train/val
- Hazardous class: keep at least 10 examples in val
- Different scenes/lighting in val (held-out distribution)

---

## 🚀 Training Pipeline

### Phase 1: Baseline Training (TACO Only)

**Goal**: Establish baseline performance (~60-70% mAP)

```powershell
# Train 50 epochs on TACO dataset only
python scripts/train_yolo.py `
    --model yolov8x-seg.pt `
    --epochs 50 `
    --batch 16 `
    --imgsz 640 `
    --name waste_baseline `
    --project outputs

# Expected results after 50 epochs:
# - mAP@0.5: 55-65%
# - plastic_bag AP: 40-50% (low - needs improvement)
# - hazardous AP: 50-60%
```

**What to Monitor:**
- Training loss should decrease steadily
- Validation mAP should plateau around epoch 40-50
- Check `outputs/waste_baseline/results.csv` for metrics

### Phase 2: Fine-Tuning with Synthetics

**Goal**: Improve plastic detection and class balance (75-85% mAP)

```powershell
# Resume from baseline, add synthetic data, train 100 more epochs
python scripts/train_yolo.py `
    --model outputs/waste_baseline/weights/best.pt `
    --epochs 150 `
    --batch 16 `
    --imgsz 640 `
    --name waste_finetuned `
    --project outputs

# Expected results after 150 epochs:
# - mAP@0.5: 75-85%
# - plastic_bag AP: 70-80% (improved!)
# - hazardous AP: 80-90% (high precision)
```

### Phase 3: High-Resolution Polish

**Goal**: Final polish for production (85-90% mAP)

```powershell
# Train with larger image size for detail
python scripts/train_yolo.py `
    --model outputs/waste_finetuned/weights/best.pt `
    --epochs 50 `
    --batch 8 `
    --imgsz 1280 `
    --name waste_production `
    --project outputs

# Expected results:
# - mAP@0.5: 85-90%
# - All classes AP > 80%
# - Hazardous precision > 90%
```

### Training Parameters Explained

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| `model` | yolov8x-seg.pt | Segmentation for transparent plastics |
| `epochs` | 150 | Sufficient for convergence |
| `batch` | 16 | Balance GPU memory and training speed |
| `imgsz` | 640 | Training size (use 1280 for inference) |
| `optimizer` | AdamW | Better for fine-tuning than SGD |
| `lr0` | 0.001 | Conservative learning rate |
| `mosaic` | 1.0 | Combines 4 images (data augmentation) |
| `mixup` | 0.15 | Blends images (handles occlusion) |
| `copy_paste` | 0.3 | Paste masks (critical for synthetics) |
| `fl_gamma` | 2.0 | Focal loss (downweight easy negatives) |

---

## 📊 Evaluation & Optimization

### Step 1: Validate Trained Model

```powershell
# Run validation to get per-class AP
python scripts/train_yolo.py `
    --validate outputs/waste_production/weights/best.pt `
    --imgsz 640
```

**Key Metrics to Check:**
- **mAP@0.5**: Should be > 75% for production
- **Hazardous AP**: Should be > 85% (safety-critical)
- **Plastic_bag AP**: Should be > 70% (was difficult initially)
- **Confusion Matrix**: Check which classes get confused

### Step 2: Optimize Per-Class Thresholds

```powershell
# Analyze validation set to find optimal thresholds
python scripts/eval_thresholds.py `
    outputs/waste_production/weights/best.pt `
    data/images/val `
    configs/data.yaml `
    --precision 0.90 `
    --recall 0.80 `
    --output threshold_report.json
```

**Output**: `threshold_report.json`
```json
{
  "optimal_thresholds": {
    "hazardous": {
      "recommended_threshold": 0.60,
      "reasoning": "High threshold - critical class requiring high precision"
    },
    "plastic_bag": {
      "recommended_threshold": 0.50,
      "reasoning": "Balance detection sensitivity with accuracy"
    },
    "general_trash": {
      "recommended_threshold": 0.45,
      "reasoning": "General waste, prioritize recall"
    }
  }
}
```

### Step 3: Test Context-Aware Detection

```powershell
# Test in_bin logic with real images
python scripts/postprocess_detector.py `
    outputs/waste_production/weights/best.pt `
    test_images/trash_bin.jpg `
    --threshold-config threshold_report.json `
    --in-bin-threshold 0.5 `
    --output results/output.jpg `
    --json results/output.json
```

**Expected Behavior:**
- Bottles inside bin → Classified as WASTE (red boxes)
- Bottles outside bin → Classified as SAFE (green boxes)
- Spatial reasoning overrides generic COCO labels

---

## 🚢 Deployment

### Step 1: Update model_manager.py

Apply per-class thresholds from `threshold_report.json`:

```python
# Add to GarbageDetectionModel class
self.class_thresholds = {
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

# In detect_and_categorize(), apply per-class threshold:
class_threshold = self.class_thresholds.get(class_name, 0.25)
if conf < class_threshold:
    continue  # Skip detection below class-specific threshold
```

### Step 2: Copy Trained Model

```powershell
# Copy production model to models/
cp outputs/waste_production/weights/best.pt models/best.pt
```

### Step 3: Restart Backend

```powershell
cd backend
.\hazard_env\Scripts\Activate.ps1
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Model will auto-load `models/best.pt` (priority over yolov8x.pt).

### Step 4: Export for Faster Inference (Optional)

```powershell
# Export to ONNX for 2-3x faster inference
python scripts/train_yolo.py --export outputs/waste_production/weights/best.pt

# This creates best.onnx
# Update model_manager.py to load ONNX instead of PT
```

---

## 📈 Expected Results

### Training Timeline

| Phase | Epochs | Dataset | mAP@0.5 | Plastic AP | Hazardous AP | Time (V100) |
|-------|--------|---------|---------|------------|--------------|-------------|
| Baseline | 50 | TACO only | 60-70% | 45-55% | 55-65% | ~4 hours |
| Fine-tune | 150 | + Synthetics | 75-85% | 70-80% | 80-90% | ~12 hours |
| Polish | 50 | High-res | 85-90% | 80-90% | 85-95% | ~6 hours |

### Real-World Performance

**Test Scenarios:**
1. **Empty Bin** → Safe (0 detections)
2. **Plastic Bag in Bin** → Hazardous (plastic_bag detected)
3. **Mixed Trash** → Hazardous (multiple waste items)
4. **Fresh Vegetables** → Safe (food items not in bin)
5. **Bottle Outside Bin** → Safe (context override)
6. **Bottle Inside Bin** → Hazardous (in_bin logic)

**Before vs After:**
- **Before** (COCO model): 0% plastic bags detected, bottles always "Safe"
- **After** (domain model): 80%+ plastic bags detected, context-aware waste classification

---

## 🔧 Troubleshooting

### Issue: GPU Out of Memory

**Solution:**
```powershell
# Reduce batch size
python scripts/train_yolo.py --batch 8  # or --batch -1 for AutoBatch

# Reduce image size
python scripts/train_yolo.py --imgsz 320  # faster but less accurate
```

### Issue: Low Plastic Bag Detection

**Solution:**
- Add more synthetic composites (2000-5000 images)
- Use transparent augmentations (lighting, blur)
- Check segmentation masks quality (must be precise)
- Increase `copy_paste` augmentation to 0.5

### Issue: High False Positives

**Solution:**
- Increase per-class thresholds (hazardous → 0.70)
- Use stricter NMS IoU (0.6 instead of 0.45)
- Add more negative examples (clean environments)

### Issue: Model Not Converging

**Solution:**
- Check data quality (bad annotations?)
- Reduce learning rate (`lr0=0.0005`)
- Disable `mosaic` last 20 epochs
- Ensure GPU is being used (check logs)

### Issue: Context Detection Not Working

**Solution:**
- Verify bins are being detected (class: "bin")
- Check overlap threshold (try 0.3 instead of 0.5)
- Use segmentation masks (not just bounding boxes)
- Debug with `postprocess_detector.py --visualize`

---

## 📚 Additional Resources

**Datasets:**
- TACO: http://tacodataset.org/
- OpenLitterMap: https://openlittermap.com/
- Drinking Waste Dataset: https://www.kaggle.com/

**Papers:**
- "TACO: Trash Annotations in Context" (2020)
- "YOLOv8: State-of-the-Art Object Detection" (Ultralytics)
- "Focal Loss for Dense Object Detection" (Lin et al.)

**Tools:**
- CVAT Annotation: https://cvat.ai/
- Roboflow Dataset Management: https://roboflow.com/
- Ultralytics Docs: https://docs.ultralytics.com/

---

## 🎯 Next Steps

1. **Active Learning Loop**
   - Deploy model to production
   - Collect low-confidence detections (0.25-0.60)
   - Human review + correction
   - Retrain every 1000 new samples

2. **Multi-Material Classification**
   - Add material head (plastic, metal, glass, organic)
   - Enables recycling categorization
   - Train with material labels

3. **Depth/3D Detection**
   - Use depth camera or stereo vision
   - Better in_bin detection for occluded items
   - Spatial volume estimation

4. **Mobile Deployment**
   - Export to TensorFlow Lite
   - Run on mobile devices (edge AI)
   - Real-time bin monitoring

---

## ✅ Success Checklist

- [ ] Downloaded TACO dataset
- [ ] Annotated 100+ local images
- [ ] Generated 1000+ synthetic composites
- [ ] Merged all datasets
- [ ] Trained baseline model (50 epochs)
- [ ] Fine-tuned with synthetics (150 epochs)
- [ ] Achieved mAP > 75%
- [ ] Optimized per-class thresholds
- [ ] Tested context-aware detection
- [ ] Deployed to production
- [ ] Set up active learning pipeline

---

**Questions or Issues?**
- Check `outputs/waste_*/results.csv` for training metrics
- Review `confusion_matrix.png` for class errors
- Test with `postprocess_detector.py` for visual debugging
- Consult Ultralytics docs: https://docs.ultralytics.com/

**Good luck with your training! 🚀**
