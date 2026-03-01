# Training Workflow - Base Training vs Fine-Tuning

## Overview

The model training is now separated into two distinct stages with organized folder structure:

1. **BASE TRAINING** - Train from scratch using general pothole dataset
   - Saved to: `runs/base_models/`
2. **FINE-TUNING** - Transfer learning using annotated/specialized dataset
   - Saved to: `runs/finetuned_models/`

---

## Stage 1: Base Training (From Scratch)

### Purpose
- Train initial model on general pothole detection dataset
- Establish baseline performance
- Create checkpoint for future fine-tuning

### Script
```bash
python train_unified.py
```

### Functions
- `train_yolov10m_base()` - Base training for YOLOv10m
- `train_yolov11m_base()` - Base training for YOLOv11m

### Usage Examples

```bash
# Default: YOLOv10m with augmentation (100 epochs)
python train_unified.py

# YOLOv11m model (75 epochs)
python train_unified.py --model yolov11m --epochs 75

# Without augmentation
python train_unified.py --no-augment

# Custom batch size and epochs
python train_unified.py --epochs 50 --batch-size 8

# YOLOv11 large model with GPU 0
python train_unified.py --model yolov11l --device 0 --epochs 100
```

### Output
- Checkpoint saved to: `runs/base_models/pothole_detector_yolov10m/weights/best.pt`
- Logs and metrics in: `runs/base_models/pothole_detector_yolov10m/`

---

## Stage 2: Fine-Tuning (Annotated Dataset)

### Purpose
- Transfer learning using multi-class annotated dataset
- Improve performance on specific damage types
- Adapt base model to annotated data characteristics

### Script
```bash
python fine_tune.py
```

### Functions
- `fine_tune_yolov10m()` - Fine-tune YOLOv10m from checkpoint
- `fine_tune_yolov11m()` - Fine-tune YOLOv11m from checkpoint

### Usage Examples

```bash
# Basic fine-tuning (from YOLOv10m base checkpoint)
python fine_tune.py --checkpoint runs/detect/pothole_detector_yolov10m/weights/best.pt --epochs 50

# Custom output folder name
python fine_tune.py --checkpoint runs/detect/pothole_detector_yolov10m/weights/best.pt \
  --epochs 100 --batch-size 8 --name finetuned_v2

# Switch to YOLOv11m model
python fine_tune.py --checkpoint runs/detect/pothole_detector_yolov10m/weights/best.pt \
  --model yolov11m --epochs 75

# YOLOv11 large with custom batch size
python fine_tune.py --checkpoint runs/detect/pothole_detector_yolov10m/weights/best.pt \
  --model yolov11l --batch-size 16 --device 0 --name yolov11l_finetuned
```

### Output
- Checkpoint saved to: `runs/finetuned_models/[custom_name]/weights/best.pt`
- Logs and metrics in: `runs/finetuned_models/[custom_name]/`

---

## Recommended Workflow

### First Time Setup
```bash
# 1. Base training on general pothole dataset
python train_unified.py --epochs 100 --batch-size 8

# 2. Evaluate base model performance
# Check metrics at: runs/base_models/pothole_detector_yolov10m/results.csv

# 3. Fine-tune on annotated dataset
python fine_tune.py --checkpoint runs/base_models/pothole_detector_yolov10m/weights/best.pt \
  --epochs 50 --batch-size 8 --name pothole_finetuned_annotated

# 4. Compare fine-tuned vs base model
```

### Iterative Improvement
```bash
# Try different augmentation strategies for base training
python train_unified.py --augment --num-augment 200

# Fine-tune using new base checkpoint
python fine_tune.py --checkpoint runs/base_models/pothole_detector_yolov10m/weights/best.pt \
  --epochs 100 --name iteration_v2

# Try different models
python train_unified.py --model yolov11m --epochs 100
python fine_tune.py --checkpoint runs/base_models/pothole_detector_yolov11m/weights/best.pt \
  --model yolov11m --epochs 50
```

---

## Key Differences

| Aspect | Base Training | Fine-Tuning |
|--------|---------------|------------|
| Script | `train_unified.py` | `fine_tune.py` |
| Starting Weights | Pretrained YOLO model | Checkpoint from base training |
| Dataset | General pothole dataset | Annotated/specialized dataset |
| Learning Rate | Initial/default | Lower (transfer learning) |
| Objective | Establish baseline | Improve on specific annotations |
| Output Folder | `pothole_detector_yolov*` | Custom name (user-defined) |

---

## Dataset Configuration

Both scripts use configuration from:
- **YAML Config**: `data/data.yaml`
- **Dataset Path**: Points to `02_DETAILED_CRACKS_ANNOTATION` (multi-class annotated dataset)

Dataset classes (from YAML):
```yaml
names:
  0: Pot           # Potholes
  1: AllCrack      # All types of cracks
  2: LongCrack     # Longitudinal cracks
  3: LatCrack      # Lateral cracks
```

---

## Troubleshooting

### Base Training Issues
- **Out of Memory**: Reduce batch size with `--batch-size 4`
- **Slow Training**: Check GPU availability with `--device 0`
- **Poor Results**: Enable augmentation with `--augment` and `--num-augment 200`

### Fine-Tuning Issues
- **Checkpoint Not Found**: Verify path to checkpoint exists
- **Model Mismatch**: Ensure `--checkpoint` and `--model` are compatible
- **No Improvement**: Try different epochs or learning strategies

---

## Next Steps

After fine-tuning:
1. Validate model on test dataset
2. Update `.env` with new model path
3. Deploy to backend for inference
4. Monitor performance in production

For deployment, update `.env`:
```
PREDICTING_MODELS_PATH=runs/finetuned_models/[finetuned_model]/weights/best.pt
```
