# Model Storage Structure

## Folder Organization in `runs/`

```
runs/
├── base_models/                    ← BASE TRAINING OUTPUT
│   ├── pothole_detector_yolov10m/
│   │   ├── weights/
│   │   │   ├── best.pt             ← ✓ Use this for fine-tuning
│   │   │   └── last.pt
│   │   ├── results.csv
│   │   └── ...
│   │
│   ├── pothole_detector_yolov11m/
│   │   ├── weights/
│   │   │   ├── best.pt
│   │   │   └── last.pt
│   │   └── ...
│   │
│   └── pothole_detector_yolov11l/
│       ├── weights/
│       │   ├── best.pt
│       │   └── last.pt
│       └── ...
│
└── finetuned_models/               ← FINE-TUNING OUTPUT
    ├── pothole_finetuned_annotated/
    │   ├── weights/
    │   │   ├── best.pt             ← ✓ Production model
    │   │   └── last.pt
    │   ├── results.csv
    │   └── ...
    │
    ├── iteration_v2/
    │   ├── weights/
    │   │   ├── best.pt
    │   │   └── last.pt
    │   └── ...
    │
    └── yolov11l_finetuned/
        ├── weights/
        │   ├── best.pt
        │   └── last.pt
        └── ...
```

---

## Usage Pattern

### Stage 1: Base Training
```bash
python train_unified.py
↓
Saved to: runs/base_models/pothole_detector_yolov10m/weights/best.pt
```

### Stage 2: Fine-Tuning
```bash
python fine_tune.py --checkpoint runs/base_models/pothole_detector_yolov10m/weights/best.pt
↓
Saved to: runs/finetuned_models/pothole_finetuned_annotated/weights/best.pt
```

### Stage 3: Deployment
```bash
# Update .env
PREDICTING_MODELS_PATH=runs/finetuned_models/pothole_finetuned_annotated/weights/best.pt
```

---

## Key Points

1. **Base Models** - Stored separately in `runs/base_models/`
   - Clean organization of initial models
   - Easy to find checkpoints for fine-tuning
   - Multiple base models can coexist

2. **Fine-Tuned Models** - Stored separately in `runs/finetuned_models/`
   - Production-ready models
   - Can iterate on fine-tuning without polluting base models
   - Custom naming for different versions

3. **Model Selection**
   - Always use best.pt for inference
   - Use last.pt for resuming interrupted training
   - Keep base_models for reference and comparison

---

## Example Iteration Workflow

```bash
# Iteration 1: Base training (YOLOv10m)
python train_unified.py --epochs 100
# Output: runs/base_models/pothole_detector_yolov10m/weights/best.pt

# Iteration 1: Fine-tuning
python fine_tune.py --checkpoint runs/base_models/pothole_detector_yolov10m/weights/best.pt \
  --epochs 50 --name finetuned_v1
# Output: runs/finetuned_models/finetuned_v1/weights/best.pt

# Iteration 2: Fine-tuning with different settings
python fine_tune.py --checkpoint runs/base_models/pothole_detector_yolov10m/weights/best.pt \
  --epochs 75 --batch-size 8 --name finetuned_v2
# Output: runs/finetuned_models/finetuned_v2/weights/best.pt

# Iteration 3: Try different base model
python train_unified.py --model yolov11m --epochs 120
# Output: runs/base_models/pothole_detector_yolov11m/weights/best.pt

# Fine-tune new base model
python fine_tune.py --checkpoint runs/base_models/pothole_detector_yolov11m/weights/best.pt \
  --model yolov11m --epochs 50 --name yolov11m_finetuned
# Output: runs/finetuned_models/yolov11m_finetuned/weights/best.pt

# Compare and deploy best model
# Check results at: runs/finetuned_models/finetuned_v2/results.csv
# If best, update .env to use: runs/finetuned_models/finetuned_v2/weights/best.pt
```

---

## Cleanup Guidelines

### Keep
- ✓ Base model checkpoints (for re-finetuning)
- ✓ Best fine-tuned models (for production)
- ✓ Latest results.csv files (for comparison)

### Safe to Delete
- ✗ Old fine-tuned iterations (if not needed for comparison)
- ✗ last.pt files (after training completes successfully)
- ✗ Previous base models (if new versions are better)

```bash
# Example: Clean up old fine-tuned versions (keeping latest)
rm -r runs/finetuned_models/finetuned_v1/
rm -r runs/finetuned_models/iteration_*/ # except latest
```

---

## Quick Reference

| Task | Command | Output Location |
|------|---------|-----------------|
| Base training | `python train_unified.py` | `runs/base_models/` |
| Fine-tuning | `python fine_tune.py --checkpoint ...` | `runs/finetuned_models/` |
| Get model path | ls -la `runs/base_models/*/weights/best.pt` | Model path |
| Deploy model | Update .env `PREDICTING_MODELS_PATH` | Production |
| View results | cat `runs/finetuned_models/*/results.csv` | Metrics |
