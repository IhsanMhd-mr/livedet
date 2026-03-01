"""
FINE-TUNING SCRIPT - Transfer Learning for Pothole Detection
Loads pre-trained checkpoint and fine-tunes on new/different dataset
Ideal for improving model performance on specific annotations or new data

Usage:
    python fine_tune.py --checkpoint runs/detect/pothole_detector_yolov10m/weights/best.pt --epochs 50
    python fine_tune.py --checkpoint runs/detect/pothole_detector_yolov10m/weights/best.pt --epochs 100 --batch-size 8 --name finetuned_v2
    python fine_tune.py --checkpoint runs/detect/pothole_detector_yolov10m/weights/best.pt --model yolov11m --epochs 75
"""

import sys
import logging
import argparse
from pathlib import Path
import os

# ═════════════════════════════════════════════════════════════════════════════════
# VERIFY PYTHON ENVIRONMENT
# ═════════════════════════════════════════════════════════════════════════════════

python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
python_path = sys.executable
venv_gpu_path = str(Path(__file__).parent.parent / 'venv-gpu').lower()

if venv_gpu_path not in python_path.lower():
    print("[ERROR] Not using venv-gpu environment!")
    print(f"Expected: {venv_gpu_path}")
    print(f"Actual: {python_path}")
    sys.exit(1)
elif not python_version.startswith("3.10"):
    print(f"[ERROR] Python version mismatch! Got {python_version}, expected 3.10.x")
    sys.exit(1)
else:
    print(f"[OK] Using venv-gpu (Python {python_version})")

# Disable font downloads
os.environ['YOLOV5_DISABLE_TELEMETRY'] = '1'

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ═════════════════════════════════════════════════════════════════════════════════

try:
    from ultralytics import YOLO
except ImportError:
    logger.error("ultralytics not installed. Run: pip install ultralytics")
    sys.exit(1)

# Load config
backend_path = Path(__file__).parent.parent / 'backend'
sys.path.insert(0, str(backend_path))
from config import config

# ═════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION & PATHS
# ═════════════════════════════════════════════════════════════════════════════════

project_root = Path(__file__).parent.parent
data_dir = project_root / 'data'
data_yaml = data_dir / 'data.yaml'


# ═════════════════════════════════════════════════════════════════════════════════
# FINE-TUNING FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════════

def fine_tune_yolov10m(checkpoint_path, epochs, batch_size, device, patience, output_name):
    """
    Fine-tune YOLOv10m from a pretrained checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint.pt file
        epochs: Number of fine-tuning epochs
        batch_size: Batch size for training
        device: Device to train on (cpu, 0, 1, etc.)
        patience: Early stopping patience
        output_name: Name of output subfolder
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("FINE-TUNING YOLOv10m (From Checkpoint)")
    logger.info("=" * 80)
    logger.info(f"Configuration:")
    logger.info(f"  Model: YOLOv10m")
    logger.info(f"  Checkpoint: {checkpoint_path}")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Device: {device}")
    logger.info(f"  Patience: {patience}")
    logger.info(f"  Dataset: {data_yaml}")
    logger.info(f"  Output folder: {output_name}")
    logger.info("")
    
    # Verify checkpoint exists
    if not Path(checkpoint_path).exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    # Load checkpoint
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    model = YOLO(checkpoint_path)
    logger.info("✓ Checkpoint loaded successfully")
    logger.info("")
    
    # Start fine-tuning
    logger.info("Starting fine-tuning...")
    try:
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=640,
            batch=batch_size,
            device=device,
            patience=patience,
            save=True,
            project=str(project_root / 'runs' / 'finetuned_models'),
            name=output_name,
            workers=0,
            close_mosaic=5,
            plots=False,
        )
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("✓ YOLOv10m FINE-TUNING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Results: runs/finetuned_models/{output_name}/")
        logger.info(f"Best model: runs/finetuned_models/{output_name}/weights/best.pt")
        logger.info("")
        
        return results
        
    except Exception as e:
        logger.error(f"Fine-tuning failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def fine_tune_yolov11m(checkpoint_path, epochs, batch_size, device, patience, model_size, output_name):
    """
    Fine-tune YOLOv11 from a pretrained checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint.pt file
        epochs: Number of fine-tuning epochs
        batch_size: Batch size for training
        device: Device to train on (cpu, 0, 1, etc.)
        patience: Early stopping patience
        model_size: Model size (n, s, m, l, x)
        output_name: Name of output subfolder
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"FINE-TUNING YOLOv11{model_size} (From Checkpoint)")
    logger.info("=" * 80)
    logger.info(f"Configuration:")
    logger.info(f"  Model: YOLOv11{model_size}")
    logger.info(f"  Checkpoint: {checkpoint_path}")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Device: {device}")
    logger.info(f"  Patience: {patience}")
    logger.info(f"  Dataset: {data_yaml}")
    logger.info(f"  Output folder: {output_name}")
    logger.info("")
    
    # Verify checkpoint exists
    if not Path(checkpoint_path).exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    # Load checkpoint
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    model = YOLO(checkpoint_path)
    logger.info("✓ Checkpoint loaded successfully")
    logger.info("")
    
    # Start fine-tuning
    logger.info("Starting fine-tuning...")
    try:
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=640,
            batch=batch_size,
            device=device,
            patience=patience,
            save=True,
            project=str(project_root / 'runs' / 'finetuned_models'),
            name=output_name,
            workers=0,
            close_mosaic=5,
            plots=False,
        )
        
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"✓ YOLOv11{model_size} FINE-TUNING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Results: runs/finetuned_models/{output_name}/")
        logger.info(f"Best model: runs/finetuned_models/{output_name}/weights/best.pt")
        logger.info("")
        
        return results
        
    except Exception as e:
        logger.error(f"Fine-tuning failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ═════════════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSER
# ═════════════════════════════════════════════════════════════════════════════════

def parse_arguments():
    """Parse command-line arguments for fine-tuning"""
    parser = argparse.ArgumentParser(
        description="Fine-tuning script with transfer learning for pothole detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fine_tune.py --checkpoint runs/detect/pothole_detector_yolov10m/weights/best.pt --epochs 50
  python fine_tune.py --checkpoint runs/detect/pothole_detector_yolov10m/weights/best.pt --epochs 100 --batch-size 8 --name finetuned_v2
  python fine_tune.py --checkpoint runs/detect/pothole_detector_yolov10m/weights/best.pt --model yolov11m --epochs 75
  python fine_tune.py --checkpoint runs/detect/pothole_detector_yolov10m/weights/best.pt --model yolov11l --batch-size 16 --device 0 --name yolov11l_finetuned
        """
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to checkpoint to fine-tune from (required)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='yolov10m',
        choices=['yolov10m', 'yolov11n', 'yolov11s', 'yolov11m', 'yolov11l', 'yolov11x'],
        help='Model architecture (default: yolov10m)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of fine-tuning epochs (default: 10)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size for fine-tuning (default: 16)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='gpu',
        choices=['cpu', 'gpu', '0', '1'],
        help='Device to train on (default: gpu)'
    )
    
    parser.add_argument(
        '--patience',
        type=int,
        default=20,
        help='Early stopping patience in epochs (default: 20)'
    )
    
    parser.add_argument(
        '--name',
        type=str,
        default=None,
        help='Custom output folder name'
    )
    
    return parser.parse_args()


# ═════════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════════

def main():
    """Main fine-tuning function"""
    
    # Verify data.yaml exists
    if not data_yaml.exists():
        logger.error(f"data.yaml not found: {data_yaml}")
        sys.exit(1)
    
    # Parse arguments
    args = parse_arguments()
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("FINE-TUNING POTHOLE DETECTION MODEL")
    logger.info("=" * 80)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Data: {data_yaml}")
    logger.info("=" * 80)
    
    # Determine output name
    if args.name:
        output_name = args.name
    else:
        checkpoint_name = Path(args.checkpoint).parent.parent.name
        output_name = f"{checkpoint_name}_finetuned"
    
    # Fine-tune model
    if args.model == 'yolov10m':
        fine_tune_yolov10m(args.checkpoint, args.epochs, args.batch_size, args.device, args.patience, output_name)
    elif args.model.startswith('yolov11'):
        model_size = args.model.replace('yolov11', '')
        fine_tune_yolov11m(args.checkpoint, args.epochs, args.batch_size, args.device, args.patience, model_size, output_name)
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("✓ Fine-tuning script completed!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
