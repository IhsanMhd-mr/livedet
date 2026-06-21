"""
AUTOMATED FINE-TUNING LOOP
Iteratively trains YOLO11s on the cleaned dataset, checks if validation mAP@50
reaches the 85% target, and if not, automatically adjusts parameters and trains again.

Usage:
    python model_training/auto_tune.py --target-map 0.85 --max-rounds 3
"""

import sys
import logging
import argparse
import time
from pathlib import Path
import os

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Resolve paths
project_root = Path(__file__).parent.parent
data_yaml = project_root / 'data' / 'data.yaml'

try:
    from ultralytics import YOLO
except ImportError:
    logger.error("ultralytics not installed. Run: pip install ultralytics")
    sys.exit(1)


def run_training_round(checkpoint_path, round_num, lr, imgsz, freeze, epochs=60):
    """Runs a single fine-tuning round and returns the best model path and its mAP@50 score."""
    output_name = f"pothole_yolo11s_autotune_r{round_num}"
    logger.info(f"\n" + "=" * 80)
    logger.info(f"STARTING AUTOTUNE ROUND {round_num}")
    logger.info(f"  Base weights:   {checkpoint_path}")
    logger.info(f"  Learning Rate:  {lr}")
    logger.info(f"  Image Size:     {imgsz}")
    logger.info(f"  Freeze layers:  {freeze}")
    logger.info(f"  Max Epochs:     {epochs}")
    logger.info("=" * 80)

    model = YOLO(str(checkpoint_path))

    # We use a solid augmentation profile for all rounds
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=8,
        device=0,
        patience=15,  # Slightly shorter patience to keep autotune loops fast
        save=True,
        project=str(project_root / 'runs' / 'base_models'),
        name=output_name,
        workers=2,
        close_mosaic=5,
        plots=True,
        freeze=freeze,
        # Hyperparameters
        lr0=lr,
        lrf=0.01,
        warmup_epochs=3,
        cos_lr=True,
        # Native augmentations
        mosaic=1.0,
        mixup=0.15,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
    )

    # Perform evaluation at the final resolution
    logger.info(f"\nEvaluating Round {round_num} results...")
    best_weights = project_root / 'runs' / 'base_models' / output_name / 'weights' / 'best.pt'
    if not best_weights.exists():
        logger.error(f"Best weight file not found for round {round_num}")
        return None, 0.0

    eval_model = YOLO(str(best_weights))
    metrics = eval_model.val(
        data=str(data_yaml),
        device=0,
        plots=False,
    )

    map50 = float(metrics.results_dict.get('metrics/mAP50(B)', 0))
    logger.info(f"\nRound {round_num} Finished. Validation mAP@50: {map50:.4f} ({map50*100:.2f}%)")
    return best_weights, map50


def main():
    parser = argparse.ArgumentParser(description="Automated Iterative YOLO Fine-Tuner")
    parser.add_argument('--checkpoint', type=str, default=None, help='Initial checkpoint weights')
    parser.add_argument('--target-map', type=float, default=0.85, help='Target mAP@50 score (e.g. 0.85)')
    parser.add_argument('--max-rounds', type=int, default=3, help='Maximum fine-tuning iterations')
    args = parser.parse_args()

    # Determine initial checkpoint
    initial_checkpoint = args.checkpoint
    if not initial_checkpoint:
        default_pt = project_root / 'runs' / 'base_models' / 'pothole_detector_yolo11s' / 'weights' / 'best.pt'
        if default_pt.exists():
            initial_checkpoint = str(default_pt)
        else:
            initial_checkpoint = 'yolo11s.pt'

    logger.info(f"Starting autotune script. Target: {args.target_map*100:.1f}% mAP@50")

    current_weights = initial_checkpoint
    current_lr = 0.0015
    current_imgsz = 800
    current_freeze = 10
    
    # Pre-defined tuning adjustment roadmap for subsequent rounds
    adjustments = [
        # Round 2 adjustments if Target not reached
        {'lr': 0.0008, 'imgsz': 800, 'freeze': 8,   'epochs': 50}, 
        # Round 3 adjustments
        {'lr': 0.0004, 'imgsz': 960, 'freeze': 5,   'epochs': 50}, 
    ]

    for round_num in range(1, args.max_rounds + 1):
        best_path, map50 = run_training_round(
            checkpoint_path=current_weights,
            round_num=round_num,
            lr=current_lr,
            imgsz=current_imgsz,
            freeze=current_freeze,
            epochs=100 if round_num == 1 else adjustments[round_num-2]['epochs']
        )

        if not best_path:
            logger.error("Auto-tuning failed during training.")
            sys.exit(1)

        current_weights = best_path

        # Check if target reached
        if map50 >= args.target_map:
            logger.info("\n" + "=" * 80)
            logger.info(f"🎯 SUCCESS: Target mAP@50 ({args.target_map*100:.1f}%) reached!")
            logger.info(f"Best Model Path: {current_weights}")
            logger.info("=" * 80)
            break
        else:
            logger.warning(f"\n⚠️ Target not met ({map50*100:.1f}% < {args.target_map*100:.1f}%)")
            
            if round_num < args.max_rounds:
                # Apply next set of hyperparameter adjustments
                next_adj = adjustments[round_num-1]
                current_lr = next_adj['lr']
                current_imgsz = next_adj['imgsz']
                current_freeze = next_adj['freeze']
                logger.info(f"Preparing next round with adjusted hyperparameters...")
            else:
                logger.error(f"\n❌ Reached maximum tuning iterations ({args.max_rounds} rounds) without hitting the target.")
                logger.info(f"Best model obtained: {current_weights} with mAP@50={map50*100:.2f}%")


if __name__ == '__main__':
    main()
