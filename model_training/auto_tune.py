"""
AUTOMATED HYPERPARAMETER TUNING LOOP
Iteratively trains a YOLO model on the cleaned dataset, checks if validation mAP@50
reaches the target (e.g. 85%), and if not, automatically adjusts hyperparameters
and triggers subsequent training rounds.

This script implements a heuristic-driven parameter search strategy that adjusts:
  1. Learning Rate (LR): Progressively reduced to avoid overshooting local minima.
  2. Image Resolution (imgsz): Increased in later rounds to detect finer details.
  3. Freeze Layers: Gradually unfrozen to allow deeper feature extractor adaptation.

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
    """
    Runs a single fine-tuning round with the specified hyperparameters.
    
    Args:
        checkpoint_path (str): Path to starting weights (.pt file).
        round_num (int): Current tuning iteration index.
        lr (float): Initial learning rate (lr0).
        imgsz (int): Image training and validation resolution.
        freeze (int): Number of backbone layers to freeze from the bottom.
        epochs (int): Number of training epochs for this round.
        
    Returns:
        tuple: (best_weights_path, map50_score) achieved in this round.
    """
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

    # Trigger training using Ultralytics engine with specified hyperparameter overrides
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=8,
        device=0,
        patience=15,  # Shorter patience to prevent redundant epochs in autotuning
        save=True,
        project=str(project_root / 'runs' / 'base_models'),
        name=output_name,
        workers=2,
        close_mosaic=5,
        plots=True,
        freeze=freeze,       # Overriden layer freeze hyperparameter
        
        # Hyperparameters
        lr0=lr,              # Overriden initial learning rate
        lrf=0.01,            # Final learning rate fraction (lrf * lr0 is final LR)
        warmup_epochs=3,     # Gradual warmup period
        cos_lr=True,         # Cosine learning rate decay scheduler
        
        # Native augmentations
        mosaic=1.0,
        mixup=0.15,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
    )

    # Perform formal evaluation at the final target resolution
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

    # Extract mAP@50 metric for check against goal
    map50 = float(metrics.results_dict.get('metrics/mAP50(B)', 0))
    logger.info(f"\nRound {round_num} Finished. Validation mAP@50: {map50:.4f} ({map50*100:.2f}%)")
    return best_weights, map50


def main():
    parser = argparse.ArgumentParser(description="Automated Iterative YOLO Fine-Tuner")
    parser.add_argument('--checkpoint', type=str, default=None, help='Initial checkpoint weights')
    parser.add_argument('--target-map', type=float, default=0.85, help='Target mAP@50 score (e.g. 0.85)')
    parser.add_argument('--max-rounds', type=int, default=3, help='Maximum fine-tuning iterations')
    args = parser.parse_args()

    # Determine starting checkpoint
    initial_checkpoint = args.checkpoint
    if not initial_checkpoint:
        default_pt = project_root / 'runs' / 'base_models' / 'pothole_detector_yolo11s' / 'weights' / 'best.pt'
        if default_pt.exists():
            initial_checkpoint = str(default_pt)
        else:
            initial_checkpoint = 'yolo11s.pt'

    logger.info(f"Starting autotune script. Target: {args.target_map*100:.1f}% mAP@50")

    # Round 1 default parameters
    current_weights = initial_checkpoint
    current_lr = 0.0015
    current_imgsz = 800
    current_freeze = 10
    
    # Pre-defined tuning roadmap. If a round fails to hit the target,
    # the loop moves to the next dictionary to run the next round.
    adjustments = [
        # Round 2: Reduce LR, unfreeze 2 extra layers (freeze 8 instead of 10) to let more parameters adapt.
        {'lr': 0.0008, 'imgsz': 800, 'freeze': 8,   'epochs': 50}, 
        # Round 3: Further reduce LR, increase resolution to 960 to capture micro-details, freeze only 5 layers.
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

        # Success condition check
        if map50 >= args.target_map:
            logger.info("\n" + "=" * 80)
            logger.info(f"🎯 SUCCESS: Target mAP@50 ({args.target_map*100:.1f}%) reached!")
            logger.info(f"Best Model Path: {current_weights}")
            logger.info("=" * 80)
            break
        else:
            logger.warning(f"\n⚠️ Target not met ({map50*100:.1f}% < {args.target_map*100:.1f}%)")
            
            # If we haven't exceeded maximum rounds, load parameters for the next round
            if round_num < args.max_rounds:
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
