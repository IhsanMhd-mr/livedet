"""
FINE-TUNE TRAINED MODEL
Loads the pre-trained YOLO checkpoint and fine-tunes it on the pre-cleaned dataset.
Assumes data cleaning/conversion has already been completed by `data_cleaning.py`.

Usage:
    python fine_tune.py
    python fine_tune.py --epochs 50 --batch-size 8
"""

import sys
import logging
import argparse
import json
import time
from pathlib import Path
import os

# Disable font downloads
os.environ['YOLOV5_DISABLE_TELEMETRY'] = '1'

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
except ImportError:
    logger.error("ultralytics not installed. Run: pip install ultralytics")
    sys.exit(1)

# Resolve paths
project_root = Path(__file__).parent.parent
data_yaml = project_root / 'data' / 'data.yaml'
default_checkpoint = project_root / 'runs' / 'base_models' / 'pothole_detector_yolo11s' / 'weights' / 'best.pt'


def fine_tune_model(checkpoint_path, epochs, batch_size, device, patience, output_name, lr0=0.001, imgsz=800, freeze=10):
    """
    Fine-tune YOLO model on the pre-cleaned dataset.
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("FINE-TUNE MODEL ON CLEANED DATASET")
    logger.info("=" * 80)

    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        logger.error(f"Checkpoint not found: {checkpoint}")
        logger.error("Available checkpoints:")
        for pt in (project_root / 'runs').rglob('best.pt'):
            logger.error(f"  {pt}")
        sys.exit(1)

    logger.info(f"  Checkpoint:     {checkpoint}")
    logger.info(f"  Dataset config: {data_yaml}")
    logger.info(f"  Epochs:         {epochs}")
    logger.info(f"  Batch size:     {batch_size}")
    logger.info(f"  Device:         {device}")
    logger.info(f"  Patience:       {patience}")
    logger.info(f"  Learning Rate:  {lr0}")
    logger.info(f"  Output folder:  {output_name}")
    logger.info(f"  Image size:     {imgsz}")
    logger.info(f"  Freeze layers:  {freeze}")
    logger.info("")

    logger.info("Loading pre-trained checkpoint...")
    model = YOLO(str(checkpoint))
    logger.info("  Model loaded successfully")
    logger.info("")

    logger.info("Starting training...")
    start_time = time.time()

    try:
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device=device,
            patience=patience,
            save=True,
            project=str(project_root / 'runs' / 'base_models'),
            name=output_name,
            workers=2,
            close_mosaic=5,
            plots=True,
            freeze=freeze,
            # Fine-tuning parameters
            lr0=lr0,
            lrf=0.01,
            warmup_epochs=3,
            cos_lr=True,
            # Advanced native augmentations
            mosaic=1.0,
            mixup=0.15,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
        )

        duration = time.time() - start_time
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"FINE-TUNING COMPLETE ({duration:.0f}s)")
        logger.info("=" * 80)
        output_dir = project_root / 'runs' / 'base_models' / output_name
        logger.info(f"  Results saved to: {output_dir}")
        logger.info(f"  Best weight file: {output_dir / 'weights' / 'best.pt'}")
        logger.info("")

        return results

    except Exception as e:
        logger.error(f"Fine-tuning failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def evaluate_and_compare(output_name):
    """
    Evaluate the fine-tuned model and save metrics comparison.
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("EVALUATING MODEL")
    logger.info("=" * 80)

    v2_weights = project_root / 'runs' / 'base_models' / output_name / 'weights' / 'best.pt'
    if not v2_weights.exists():
        logger.warning(f"Fine-tuned weights file not found: {v2_weights}")
        return

    logger.info("Evaluating model on validation set...")
    model_v2 = YOLO(str(v2_weights))
    metrics_v2 = model_v2.val(
        data=str(data_yaml),
        project=str(project_root / 'runs' / 'evaluation'),
        name='yolo11s_v2_evaluation',
        save=True,
        plots=True,
        device='0',
    )

    # Extract metrics
    v2_metrics = {
        'precision': float(metrics_v2.results_dict.get('metrics/precision(B)', 0)),
        'recall': float(metrics_v2.results_dict.get('metrics/recall(B)', 0)),
        'map50': float(metrics_v2.results_dict.get('metrics/mAP50(B)', 0)),
        'map50_95': float(metrics_v2.results_dict.get('metrics/mAP50-95(B)', 0)),
        'speed_preprocess': float(metrics_v2.speed.get('preprocess', 0)),
        'speed_inference': float(metrics_v2.speed.get('inference', 0)),
        'speed_postprocess': float(metrics_v2.speed.get('postprocess', 0)),
        'file_size_mb': v2_weights.stat().st_size / (1024 * 1024),
    }

    # Load baseline metrics for comparison
    v1_metrics_path = project_root / 'model_comparison_results' / 'exact_metrics.json'
    v1_m = {}
    if v1_metrics_path.exists():
        try:
            with open(v1_metrics_path) as f:
                v1_data = json.load(f)
            v1_m = v1_data.get('YOLO11s', {})
        except Exception as e:
            logger.warning(f"Failed to load baseline metrics: {e}")

    logger.info("")
    logger.info("=" * 70)
    logger.info("MODEL METRICS")
    logger.info("=" * 70)
    logger.info(f"{'Metric':<20} {'v1 (Baseline)':>15} {'v2 (Fine-Tuned)':>17} {'Delta':>10}")
    logger.info("-" * 70)

    comparisons = [
        ('mAP@50',    v1_m.get('map50', 0),    v2_metrics['map50']),
        ('mAP@50-95', v1_m.get('map50_95', 0), v2_metrics['map50_95']),
        ('Precision', v1_m.get('precision', 0), v2_metrics['precision']),
        ('Recall',    v1_m.get('recall', 0),    v2_metrics['recall']),
    ]

    for name, v1, v2 in comparisons:
        delta = v2 - v1
        arrow = '+' if delta >= 0 else ''
        logger.info(f"  {name:<18} {v1*100:>13.2f}% {v2*100:>15.2f}% {arrow}{delta*100:>8.2f}%")

    logger.info("=" * 70)

    # Save metrics JSON
    v2_output = {'YOLO11s_v2_finetuned': v2_metrics}
    v2_metrics_path = project_root / 'model_comparison_results' / 'exact_metrics_v2.json'
    try:
        with open(v2_metrics_path, 'w') as f:
            json.dump(v2_output, f, indent=4)
        logger.info(f"Saved fine-tune metrics to: {v2_metrics_path}")
    except Exception as e:
        logger.error(f"Failed to save metrics JSON: {e}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLO model on pre-cleaned dataset")
    parser.add_argument('--checkpoint', type=str, default=None,
                        help=f'Path to pre-trained checkpoint (default: {default_checkpoint})')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs (default: 150)')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size (default: 8)')
    parser.add_argument('--device', type=str, default='0', choices=['cpu', '0', '1'], help='Device (default: 0)')
    parser.add_argument('--patience', type=int, default=25, help='Patience epochs (default: 25)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--name', type=str, default='pothole_detector_yolo11s_v2', help='Output name')
    parser.add_argument('--imgsz', type=int, default=800, help='Image resolution (default: 800)')
    parser.add_argument('--freeze', type=int, default=10, help='Backbone layers to freeze (default: 10)')
    parser.add_argument('--skip-eval', action='store_true', help='Skip evaluation')

    args = parser.parse_args()
    checkpoint = args.checkpoint or str(default_checkpoint)

    # Train
    results = fine_tune_model(
        checkpoint_path=checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        patience=args.patience,
        output_name=args.name,
        lr0=args.lr,
        imgsz=args.imgsz,
        freeze=args.freeze,
    )

    if results is None:
        logger.error("Fine-tuning failed.")
        sys.exit(1)

    # Evaluate
    if not args.skip_eval:
        evaluate_and_compare(args.name)


if __name__ == '__main__':
    main()
