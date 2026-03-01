"""
UNIFIED TRAINING SCRIPT - Base Training for Pothole Detection
Supports: YOLOv10m, YOLOv11m, Custom Training
Features: Advanced data augmentation, Multiple model variants, Configuration options

TRAINING WORKFLOW:
  Stage 1 (BASE TRAINING):  python train_unified.py          → Trains model from scratch
  Stage 2 (FINE-TUNING):    python fine_tune.py               → Fine-tunes on annotated dataset

Usage:
    python train_unified.py                                    # Default: YOLOv10m with augmentation
    python train_unified.py --model yolov11m --epochs 100      # YOLOv11m
    python train_unified.py --model yolov10m --no-augment      # Without augmentation
    python train_unified.py --model yolov11m --batch-size 32 --device gpu
"""

import sys
import logging
import argparse
from pathlib import Path
import os
import pandas as pd
import numpy as np

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

try:
    from imgaug import augmenters as iaa
    from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
    import imageio
    IMGAUG_AVAILABLE = True
except ImportError:
    IMGAUG_AVAILABLE = False
    logger.warning("imgaug not available - augmentation disabled (install: pip install imgaug imageio)")

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
# DATA AUGMENTATION SETUP
# ═════════════════════════════════════════════════════════════════════════════════

def setup_augmentor():
    """Setup imgaug augmentation pipeline with random transformations"""
    if not IMGAUG_AVAILABLE:
        logger.warning("imgaug not available, augmentation disabled")
        return None
    
    # Randomly applies 2 of 5 augmentation techniques to each image
    augmentor = iaa.SomeOf(2, [    
        iaa.Affine(scale=(0.8, 1.2)),           # Scale: 80-120%
        iaa.Affine(rotate=(-15, 15)),           # Rotate: ±15 degrees
        iaa.Fliplr(1),                          # Horizontal flip
        iaa.Affine(shear=(-5, 5)),              # Shear: ±5 degrees
        iaa.GaussianBlur(sigma=(1.0, 3.0)),     # Blur: sigma 1.0-3.0
    ])
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("DATA AUGMENTATION PIPELINE")
    logger.info("=" * 80)
    logger.info("✓ Augmentation enabled")
    logger.info("✓ Randomly applies 2 of 5 techniques per image:")
    logger.info("  1. Scale: 0.8x to 1.2x")
    logger.info("  2. Rotation: ±15 degrees")
    logger.info("  3. Horizontal flip")
    logger.info("  4. Shear: ±5 degrees")
    logger.info("  5. Gaussian blur: sigma 1.0-3.0")
    logger.info("=" * 80)
    logger.info("")
    
    return augmentor


def analyze_annotation_bbox(annotation_df):
    """Analyze bounding box dimensions to understand data distribution"""
    logger.info("")
    logger.info("=" * 80)
    logger.info("BOUNDING BOX ANALYSIS")
    logger.info("=" * 80)
    
    y_max = annotation_df['y'].values + annotation_df['h'].values
    y_max = np.sort(y_max, axis=None)
    
    logger.info("\nBounding Box Height (y_max) Distribution:")
    logger.info("  Percentile | Value")
    logger.info("  " + "-" * 25)
    
    # 0-100 by 10
    for i in range(0, 101, 10):
        idx = int(len(y_max) * (float(i) / 100))
        idx = min(idx, len(y_max) - 1)
        val = y_max[idx]
        logger.info(f"  {i:3d}%      | {val:.1f}")
    
    # 90-100 by 1 (fine-grained)
    logger.info("\n  Fine-grained analysis (90-100%):")
    for i in range(90, 101, 1):
        idx = int(len(y_max) * (float(i) / 100))
        idx = min(idx, len(y_max) - 1)
        val = y_max[idx]
        logger.info(f"  {i:3d}%      | {val:.1f}")
    
    logger.info("=" * 80)
    logger.info("")


def augment_img_bbox(annot_df, path, augmentor, img_id, suffix):
    """
    Augment single image and corresponding bounding boxes
    
    Args:
        annot_df: DataFrame with bounding box coordinates
        path: Image file path
        augmentor: imgaug augmentation pipeline
        img_id: Image ID for lookup
        suffix: Suffix for augmented image filename
    
    Returns:
        DataFrame with augmented BBox coordinates
    """
    bbox_coords = annot_df[annot_df['image_id'] == img_id]
    
    if bbox_coords.empty:
        logger.debug(f"No bboxes for {img_id}")
        return pd.DataFrame(columns=['image_id', 'x', 'y', 'x_max', 'y_max'])
    
    bb_array = bbox_coords.loc[:, ['x', 'y', 'x_max', 'y_max']].values
    
    try:
        image = imageio.imread(path)
    except Exception as e:
        logger.error(f"Error reading {path}: {e}")
        return pd.DataFrame(columns=['image_id', 'x', 'y', 'x_max', 'y_max'])
    
    # Apply augmentation
    bbs = BoundingBoxesOnImage.from_xyxy_array(bb_array, shape=image.shape)
    image_aug, bbs_aug = augmentor(image=image, bounding_boxes=bbs)
    
    # Handle bboxes outside image after augmentation
    bbs_aug = bbs_aug.remove_out_of_image()
    bbs_aug = bbs_aug.clip_out_of_image()
    
    # Save augmented image
    output_dir = Path(path).parent
    output_path = output_dir / f"{img_id}_{suffix}.JPG"
    
    try:
        imageio.imwrite(str(output_path), image_aug)
    except Exception as e:
        logger.error(f"Error saving {output_path}: {e}")
        return pd.DataFrame(columns=['image_id', 'x', 'y', 'x_max', 'y_max'])
    
    # Convert augmented bboxes to array
    bbs_array = bbs_aug.to_xyxy_array()
    
    if len(bbs_array) == 0:
        logger.debug(f"No bboxes after augmentation for {img_id}")
        return pd.DataFrame(columns=['image_id', 'x', 'y', 'x_max', 'y_max'])
    
    # Create DataFrame for augmented annotations
    img_id_array = np.empty([bbs_array.shape[0], 1])
    final = np.concatenate((img_id_array, bbs_array), axis=1)
    df = pd.DataFrame(final, columns=['image_id', 'x', 'y', 'x_max', 'y_max'])
    df['image_id'] = f'{img_id}_{suffix}'
    
    return df


def augment_data(train_img_df, annot_df, augmentor, num_samples):
    """
    Augment multiple images from dataset
    
    Args:
        train_img_df: DataFrame with image paths
        annot_df: DataFrame with bounding box coordinates
        augmentor: imgaug augmentation pipeline
        num_samples: Number of images to augment
    
    Returns:
        DataFrame with all augmented BBox coordinates
    """
    logger.info(f"Starting augmentation of {num_samples} images...")
    
    final_df = pd.DataFrame(columns=['image_id', 'x', 'y', 'x_max', 'y_max'])
    
    # Filter positive images (with potholes)
    positive_images = train_img_df[
        train_img_df['path'].str.contains('positive', case=False)
    ]
    
    if len(positive_images) == 0:
        logger.warning("No positive images found in dataset")
        return final_df
    
    logger.info(f"Found {len(positive_images)} positive images to sample from")
    
    for i in range(num_samples):
        # Randomly select a positive image
        idx = np.random.randint(len(positive_images), size=1)[0]
        row = positive_images.iloc[idx, :]
        path = row['path']
        img_id = Path(path).stem
        
        # Augment image and bboxes
        df = augment_img_bbox(annot_df, path, augmentor, img_id, i)
        
        if not df.empty:
            final_df = pd.concat([final_df, df], ignore_index=True)
            
            # Progress update every 10%
            if (i + 1) % max(1, num_samples // 10) == 0:
                logger.info(f"  Progress: {i + 1}/{num_samples} images augmented")
    
    logger.info(f"✓ Generated {len(final_df)} augmented bounding boxes")
    return final_df


def load_and_prepare_data(augment=True, num_augment=100):
    """
    Load training data with optional augmentation
    
    Args:
        augment: Whether to perform augmentation
        num_augment: Number of images to augment
    
    Returns:
        DataFrame with annotations (original + augmented)
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("LOADING & PREPARING TRAINING DATA")
    logger.info("=" * 80)
    
    # Load annotation data
    annotation_csv = data_dir / 'train_df.csv'
    if not annotation_csv.exists():
        logger.warning(f"Annotation file not found: {annotation_csv}")
        logger.warning("Skipping augmentation, proceeding with direct training...")
        return None
    
    annotation_df = pd.read_csv(str(annotation_csv))
    
    # Remove duplicates
    original_size = len(annotation_df)
    annotation_df.drop_duplicates(keep='first', inplace=True)
    annotation_df.reset_index(inplace=True, drop=True)
    
    duplicates_removed = original_size - len(annotation_df)
    if duplicates_removed > 0:
        logger.info(f"✓ Removed {duplicates_removed} duplicate annotations")
    
    # Add x_max and y_max columns (right and bottom edges)
    annotation_df['x_max'] = annotation_df['x'] + annotation_df['w']
    annotation_df['y_max'] = annotation_df['y'] + annotation_df['h']
    
    logger.info(f"✓ Loaded {len(annotation_df)} annotations")
    
    # Analyze bbox dimensions
    analyze_annotation_bbox(annotation_df)
    
    # Perform augmentation if enabled
    if augment and IMGAUG_AVAILABLE:
        logger.info(f"Augmentation enabled: augmenting {num_augment} images...")
        
        # Load training image paths
        train_img_csv = data_dir / 'train_images.csv'
        
        if Path(train_img_csv).exists():
            train_img_df = pd.read_csv(str(train_img_csv))
            logger.info(f"✓ Loaded {len(train_img_df)} training image paths")
            
            # Setup augmentor
            augmentor = setup_augmentor()
            
            # Perform augmentation
            augmented_df = augment_data(train_img_df, annotation_df, augmentor, num_augment)
            
            if not augmented_df.empty:
                # Merge augmented data with original
                annotation_df = pd.concat([annotation_df, augmented_df], ignore_index=True)
                
                # Save augmented annotations
                augmented_output = data_dir / 'augmented_annotations.csv'
                augmented_df.to_csv(str(augmented_output), index=False)
                logger.info(f"✓ Saved augmented annotations: {augmented_output}")
                
                logger.info(f"✓ Total annotations after augmentation: {len(annotation_df)}")
                logger.info(f"  - Original: {original_size - duplicates_removed}")
                logger.info(f"  - Augmented: {len(augmented_df)}")
                logger.info(f"  - Increase: {(len(augmented_df)/(original_size - duplicates_removed))*100:.1f}%")
        else:
            logger.warning(f"train_images.csv not found: {train_img_csv}")
            logger.warning("Skipping augmentation")
    elif augment and not IMGAUG_AVAILABLE:
        logger.warning("Augmentation requested but imgaug not available")
        logger.warning("Run: pip install imgaug imageio")
    
    logger.info("=" * 80)
    logger.info("")
    
    return annotation_df


# ═════════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING FUNCTIONS - BASE TRAINING (From Scratch)
# ═════════════════════════════════════════════════════════════════════════════════

def train_yolov10m_base(epochs, batch_size, device, patience):
    """Train YOLOv10m model from scratch (Base Training)
    
    This is for initial model training from pretrained weights.
    For fine-tuning on annotated data, use fine_tune.py instead.
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("BASE TRAINING: YOLOv10m (From Scratch)")
    logger.info("=" * 80)
    logger.info(f"Configuration:")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Device: {device}")
    logger.info(f"  Patience: {patience}")
    logger.info(f"  Dataset: {data_yaml}")
    logger.info("")
    
    logger.info("Loading YOLOv10m model...")
    model = YOLO('yolov10m.pt')
    logger.info("✓ Model loaded")
    logger.info("")
    
    logger.info("Starting training...")
    try:
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=640,
            batch=batch_size,
            device=device,
            patience=patience,
            save=True,
            project=str(project_root / 'runs' / 'base_models'),
            name='pothole_detector_yolov10m',
            workers=config.NUM_WORKERS,
            close_mosaic=5,
            plots=False,
        )
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("✓ YOLOv10m BASE TRAINING COMPLETE")
        logger.info("="*80)
        logger.info(f"Results: runs/base_models/pothole_detector_yolov10m/")
        logger.info(f"Best model: runs/base_models/pothole_detector_yolov10m/weights/best.pt")
        logger.info(f"\nNext: Use fine_tune.py to fine-tune on annotated dataset")
        
        return results
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def train_yolov11m_base(epochs, batch_size, device, patience, model_size='m'):
    """Train YOLOv11 model from scratch (Base Training)
    
    This is for initial model training from pretrained weights.
    For fine-tuning on annotated data, use fine_tune.py instead.
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"BASE TRAINING: YOLOv11{model_size} (From Scratch)")
    logger.info("=" * 80)
    logger.info(f"Configuration:")
    logger.info(f"  Model: YOLOv11{model_size}")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Device: {device}")
    logger.info(f"  Patience: {patience}")
    logger.info(f"  Dataset: {data_yaml}")
    logger.info("")
    
    model_name = f'yolov11{model_size}'
    logger.info(f"Loading {model_name} model...")
    model = YOLO(f'{model_name}.pt')
    logger.info("✓ Model loaded")
    logger.info("")
    
    logger.info("Starting training...")
    try:
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=640,
            batch=batch_size,
            device=device,
            patience=patience,
            save=True,
            project=str(project_root / 'runs' / 'base_models'),
            name=f'pothole_detector_yolov11{model_size}',
            workers=config.NUM_WORKERS,
            close_mosaic=5,
            plots=False,
        )
        
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"✓ YOLOv11{model_size} BASE TRAINING COMPLETE")
        logger.info("="*80)
        logger.info(f"Results: runs/base_models/pothole_detector_yolov11{model_size}/")
        logger.info(f"Best model: runs/base_models/pothole_detector_yolov11{model_size}/weights/best.pt")
        logger.info(f"\nNext: Use fine_tune.py to fine-tune on annotated dataset")
        
        return results
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ═════════════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSER
# ═════════════════════════════════════════════════════════════════════════════════

def parse_arguments():
    """Parse command-line arguments for BASE TRAINING"""
    parser = argparse.ArgumentParser(
        description="Base training script with data augmentation for pothole detection (from scratch)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_unified.py                                    # YOLOv10m + augmentation (default)
  python train_unified.py --model yolov11m --epochs 100      # YOLOv11m
  python train_unified.py --augment --num-augment 200        # More augmentation
  python train_unified.py --no-augment                       # Without augmentation

NOTE: For fine-tuning on annotated data, use: python fine_tune.py
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='yolov10m',
        choices=['yolov10m', 'yolov11n', 'yolov11s', 'yolov11m', 'yolov11l', 'yolov11x'],
        help='Model to train (default: yolov10m)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs (default: 100)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size for training (default: 16)'
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
        '--augment',
        action='store_true',
        default=True,
        help='Enable data augmentation (enabled by default)'
    )
    
    parser.add_argument(
        '--no-augment',
        action='store_false',
        dest='augment',
        help='Disable data augmentation'
    )
    
    parser.add_argument(
        '--num-augment',
        type=int,
        default=100,
        help='Number of images to augment (default: 100)'
    )
    
    return parser.parse_args()


# ═════════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════════

def main():
    """Main training function"""
    
    # Verify data.yaml exists
    if not data_yaml.exists():
        logger.error(f"data.yaml not found: {data_yaml}")
        sys.exit(1)
    
    # Parse arguments
    args = parse_arguments()
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("BASE TRAINING - POTHOLE DETECTION (From Scratch)")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model}")
    logger.info(f"Data: {data_yaml}")
    logger.info(f"Augmentation: {'✓ Enabled' if args.augment else '✗ Disabled'}")
    logger.info("=" * 80)
    
    # Load and prepare data
    load_and_prepare_data(augment=args.augment, num_augment=args.num_augment)
    
    # Train model from scratch
    if args.model == 'yolov10m':
        train_yolov10m_base(args.epochs, args.batch_size, args.device, args.patience)
    elif args.model.startswith('yolov11'):
        model_size = args.model.replace('yolov11', '')
        train_yolov11m_base(args.epochs, args.batch_size, args.device, args.patience, model_size)
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("✓ Base training script completed!")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Verify model performance at: runs/base_models/pothole_detector_yolov*")
    logger.info("  2. For fine-tuning on annotated data, run: python fine_tune.py")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
