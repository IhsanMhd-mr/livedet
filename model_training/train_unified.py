"""
UNIFIED TRAINING SCRIPT - Base Training for Pothole Detection
Supports: YOLOv8, YOLOv10, YOLO11, Custom Training
Features: Advanced data augmentation, Multiple model variants, Configuration options

TWO-STAGE TRAINING PARADIGM:
  We split the training workflow into two distinct stages to balance general visual
  generalization with target domain specialization.

  Stage 1 (BASE TRAINING) - python train_unified.py:
    Goal: Trains a model initialized with pre-trained COCO weights on a broad, general 
          pothole dataset. Helps the model learn generic visual features of potholes 
          (edges, shapes, shadows) under diverse perspectives.
    Key Hyperparameters (Default):
      - Epochs: 100
      - Image Size (imgsz): 640 (standard training size)
      - Batch Size: 16
      - Early Stopping Patience: 20
      - Mosaic Augmentation Closing: Last 5 epochs (close_mosaic=5)
      - Augmentations: Custom imgaug pipeline (Scale, Rotate, Fliplr, Shear, Blur)

  Stage 2 (FINE-TUNING) - python fine_tune.py:
    Goal: Loads the Stage 1 best.pt checkpoint and fine-tunes it on the specialized, 
          project-specific annotated dataset to adapt it to the target system's deployment environment.
    Key Hyperparameters (Default):
      - Checkpoint: best.pt from Stage 1
      - Epochs: 150
      - Image Size (imgsz): 800 (higher resolution for fine-detail detection)
      - Batch Size: 8 (reduced to accommodate higher resolution VRAM footprint)
      - Early Stopping Patience: 25
      - Backbone Freezing: First 10 layers (freezes early feature extraction layers to preserve general knowledge)
      - Learning Rate (lr0): 0.001 (with Cosine Learning Rate Scheduler enabled)
      - Native Augmentations: mosaic=1.0, mixup=0.15, HSV tuning

  Analogy:
    Base training teaches the model what a pothole looks like in the general world;
    fine-tuning teaches it how to detect the specific potholes seen by our cameras.

Usage:
    python train_unified.py                                    # Default: YOLOv10m with augmentation
    python train_unified.py --model yolov8s --epochs 100        # YOLOv8s
    python train_unified.py --model yolov11m --epochs 100       # YOLOv11m
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
    """
    Setup imgaug augmentation pipeline with random transformations.
    
    Data augmentation is crucial for small/medium datasets. It introduces artificial 
    variance, helping the model generalize to different conditions (e.g., lighting, 
    perspectives, cameras, blur) and reduces overfitting.
    
    Returns:
        iaa.SomeOf: A pipeline that applies a subset of transformations, or None if imgaug is unavailable.
    """
    if not IMGAUG_AVAILABLE:
        logger.warning("imgaug not available, augmentation disabled")
        return None
    
    # iaa.SomeOf(2, [...]) randomly selects exactly 2 of the 5 defined transformations
    # for each image passed through the pipeline, ensuring a variety of combinations.
    augmentor = iaa.SomeOf(2, [    
        iaa.Affine(scale=(0.8, 1.2)),           # Zoom in/out: Scales image by 80% to 120%
        iaa.Affine(rotate=(-15, 15)),           # Rotation: Rotates image by -15 to +15 degrees
        iaa.Fliplr(1),                          # Horizontal Flip: Flips image horizontally (mirror effect)
        iaa.Affine(shear=(-5, 5)),              # Shear: Tilts the image by -5 to +5 degrees
        iaa.GaussianBlur(sigma=(1.0, 3.0)),     # Gaussian Blur: Simulates camera out-of-focus or motion blur
    ])
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("DATA AUGMENTATION PIPELINE INITIALIZED")
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
    """
    Analyze bounding box dimensions to understand the spatial distribution of targets.
    
    This computes and prints percentiles of 'y_max' (the bottom edge of each bounding box).
    Knowing where objects are situated vertically in the frame is useful because potholes 
    are typically located on the road surface, which corresponds to the lower region of images.
    
    Args:
        annotation_df (pd.DataFrame): DataFrame containing annotations with columns 'y' and 'h'.
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("BOUNDING BOX ANALYSIS")
    logger.info("=" * 80)
    
    # Calculate bottom edge of bounding boxes (y + height)
    y_max = annotation_df['y'].values + annotation_df['h'].values
    y_max = np.sort(y_max, axis=None)
    
    logger.info("\nBounding Box Height (y_max) Distribution:")
    logger.info("  Percentile | Value (Pixels)")
    logger.info("  " + "-" * 25)
    
    # Print deciles (10% increments) to see the overall distribution
    for i in range(0, 101, 10):
        idx = int(len(y_max) * (float(i) / 100))
        idx = min(idx, len(y_max) - 1)
        val = y_max[idx]
        logger.info(f"  {i:3d}%      | {val:.1f}")
    
    # Print detailed percentile distribution from 90% to 100% 
    # to understand where the largest/lowest potholes reside.
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
    Augment a single image and adjust its corresponding bounding boxes.
    
    When an image undergoes spatial transformation (rotation, scaling, etc.), the bounding
    box coordinates must be recalculated to track the target's new position.
    
    Args:
        annot_df (pd.DataFrame): DataFrame containing all bounding box annotations.
        path (str): File path of the image to augment.
        augmentor (iaa.Augmenter): Instantiated imgaug pipeline.
        img_id (str): The identifier of the target image in the annotations.
        suffix (str/int): Unique suffix for the output filename to avoid overwriting.
    
    Returns:
        pd.DataFrame: DataFrame containing the updated coordinates of the augmented bounding boxes.
    """
    # 1. Retrieve existing annotations for this image
    bbox_coords = annot_df[annot_df['image_id'] == img_id]
    
    if bbox_coords.empty:
        logger.debug(f"No bboxes for {img_id}")
        return pd.DataFrame(columns=['image_id', 'x', 'y', 'x_max', 'y_max'])
    
    # 2. Extract bounding box coordinates as [x_min, y_min, x_max, y_max]
    bb_array = bbox_coords.loc[:, ['x', 'y', 'x_max', 'y_max']].values
    
    # 3. Read image from disk
    try:
        image = imageio.imread(path)
    except Exception as e:
        logger.error(f"Error reading {path}: {e}")
        return pd.DataFrame(columns=['image_id', 'x', 'y', 'x_max', 'y_max'])
    
    # 4. Wrap bounding boxes in imgaug's custom structure
    bbs = BoundingBoxesOnImage.from_xyxy_array(bb_array, shape=image.shape)
    
    # 5. Apply the augmentation pipeline to both image and bounding boxes
    # imgaug updates the bounding boxes coordinates along with the pixels
    image_aug, bbs_aug = augmentor(image=image, bounding_boxes=bbs)
    
    # 6. Post-process bounding boxes:
    # - remove_out_of_image(): Drops any box that is shifted completely outside the canvas.
    # - clip_out_of_image(): Clips boxes that are partially outside back to the image edges.
    bbs_aug = bbs_aug.remove_out_of_image()
    bbs_aug = bbs_aug.clip_out_of_image()
    
    # 7. Save the augmented image to the same directory
    output_dir = Path(path).parent
    output_path = output_dir / f"{img_id}_{suffix}.JPG"
    
    try:
        imageio.imwrite(str(output_path), image_aug)
    except Exception as e:
        logger.error(f"Error saving {output_path}: {e}")
        return pd.DataFrame(columns=['image_id', 'x', 'y', 'x_max', 'y_max'])
    
    # 8. Extract the updated bounding box coordinates as an array
    bbs_array = bbs_aug.to_xyxy_array()
    
    if len(bbs_array) == 0:
        logger.debug(f"No bboxes after augmentation for {img_id}")
        return pd.DataFrame(columns=['image_id', 'x', 'y', 'x_max', 'y_max'])
    
    # 9. Format output DataFrame with the new image ID (original + suffix) and coordinates
    img_id_array = np.empty([bbs_array.shape[0], 1])
    final = np.concatenate((img_id_array, bbs_array), axis=1)
    df = pd.DataFrame(final, columns=['image_id', 'x', 'y', 'x_max', 'y_max'])
    df['image_id'] = f'{img_id}_{suffix}'
    
    return df


def augment_data(train_img_df, annot_df, augmentor, num_samples):
    """
    Augment multiple randomly-sampled positive images from the training dataset.
    
    Args:
        train_img_df (pd.DataFrame): DataFrame listing training images and their paths.
        annot_df (pd.DataFrame): DataFrame containing annotations.
        augmentor (iaa.Augmenter): Instantiated imgaug pipeline.
        num_samples (int): Number of images to augment.
    
    Returns:
        pd.DataFrame: DataFrame containing all generated bounding box coordinates.
    """
    logger.info(f"Starting augmentation of {num_samples} images...")
    
    final_df = pd.DataFrame(columns=['image_id', 'x', 'y', 'x_max', 'y_max'])
    
    # Only augment positive images (those containing potholes).
    # Augmenting negative/background-only images is not useful for object detection.
    positive_images = train_img_df[
        train_img_df['path'].str.contains('positive', case=False)
    ]
    
    if len(positive_images) == 0:
        logger.warning("No positive images found in dataset")
        return final_df
    
    logger.info(f"Found {len(positive_images)} positive images to sample from")
    
    # Loop to generate the requested number of augmented samples
    for i in range(num_samples):
        # Randomly select a positive image indices
        idx = np.random.randint(len(positive_images), size=1)[0]
        row = positive_images.iloc[idx, :]
        path = row['path']
        img_id = Path(path).stem
        
        # Apply the single-image augmentation pipeline
        df = augment_img_bbox(annot_df, path, augmentor, img_id, i)
        
        # Concatenate successful augmentations to the final DataFrame
        if not df.empty:
            final_df = pd.concat([final_df, df], ignore_index=True)
            
            # Print a progress indicator at 10% increments
            if (i + 1) % max(1, num_samples // 10) == 0:
                logger.info(f"  Progress: {i + 1}/{num_samples} images augmented")
    
    logger.info(f"✓ Generated {len(final_df)} augmented bounding boxes")
    return final_df


def load_and_prepare_data(augment=True, num_augment=100):
    """
    Load annotations, perform validation steps, and run data augmentation if requested.
    
    This prepares the raw CSV dataset into a form compatible with augmentation and returns
    the final combined dataset.
    
    Args:
        augment (bool): If True, run data augmentation.
        num_augment (int): Number of images to augment.
        
    Returns:
        pd.DataFrame: Cleaned and expanded DataFrame of target bounding boxes.
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("LOADING & PREPARING TRAINING DATA")
    logger.info("=" * 80)
    
    # 1. Load annotation CSV
    annotation_csv = data_dir / 'train_df.csv'
    if not annotation_csv.exists():
        logger.warning(f"Annotation file not found: {annotation_csv}")
        logger.warning("Skipping augmentation, proceeding with direct training...")
        return None
    
    annotation_df = pd.read_csv(str(annotation_csv))
    
    # 2. De-duplicate annotations to remove identical overlapping coordinates
    original_size = len(annotation_df)
    annotation_df.drop_duplicates(keep='first', inplace=True)
    annotation_df.reset_index(inplace=True, drop=True)
    
    duplicates_removed = original_size - len(annotation_df)
    if duplicates_removed > 0:
        logger.info(f"✓ Removed {duplicates_removed} duplicate annotations")
    
    # 3. Add explicit x_max and y_max columns (required by imgaug package)
    annotation_df['x_max'] = annotation_df['x'] + annotation_df['w']
    annotation_df['y_max'] = annotation_df['y'] + annotation_df['h']
    
    logger.info(f"✓ Loaded {len(annotation_df)} annotations")
    
    # 4. Run statistical analysis on bounding box coordinates
    analyze_annotation_bbox(annotation_df)
    
    # 5. Perform data augmentation if enabled
    if augment and IMGAUG_AVAILABLE:
        logger.info(f"Augmentation enabled: augmenting {num_augment} images...")
        
        # Load the CSV file containing paths to images
        train_img_csv = data_dir / 'train_images.csv'
        
        if Path(train_img_csv).exists():
            train_img_df = pd.read_csv(str(train_img_csv))
            logger.info(f"✓ Loaded {len(train_img_df)} training image paths")
            
            # Setup augmentor pipeline
            augmentor = setup_augmentor()
            
            # Generate augmented images & boxes
            augmented_df = augment_data(train_img_df, annotation_df, augmentor, num_augment)
            
            if not augmented_df.empty:
                # Merge augmented samples with the original dataset
                annotation_df = pd.concat([annotation_df, augmented_df], ignore_index=True)
                
                # Save the new bounding boxes to disk for reference/verification
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
# MODEL TRAINING FUNCTIONS - BASE TRAINING (From Scratch / Pre-trained COCO)
# ═════════════════════════════════════════════════════════════════════════════════

def train_yolo_base(model_name, epochs, batch_size, device, patience):
    """
    Train any supported YOLO model (YOLOv8, YOLOv10, YOLO11) from scratch/pretrained COCO checkpoint.
    
    This function handles name mapping, loads weights, and runs the training loop using 
    standard hyperparameters.
    
    Args:
        model_name (str): Name of the model to train (e.g., 'yolov8s', 'yolo11s').
        epochs (int): Max number of complete training passes over the dataset.
        batch_size (int): Batch size (images per optimization step).
        device (str): Computation device ('cpu', 'gpu', or specific GPU index like '0').
        patience (int): Early stopping patience (stop training if no improvement for N epochs).
        
    Returns:
        ultralytics.utils.metrics.DetMetrics: Training results/metrics object, or None if failed.
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"BASE TRAINING: {model_name.upper()} (From Scratch / Pre-trained COCO)")
    logger.info("=" * 80)
    logger.info(f"Configuration:")
    logger.info(f"  Model:      {model_name}")
    logger.info(f"  Epochs:     {epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Device:     {device}")
    logger.info(f"  Patience:   {patience}")
    logger.info(f"  Dataset:    {data_yaml}")
    logger.info("")
    
    # Map the model_name to the correct weights filename.
    # Ultralytics naming convention: YOLO11 weights are yolo11*.pt (no 'v').
    # YOLOv8 and YOLOv10 weights are yolov8*.pt and yolov10*.pt respectively.
    weights_name = model_name
    if weights_name.startswith('yolov11'):
        weights_name = weights_name.replace('yolov11', 'yolo11')
    
    # Append .pt if not present
    if not weights_name.endswith('.pt'):
        weights_name = f"{weights_name}.pt"
        
    logger.info(f"Loading {weights_name} weights...")
    try:
        model = YOLO(weights_name)
        logger.info("✓ Model loaded successfully")
        logger.info("")
    except Exception as e:
        logger.error(f"Failed to load model weights '{weights_name}': {e}")
        return None
    
    logger.info("Starting training...")
    try:
        results = model.train(
            data=str(data_yaml),            # Path to dataset configuration YAML
            epochs=epochs,                  # Maximum number of epochs
            imgsz=640,                      # Resolution to resize input images to
            batch=batch_size,               # Number of images per batch
            device=device,                  # GPU or CPU
            patience=patience,              # Epoch count to wait before early stopping
            save=True,                      # Save checkpoint weights and results
            project=str(project_root / 'runs' / 'base_models'),  # Directory to save runs
            name=f'pothole_detector_{model_name}',               # Run directory name
            workers=int(os.getenv('NUM_WORKERS', 2)),           # CPU workers for data loading
            
            # close_mosaic=5 disables mosaic augmentation in the final 5 epochs.
            # Mosaic augmentation is excellent for general context, but can introduce
            # artificial boundaries that degrade precise bbox coordinates. Disabling it
            # at the end allows the model to refine bounding box alignment.
            close_mosaic=5,                 
            plots=True,                     # Enable generating plots during training to track progress
        )
        
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"✓ {model_name.upper()} BASE TRAINING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Results: runs/base_models/pothole_detector_{model_name}/")
        logger.info(f"Best model: runs/base_models/pothole_detector_{model_name}/weights/best.pt")
        logger.info(f"\nNext: Use fine_tune.py to fine-tune on annotated dataset")
        
        return results
        
    except Exception as e:
        logger.error(f"Training failed for {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


# ═════════════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSER
# ═════════════════════════════════════════════════════════════════════════════════

def parse_arguments():
    """
    Parse command-line arguments to configure training runs.
    
    Returns:
        argparse.Namespace: Object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Base training script with data augmentation for pothole detection (from scratch)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_unified.py                                    # YOLOv10m + augmentation (default)
  python train_unified.py --model yolov8s --epochs 100        # YOLOv8s
  python train_unified.py --model yolov11s --epochs 100       # YOLOv11s
  python train_unified.py --augment --num-augment 200        # More augmentation
  python train_unified.py --no-augment                       # Without augmentation

NOTE: For fine-tuning on annotated data, use: python fine_tune.py
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='yolov10m',
        choices=[
            'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x',
            'yolov10n', 'yolov10s', 'yolov10m', 'yolov10l', 'yolov10x',
            'yolov11n', 'yolov11s', 'yolov11m', 'yolov11l', 'yolov11x',
            'yolo11n', 'yolo11s', 'yolo11m', 'yolo11l', 'yolo11x'
        ],
        help='Model architecture variant to train (default: yolov10m)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Maximum number of training epochs (default: 100)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size (images per step; default: 16)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='gpu',
        choices=['cpu', 'gpu', '0', '1'],
        help='Execution device. Use GPU for training acceleration (default: gpu)'
    )
    
    parser.add_argument(
        '--patience',
        type=int,
        default=20,
        help='Early stopping patience count in epochs (default: 20)'
    )
    
    parser.add_argument(
        '--augment',
        action='store_true',
        default=True,
        help='Enable random transformations (enabled by default)'
    )
    
    parser.add_argument(
        '--no-augment',
        action='store_false',
        dest='augment',
        help='Disable data augmentation transformations'
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
    """
    Main orchestrator for the base model training.
    
    Validates files, processes dataset details, triggers augmentation if active, 
    and launches the selected YOLO base training routine.
    """
    
    # 1. Verify existence of the YOLO dataset definition YAML
    if not data_yaml.exists():
        logger.error(f"data.yaml not found: {data_yaml}")
        sys.exit(1)
    
    # 2. Parse execution flags
    args = parse_arguments()
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("BASE TRAINING - POTHOLE DETECTION (From Scratch)")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model}")
    logger.info(f"Data: {data_yaml}")
    logger.info(f"Augmentation: {'✓ Enabled' if args.augment else '✗ Disabled'}")
    logger.info("=" * 80)
    
    # 3. Load input datasets, perform cleaning, and run augmentation if active
    load_and_prepare_data(augment=args.augment, num_augment=args.num_augment)
    
    # 4. Train the selected YOLO model version using the unified training function
    train_yolo_base(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        patience=args.patience
    )
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("✓ Base training script completed!")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Verify model performance at: runs/base_models/pothole_detector_yolov* or yolo11*")
    logger.info("  2. For fine-tuning on annotated data, run: python fine_tune.py")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
