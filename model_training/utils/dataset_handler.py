"""
Dataset Handler Module
Prepares datasets in YOLO format for training
"""

import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


class DatasetHandler:
    """
    Manages dataset organization and YOLO format conversion.
    Creates train/val/test splits and YOLO-compatible directory structure.
    """
    
    def __init__(self, dataset_root, output_root):
        """
        Initialize dataset handler.
        
        Args:
            dataset_root: Root directory containing raw images/labels
            output_root: Output directory for YOLO-formatted dataset
        """
        self.dataset_root = Path(dataset_root)
        self.output_root = Path(output_root)
        
        self.images_dir = self.output_root / 'images'
        self.labels_dir = self.output_root / 'labels'
        
        logger.info(f"DatasetHandler initialized: {dataset_root} -> {output_root}")
    
    def create_yolo_structure(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """
        Create YOLO directory structure with train/val/test splits.
        
        Args:
            train_ratio: Fraction for training set (default: 0.7)
            val_ratio: Fraction for validation set (default: 0.15)
            test_ratio: Fraction for test set (default: 0.15)
            
        Returns:
            Dict with created paths
        """
        assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1.0"
        
        # Create directory structure
        splits = ['train', 'val', 'test']
        for split in splits:
            (self.images_dir / split).mkdir(parents=True, exist_ok=True)
            (self.labels_dir / split).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Created YOLO structure at {self.output_root}")
        
        return {
            'images_train': self.images_dir / 'train',
            'images_val': self.images_dir / 'val',
            'images_test': self.images_dir / 'test',
            'labels_train': self.labels_dir / 'train',
            'labels_val': self.labels_dir / 'val',
            'labels_test': self.labels_dir / 'test',
        }
    
    def create_data_yaml(self, num_classes=1, class_names=None):
        """
        Create YOLO data.yaml configuration file.
        
        Args:
            num_classes: Number of object classes (default: 1 for pothole)
            class_names: List of class names (default: ['pothole'])
            
        Returns:
            Path to created yaml file
        """
        if class_names is None:
            class_names = ['pothole']
        
        yaml_content = f"""path: {self.output_root}
train: images/train
val: images/val
test: images/test

nc: {num_classes}
names:
"""
        for i, name in enumerate(class_names):
            yaml_content += f"  {i}: {name}\n"
        
        yaml_path = self.output_root / 'data.yaml'
        yaml_path.write_text(yaml_content)
        logger.info(f"Created data.yaml at {yaml_path}")
        
        return yaml_path
    
    def validate_labels(self):
        """
        Validate YOLO label format and completeness.
        
        YOLO label format per line:
        <class_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>
        
        Returns:
            Dict with validation report
        """
        report = {
            'total_images': 0,
            'total_labels': 0,
            'valid_labels': 0,
            'invalid_labels': 0,
            'missing_labels': [],
            'invalid_format': []
        }
        
        for split in ['train', 'val', 'test']:
            images = list((self.images_dir / split).glob('*'))
            labels = list((self.labels_dir / split).glob('*.txt'))
            
            report['total_images'] += len(images)
            report['total_labels'] += len(labels)
            
            # Check for missing labels
            for img_path in images:
                label_path = (self.labels_dir / split) / f"{img_path.stem}.txt"
                if not label_path.exists():
                    report['missing_labels'].append(str(img_path))
            
            # Validate label format
            for label_path in labels:
                try:
                    with open(label_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) != 5:
                                report['invalid_format'].append(f"{label_path}: {line.strip()}")
                            else:
                                # Verify values are floats/ints
                                try:
                                    class_id = int(parts[0])
                                    values = [float(v) for v in parts[1:]]
                                    # Check if normalized (0-1)
                                    if all(0 <= v <= 1 for v in values):
                                        report['valid_labels'] += 1
                                    else:
                                        report['invalid_format'].append(f"{label_path}: values out of range")
                                except ValueError:
                                    report['invalid_format'].append(f"{label_path}: non-numeric values")
                except Exception as e:
                    report['invalid_labels'] += 1
                    logger.error(f"Error reading {label_path}: {e}")
        
        logger.info(f"Validation report: {report['valid_labels']} valid, "
                   f"{report['invalid_labels']} invalid, {len(report['missing_labels'])} missing")
        
        return report
    
    def split_dataset(self, images_dir, labels_dir=None, 
                     train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                     random_state=42):
        """
        Split dataset into train/val/test and organize into YOLO structure.
        
        Args:
            images_dir: Directory containing all images
            labels_dir: Directory containing YOLO labels (optional)
            train_ratio: Training fraction
            val_ratio: Validation fraction
            test_ratio: Test fraction
            random_state: Random seed for reproducibility
        """
        images_path = Path(images_dir)
        image_files = list(images_path.glob('*'))
        
        # Split indices
        train_idx, test_idx = train_test_split(
            range(len(image_files)), 
            test_size=test_ratio,
            random_state=random_state
        )
        train_idx, val_idx = train_test_split(
            train_idx,
            test_size=val_ratio / (train_ratio + val_ratio),
            random_state=random_state
        )
        
        splits = {
            'train': train_idx,
            'val': val_idx,
            'test': test_idx
        }
        
        # Copy files
        for split, indices in splits.items():
            for idx in indices:
                img_file = image_files[idx]
                dest = self.images_dir / split / img_file.name
                shutil.copy2(img_file, dest)
                
                if labels_dir:
                    label_file = Path(labels_dir) / f"{img_file.stem}.txt"
                    if label_file.exists():
                        label_dest = self.labels_dir / split / label_file.name
                        shutil.copy2(label_file, label_dest)
        
        logger.info(f"Dataset split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
        
        return splits
