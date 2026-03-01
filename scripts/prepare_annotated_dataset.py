"""
Data Preparation Script for Annotated Pothole Dataset
Converts XML annotations to YOLO format and organizes training/test split

Usage:
    python prepare_annotated_dataset.py --input /path/to/annotated-images --output ./dataset_yolo
"""

import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil
import argparse
import logging
from typing import Dict, List, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("DataPrep")


class AnnotatedDatasetConverter:
    """Convert XML annotations to YOLO format"""
    
    def __init__(self, input_dir: Path, output_dir: Path, splits_file: Path = None):
        """
        Initialize converter
        
        Args:
            input_dir: Directory with annotated-images (contains .jpg and .xml files)
            output_dir: Output directory for YOLO-formatted dataset
            splits_file: Optional splits.json file for train/test split
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.splits_file = Path(splits_file) if splits_file else None
        
        # Create output directories
        self.train_images_dir = self.output_dir / "images" / "train"
        self.train_labels_dir = self.output_dir / "labels" / "train"
        self.test_images_dir = self.output_dir / "images" / "test"
        self.test_labels_dir = self.output_dir / "labels" / "test"
        
        for dir_path in [self.train_images_dir, self.train_labels_dir, 
                         self.test_images_dir, self.test_labels_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.stats = {
            'total_images': 0,
            'train_images': 0,
            'test_images': 0,
            'total_annotations': 0,
            'skipped': 0
        }
    
    def parse_xml_annotation(self, xml_file: Path) -> Dict:
        """
        Parse XML annotation file
        
        Args:
            xml_file: Path to XML file
            
        Returns:
            Dict with image info and bounding boxes
        """
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Get image info
            filename = root.findtext('filename', '')
            size = root.find('size')
            width = int(size.findtext('width', 0)) if size is not None else 0
            height = int(size.findtext('height', 0)) if size is not None else 0
            
            # Get bounding boxes
            bboxes = []
            for obj in root.findall('object'):
                class_name = obj.findtext('name', 'pothole')
                bndbox = obj.find('bndbox')
                
                if bndbox is not None:
                    xmin = float(bndbox.findtext('xmin', 0))
                    ymin = float(bndbox.findtext('ymin', 0))
                    xmax = float(bndbox.findtext('xmax', 0))
                    ymax = float(bndbox.findtext('ymax', 0))
                    
                    bboxes.append({
                        'class': class_name,
                        'xmin': xmin,
                        'ymin': ymin,
                        'xmax': xmax,
                        'ymax': ymax
                    })
            
            return {
                'filename': filename,
                'width': width,
                'height': height,
                'bboxes': bboxes
            }
        except Exception as e:
            logger.error(f"Error parsing {xml_file}: {e}")
            return None
    
    def convert_bbox_to_yolo(self, bbox: Dict, img_width: int, img_height: int) -> str:
        """
        Convert bbox to YOLO format (class_id, center_x, center_y, width, height - all normalized)
        
        Args:
            bbox: Bounding box dict with xmin, ymin, xmax, ymax
            img_width: Image width
            img_height: Image height
            
        Returns:
            YOLO format string
        """
        # Class ID for pothole (0 if only one class)
        class_id = 0
        
        # Calculate center and dimensions
        center_x = (bbox['xmin'] + bbox['xmax']) / 2.0 / img_width
        center_y = (bbox['ymin'] + bbox['ymax']) / 2.0 / img_height
        width = (bbox['xmax'] - bbox['xmin']) / img_width
        height = (bbox['ymax'] - bbox['ymin']) / img_height
        
        # Normalize to 0-1 range
        center_x = max(0, min(1, center_x))
        center_y = max(0, min(1, center_y))
        width = max(0, min(1, width))
        height = max(0, min(1, height))
        
        return f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
    
    def get_split(self, xml_filename: str) -> str:
        """
        Determine if sample is train or test
        
        Args:
            xml_filename: Name of XML file (e.g., "img-273.xml")
            
        Returns:
            "train" or "test"
        """
        if self.splits_file and self.splits_file.exists():
            with open(self.splits_file, 'r') as f:
                splits = json.load(f)
            
            if xml_filename in splits.get('train', []):
                return 'train'
            elif xml_filename in splits.get('test', []):
                return 'test'
        
        # Default: 80/20 split based on filename number
        try:
            num = int(xml_filename.replace('img-', '').replace('.xml', ''))
            return 'train' if num % 5 != 0 else 'test'
        except:
            return 'train'
    
    def process(self):
        """Process all XML and image files"""
        
        logger.info(f"Input directory: {self.input_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        
        if self.splits_file and self.splits_file.exists():
            logger.info(f"Using splits from: {self.splits_file}")
        
        # Find all XML files
        xml_files = list(self.input_dir.glob('*.xml'))
        logger.info(f"Found {len(xml_files)} XML annotation files")
        
        for xml_file in xml_files:
            xml_name = xml_file.name
            logger.info(f"\nProcessing: {xml_name}")
            
            # Parse XML
            annotation = self.parse_xml_annotation(xml_file)
            if not annotation:
                self.stats['skipped'] += 1
                continue
            
            # Find corresponding image file
            img_stem = xml_file.stem  # e.g., "img-273"
            img_candidates = list(self.input_dir.glob(f"{img_stem}.jpg")) + \
                           list(self.input_dir.glob(f"{img_stem}.png")) + \
                           list(self.input_dir.glob(f"{img_stem}.JPG"))
            
            if not img_candidates:
                logger.warning(f"  ✗ Image file not found for {xml_name}")
                self.stats['skipped'] += 1
                continue
            
            img_file = img_candidates[0]
            logger.info(f"  ✓ Found image: {img_file.name}")
            
            # Determine split
            split = self.get_split(xml_name)
            logger.info(f"  → Split: {split}")
            
            # Set output paths
            if split == 'train':
                out_img_dir = self.train_images_dir
                out_lbl_dir = self.train_labels_dir
                self.stats['train_images'] += 1
            else:
                out_img_dir = self.test_images_dir
                out_lbl_dir = self.test_labels_dir
                self.stats['test_images'] += 1
            
            # Copy image
            out_img_path = out_img_dir / img_file.name
            shutil.copy2(img_file, out_img_path)
            logger.info(f"  ✓ Copied image to {out_img_path.relative_to(self.output_dir)}")
            
            # Create YOLO label file
            lbl_filename = f"{img_stem}.txt"
            lbl_path = out_lbl_dir / lbl_filename
            
            with open(lbl_path, 'w') as f:
                for bbox in annotation['bboxes']:
                    yolo_line = self.convert_bbox_to_yolo(
                        bbox,
                        annotation['width'],
                        annotation['height']
                    )
                    f.write(yolo_line + '\n')
                    self.stats['total_annotations'] += 1
            
            logger.info(f"  ✓ Created label file: {lbl_filename} ({len(annotation['bboxes'])} boxes)")
            self.stats['total_images'] += 1
        
        # Create data.yaml for YOLO
        self.create_data_yaml()
        
        # Print summary
        self.print_summary()
    
    def create_data_yaml(self):
        """Create data.yaml for YOLO training"""
        
        data_yaml_content = f"""# Pothole Detection Dataset
path: {self.output_dir.resolve()}
train: images/train
val: images/test
test: images/test

# Number of classes
nc: 1

# Class names
names:
  0: pothole
"""
        
        yaml_path = self.output_dir / "data.yaml"
        with open(yaml_path, 'w') as f:
            f.write(data_yaml_content)
        
        logger.info(f"\n✓ Created data.yaml at: {yaml_path}")
    
    def print_summary(self):
        """Print conversion summary"""
        
        logger.info("\n" + "=" * 80)
        logger.info("DATASET CONVERSION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total images processed: {self.stats['total_images']}")
        logger.info(f"Training images: {self.stats['train_images']}")
        logger.info(f"Test images: {self.stats['test_images']}")
        logger.info(f"Total annotations: {self.stats['total_annotations']}")
        logger.info(f"Skipped: {self.stats['skipped']}")
        logger.info("")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"├── images/")
        logger.info(f"│   ├── train/ ({self.stats['train_images']} images)")
        logger.info(f"│   └── test/ ({self.stats['test_images']} images)")
        logger.info(f"├── labels/")
        logger.info(f"│   ├── train/ ({self.stats['train_images']} labels)")
        logger.info(f"│   └── test/ ({self.stats['test_images']} labels)")
        logger.info(f"└── data.yaml")
        logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Convert annotated dataset to YOLO format"
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input directory with annotated-images (.jpg + .xml files)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='./dataset_yolo',
        help='Output directory for YOLO-formatted dataset'
    )
    
    parser.add_argument(
        '--splits',
        type=str,
        help='Path to splits.json file for train/test split'
    )
    
    args = parser.parse_args()
    
    # Validate input
    input_dir = Path(args.input)
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return
    
    # Convert
    converter = AnnotatedDatasetConverter(input_dir, Path(args.output), args.splits)
    converter.process()


if __name__ == '__main__':
    main()
