from ultralytics import YOLO
import os
import json
import shutil
from pathlib import Path
import yaml
from tqdm import tqdm

def convert_coco_to_yolo(json_path, image_dir, output_dir):
    """
    Convert COCO format annotations to YOLO format
    """
    # Create output directories with absolute paths
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)
    
    # Load COCO annotations
    with open(json_path, 'r') as f:
        coco = json.load(f)
    
    # Create image_id to annotations mapping
    image_annotations = {}
    for ann in coco['annotations']:
        image_id = ann['image_id']
        if image_id not in image_annotations:
            image_annotations[image_id] = []
        image_annotations[image_id].append(ann)
    
    # Process each image
    for img in tqdm(coco['images'], desc='Converting annotations'):
        # Get image info
        img_id = img['id']
        img_name = img['file_name']
        img_width = img['width']
        img_height = img['height']
        
        # Copy image
        src_img_path = os.path.join(image_dir, img_name)
        dst_img_path = os.path.join(output_dir, 'images', img_name)
        shutil.copy2(src_img_path, dst_img_path)
        
        # Convert annotations
        label_path = os.path.join(output_dir, 'labels', 
                                os.path.splitext(img_name)[0] + '.txt')
        
        if img_id in image_annotations:
            with open(label_path, 'w') as f:
                for ann in image_annotations[img_id]:
                    # Get bbox coordinates
                    x, y, w, h = ann['bbox']
                    # Convert to YOLO format (normalized center coordinates)
                    x_center = (x + w/2) / img_width
                    y_center = (y + h/2) / img_height
                    w = w / img_width
                    h = h / img_height
                    # Class ID (subtract 1 to start from 0)
                    class_id = ann['category_id'] - 1
                    # Write YOLO format line
                    f.write(f"{class_id} {x_center} {y_center} {w} {h}\n")

def create_dataset_yaml():
    """
    Create YOLOv8 dataset configuration file
    """
    # Get absolute path to dataset directory
    dataset_path = os.path.abspath('./datasets')
    
    data_yaml = {
        'path': dataset_path,  # absolute path to dataset root dir
        'train': os.path.join('train', 'images'),  # relative path from dataset root
        'val': os.path.join('val', 'images'),      # relative path from dataset root
        'test': os.path.join('test', 'images'),    # relative path from dataset root
        
        'nc': 6,  # number of classes
        'names': ['scratch', 'dent', 'crack', 'glass_shatter', 'tire_flat', 'lamp_broken']
    }
    
    with open('data.yaml', 'w') as f:
        yaml.dump(data_yaml, f, sort_keys=False)

def train_yolov8():
    """
    Train YOLOv8 model
    """
    # Load a model
    model = YOLO('yolov8m.pt') 
    
    # Train the model with custom settings
    results = model.train(
        data='data.yaml',
        epochs=30,
        imgsz=640,
        batch=16,
        name='car_damage_detection',
        patience=50,  # Early stopping patience
        device=0,     # GPU device
        
        # Augmentation settings
        hsv_h=0.015,  # HSV-Hue augmentation
        hsv_s=0.7,    # HSV-Saturation augmentation
        hsv_v=0.4,    # HSV-Value augmentation
        degrees=45,   # Rotation
        translate=0.2,# Translation
        scale=0.9,    # Scaling
        shear=0.0,    # Shear
        flipud=0.0,   # Vertical flip
        fliplr=0.5,   # Horizontal flip
        mosaic=1.0,   # Mosaic augmentation
        mixup=0.1,    # Mixup augmentation
        
        # Optimization settings
        optimizer='SGD',  # Optimizer (SGD, Adam, AdamW)
        lr0=0.01,        # Initial learning rate
        lrf=0.01,        # Final learning rate ratio
        momentum=0.937,   # SGD momentum/Adam beta1
        weight_decay=0.0005,  # Optimizer weight decay
        warmup_epochs=3,  # Warmup epochs
        warmup_momentum=0.8,  # Warmup initial momentum
        warmup_bias_lr=0.1,   # Warmup initial bias lr
        
        # Save settings
        save_period=10,  # Save checkpoint every x epochs
        exist_ok=True    # Overwrite existing experiment
    )
    
    return results

def main():
    # Clean up existing datasets directory if it exists
    if os.path.exists('datasets'):
        shutil.rmtree('datasets')
    
    # Create base dataset directory
    os.makedirs('datasets', exist_ok=True)
    
    # Convert datasets
    for split in ['train', 'val', 'test']:
        print(f"Converting {split} dataset...")
        convert_coco_to_yolo(
            json_path=os.path.join('cardd', f'{split}.json'),
            image_dir=os.path.join('cardd', split),
            output_dir=os.path.join('datasets', split)
        )
    
    # Create dataset configuration
    create_dataset_yaml()
    
    # Print dataset structure for verification
    print("\nDataset structure:")
    for root, dirs, files in os.walk('datasets'):
        level = root.replace('datasets', '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
    
    # Train model
    print("\nStarting YOLOv8 training...")
    results = train_yolov8()
    
    # Print results
    print("\nTraining completed!")
    print(f"Results saved to: {results.save_dir}")

if __name__ == "__main__":
    main() 