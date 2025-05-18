#!/usr/bin/env python
# Evaluation script for car damage detection model

import os
import cv2
import random
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.model_zoo import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
import matplotlib.pyplot as plt
import damage_config as dc

# Define paths
DATASET_ROOT = "CarDD_release/CarDD_COCO"
TEST_JSON = f"{DATASET_ROOT}/annotations/instances_test2017.json"
TEST_DIR = f"{DATASET_ROOT}/test2017"
OUTPUT_DIR = "output/car_damage_detection"
VISUALIZATION_DIR = f"{OUTPUT_DIR}/visualizations"

# Create visualization directory if it doesn't exist
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

# Register the test dataset if not already registered
try:
    DatasetCatalog.get("car_damage_test")
except:
    register_coco_instances("car_damage_test", {}, TEST_JSON, TEST_DIR)

# Class metadata
classes = ["dent", "scratch", "crack", "glass shatter", "lamp broken", "tire flat"]
MetadataCatalog.get("car_damage_test").thing_classes = classes
metadata = MetadataCatalog.get("car_damage_test")

def segment_car(image):
    """
    Segment the car from the background using computer vision techniques
    Returns a mask of the car and the estimated car area
    """
    # Convert to grayscale for processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use adaptive thresholding to handle different lighting conditions
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    
    # Find contours in the image
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create an empty mask for the car
    car_mask = np.zeros_like(gray)
    
    # If contours were found, use the largest one as the car
    if contours:
        # Find the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Only use contours that are reasonably large (to filter out noise)
        if cv2.contourArea(largest_contour) > (image.shape[0] * image.shape[1] * 0.1):
            # Fill the contour in the mask
            cv2.drawContours(car_mask, [largest_contour], -1, 255, -1)
    
    # If no significant contour was found, use a more aggressive approach
    if np.sum(car_mask) < (image.shape[0] * image.shape[1] * 0.1):
        # Try Otsu's thresholding as a fallback
        _, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Apply morphological operations
        dilated = cv2.dilate(otsu_thresh, kernel, iterations=2)
        car_mask = cv2.erode(dilated, kernel, iterations=1)
    
    # As a last resort, if still no good segmentation, assume car takes up most of the image
    # but leave some margin around the edges
    if np.sum(car_mask) < (image.shape[0] * image.shape[1] * 0.1):
        h, w = image.shape[:2]
        margin = int(min(h, w) * 0.1)  # 10% margin
        car_mask = np.zeros_like(gray)
        car_mask[margin:h-margin, margin:w-margin] = 255
    
    # Calculate the car area
    car_area = np.sum(car_mask > 0)
    
    return car_mask, car_area

def setup_cfg(weights_path):
    # Configure the model for inference
    cfg = get_cfg()
    
    # Load ResNet-50 FPN Mask R-CNN config from model zoo
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    
    # Set dataset for evaluation
    cfg.DATASETS.TEST = ("car_damage_test",)
    
    # Model settings
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6  # 6 damage classes
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set the threshold for display
    
    return cfg

def calculate_damage_percentage(outputs, car_mask=None):
    """Calculate percentage of damaged area based on model predictions and car area"""
    # Get the predicted masks
    instances = outputs["instances"].to("cpu")
    if len(instances) == 0:
        return 0.0
    
    # Get the car mask if not provided
    if car_mask is None:
        height, width = outputs["instances"].image_size
        car_mask = np.ones((height, width), dtype=np.uint8) * 255
    
    # Get the car area
    car_area = np.sum(car_mask > 0)
    if car_area == 0:  # Avoid division by zero
        return 0.0
    
    # Get masks and calculate damaged area
    masks = instances.pred_masks.numpy()
    classes = instances.pred_classes.numpy()
    
    # Initialize total weighted damage
    total_weighted_damage = 0.0
    
    # Special handling for tire flat to ensure it doesn't dominate
    has_tire_flat = False
    tire_flat_damage = 0.0
    
    # For each damage mask, count pixels that overlap with the car mask
    for mask, cls_id in zip(masks, classes):
        if cls_id >= len(classes):
            continue
            
        class_name = classes[cls_id]
        damage_severity = dc.DAMAGE_SEVERITY.get(class_name, 1.0)
        
        # Get component importance based on position in the image (simplified)
        # In a real implementation, this would use actual component detection
        component_importance = dc.COMPONENT_IMPORTANCE.get("default", 1.0)
        
        # Convert mask to binary uint8
        mask_uint8 = mask.astype(np.uint8) * 255
        # Bitwise AND with car mask to get only damage on the car
        damage_on_car = cv2.bitwise_and(mask_uint8, car_mask)
        # Count damaged pixels
        damaged_pixels = np.sum(damage_on_car > 0)
        
        # Calculate raw percentage (for reporting)
        raw_percentage = (damaged_pixels / car_area) * 100
        
        # Apply weights to get weighted damage value
        weighted_damage = raw_percentage * damage_severity * component_importance
        
        # Apply special handling for tire flat - repair cost rather than safety impact
        if class_name == "tire flat":
            has_tire_flat = True
            
            # Tire flat shouldn't exceed 25% damage regardless of tire size
            # (since replacing a tire is relatively inexpensive)
            tire_flat_damage = min(weighted_damage, 25.0)
            continue  # Skip adding to total_weighted_damage now
        else:
            # Cap individual damage contribution to prevent extremes
            weighted_damage = min(weighted_damage, dc.MAX_INDIVIDUAL_DAMAGE)
            total_weighted_damage += weighted_damage
    
    # Add tire flat damage after other damage processing
    if has_tire_flat:
        total_weighted_damage += tire_flat_damage
    
    # Cap the total damage at 100%
    final_damage_percentage = min(total_weighted_damage, dc.MAX_TOTAL_DAMAGE)
    
    # For very small damages, adjust the scale to be more realistic
    if final_damage_percentage > 0 and final_damage_percentage < dc.SMALL_DAMAGE_THRESHOLD:
        raw_pixel_percentage = sum(np.sum(cv2.bitwise_and(mask.astype(np.uint8) * 255, car_mask)) 
                                for mask in masks) / car_area * 100
        
        # If raw pixel percentage is really small, reduce the final percentage
        if raw_pixel_percentage < dc.VERY_SMALL_DAMAGE_THRESHOLD:
            final_damage_percentage = max(dc.MIN_DAMAGE_REPORT, final_damage_percentage * dc.VERY_SMALL_DAMAGE_FACTOR)
    
    return final_damage_percentage

def visualize_predictions(image_path, predictor, output_path=None):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        return None, 0
    
    # Segment the car
    car_mask, _ = segment_car(img)
    
    # Convert to RGB for visualization
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Make prediction
    outputs = predictor(img)
    
    # Calculate damage percentage using the car mask
    damage_percent = calculate_damage_percentage(outputs, car_mask)
    
    # Visualize the predictions
    v = Visualizer(img_rgb,
                   metadata=metadata,
                   scale=1.0,
                   instance_mode=ColorMode.IMAGE_BW)
    
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    result = v.get_image()
    
    # Add damage percentage text to the image
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    text = f"Damage: {damage_percent:.2f}%"
    cv2.putText(result, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    if output_path:
        cv2.imwrite(output_path, result)
        return output_path, damage_percent
    else:
        return result, damage_percent

def evaluate_model(weights_path, num_samples=10):
    # Set up configuration
    cfg = setup_cfg(weights_path)
    
    # Create predictor
    predictor = DefaultPredictor(cfg)
    
    # Load test dataset
    test_dataset_dicts = DatasetCatalog.get("car_damage_test")
    
    # Sample random images
    random_indices = random.sample(range(len(test_dataset_dicts)), min(num_samples, len(test_dataset_dicts)))
    
    # Process each image
    total_damage_percent = 0
    processed_images = []
    
    for idx in random_indices:
        # Get image info
        img_dict = test_dataset_dicts[idx]
        img_path = img_dict["file_name"]
        file_name = os.path.basename(img_path)
        
        # Generate output path
        output_path = os.path.join(VISUALIZATION_DIR, f"pred_{file_name}")
        
        # Process image
        output_path, damage_percent = visualize_predictions(img_path, predictor, output_path)
        
        if output_path:
            total_damage_percent += damage_percent
            processed_images.append((output_path, damage_percent))
            print(f"Processed {file_name}: Damage percentage: {damage_percent:.2f}%")
        else:
            print(f"Failed to process {file_name}")
    
    # Calculate average damage percentage
    avg_damage = total_damage_percent / len(processed_images) if processed_images else 0
    print(f"\nAverage damage percentage across {len(processed_images)} samples: {avg_damage:.2f}%")
    print(f"Visualization results saved to {VISUALIZATION_DIR}")
    
    return processed_images, avg_damage

def main():
    # Find the latest weights file
    weights_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.pth')]
    
    if not weights_files:
        print("No model weights found. Please train the model first.")
        return
    
    # Sort by file modification time to get the latest model
    weights_files.sort(key=lambda x: os.path.getmtime(os.path.join(OUTPUT_DIR, x)), reverse=True)
    latest_weights = os.path.join(OUTPUT_DIR, weights_files[0])
    
    print(f"Using model weights: {latest_weights}")
    
    # Evaluate model
    processed_images, avg_damage = evaluate_model(latest_weights, num_samples=5)
    
    print(f"Evaluation complete. Average damage: {avg_damage:.2f}%")

if __name__ == "__main__":
    main() 