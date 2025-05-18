#!/usr/bin/env python
import os
import cv2
import numpy as np
import pandas as pd
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import model_zoo
import random
import damage_config as dc
import json

# Define paths
OUTPUT_DIR = "output/car_damage_detection"
MODEL_PATH = os.path.join(OUTPUT_DIR, "model_final.pth")
CARDD_SOD_DIR = "CarDD_release/CarDD_SOD/CarDD-TR/CarDD-TR-Image"
OUTPUT_ANNOTATIONS_DIR = "cardd_sod"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_ANNOTATIONS_DIR, exist_ok=True)

# Class metadata
CLASSES = ["dent", "scratch", "crack", "glass shatter", "lamp broken", "tire flat"]

def setup_cfg():
    """Configure the model for inference"""
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6  # 6 damage classes
    cfg.MODEL.WEIGHTS = MODEL_PATH
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    return cfg

def segment_car(image):
    """Segment the car from the background"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    car_mask = np.zeros_like(gray)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > (image.shape[0] * image.shape[1] * 0.1):
            cv2.drawContours(car_mask, [largest_contour], -1, 255, -1)
    
    if np.sum(car_mask) < (image.shape[0] * image.shape[1] * 0.1):
        _, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        dilated = cv2.dilate(otsu_thresh, kernel, iterations=2)
        car_mask = cv2.erode(dilated, kernel, iterations=1)
    
    if np.sum(car_mask) < (image.shape[0] * image.shape[1] * 0.1):
        h, w = image.shape[:2]
        margin = int(min(h, w) * 0.1)
        car_mask = np.zeros_like(gray)
        car_mask[margin:h-margin, margin:w-margin] = 255
    
    return car_mask

def process_image(image_path, predictor):
    """Process a single image and return damage information"""
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    # Get car mask
    car_mask = segment_car(image)
    
    # Run inference
    outputs = predictor(image)
    instances = outputs["instances"].to("cpu")
    
    if len(instances) == 0:
        return {
            'image_path': image_path,
            'total_damage': 0.0,
            'damages': []
        }
    
    # Get predictions
    boxes = instances.pred_boxes.tensor.numpy()
    classes = instances.pred_classes.numpy()
    scores = instances.scores.numpy()
    masks = instances.pred_masks.numpy()
    
    # Calculate car area
    car_area = np.sum(car_mask > 0)
    
    damages = []
    total_weighted_damage = 0.0
    
    for box, cls_id, score, mask in zip(boxes, classes, scores, masks):
        if cls_id >= len(CLASSES):
            continue
        
        class_name = CLASSES[cls_id]
        damage_severity = dc.DAMAGE_SEVERITY.get(class_name, 1.0)
        component_importance = dc.COMPONENT_IMPORTANCE.get("default", 1.0)
        
        # Convert mask to binary uint8
        mask_uint8 = mask.astype(np.uint8) * 255
        damage_on_car = cv2.bitwise_and(mask_uint8, car_mask)
        damaged_pixels = np.sum(damage_on_car > 0)
        
        # Calculate damage percentage
        damage_percentage = (damaged_pixels / car_area) * 100 if car_area > 0 else 0
        weighted_damage = damage_percentage * damage_severity * component_importance
        
        if class_name == "tire flat":
            weighted_damage = min(weighted_damage, 25.0)
        else:
            weighted_damage = min(weighted_damage, dc.MAX_INDIVIDUAL_DAMAGE)
        
        damages.append({
            'class': class_name,
            'confidence': float(score),
            'bbox': box.tolist(),
            'damage_percentage': float(damage_percentage),
            'weighted_damage': float(weighted_damage)
        })
        
        total_weighted_damage += weighted_damage
    
    return {
        'image_path': image_path,
        'total_damage': min(float(total_weighted_damage), 100.0),
        'damages': damages
    }

def main():
    # Load model
    cfg = setup_cfg()
    predictor = DefaultPredictor(cfg)
    
    # Get list of images
    image_files = [f for f in os.listdir(CARDD_SOD_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Take 10 random images for testing
    selected_images = random.sample(image_files, min(10, len(image_files)))
    
    # Process images and collect results
    results = []
    for image_file in selected_images:
        image_path = os.path.join(CARDD_SOD_DIR, image_file)
        result = process_image(image_path, predictor)
        if result:
            results.append(result)
    
    # Create DataFrame for annotations
    annotations = []
    for result in results:
        image_name = os.path.basename(result['image_path'])
        for damage in result['damages']:
            annotations.append({
                'image_name': image_name,
                'damage_class': damage['class'],
                'confidence': damage['confidence'],
                'bbox_x1': damage['bbox'][0],
                'bbox_y1': damage['bbox'][1],
                'bbox_x2': damage['bbox'][2],
                'bbox_y2': damage['bbox'][3],
                'damage_percentage': damage['damage_percentage'],
                'weighted_damage': damage['weighted_damage']
            })
    
    # Save annotations to CSV
    df = pd.DataFrame(annotations)
    csv_path = os.path.join(OUTPUT_ANNOTATIONS_DIR, 'cardd_sod_annotations.csv')
    df.to_csv(csv_path, index=False)
    
    # Save summary statistics
    summary = {
        'total_images': len(selected_images),
        'total_annotations': len(annotations),
        'damage_class_distribution': df['damage_class'].value_counts().to_dict(),
        'average_confidence': df['confidence'].mean(),
        'average_damage_percentage': df['damage_percentage'].mean()
    }
    
    summary_path = os.path.join(OUTPUT_ANNOTATIONS_DIR, 'annotation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Processed {len(selected_images)} images")
    print(f"Created {len(annotations)} annotations")
    print(f"Results saved to {csv_path}")
    print(f"Summary saved to {summary_path}")

if __name__ == "__main__":
    main() 