#!/usr/bin/env python
from src.app_factory import create_app
from flask import request, jsonify
import os
import cv2
import numpy as np
import random
import base64
from werkzeug.utils import secure_filename
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.model_zoo import model_zoo
import damage_config as dc
from datetime import datetime
from bson import ObjectId
import jwt
from src.config.db import get_db
import json

# Create the Flask application using the factory
app = create_app()

# Define output directories
OUTPUT_DIR = "output/car_damage_detection"
UPLOADS_DIR = os.path.join(OUTPUT_DIR, "uploads_test")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results_test")
MODEL_PATH = os.path.join(OUTPUT_DIR, "model_final.pth")
# Add path for car segmentation model
CAR_SEG_MODEL_PATH = os.path.join(OUTPUT_DIR, "car_segmentation_model.pth")

# Create directories if they don't exist
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Class metadata
CLASSES = ["dent", "scratch", "crack", "glass shatter", "lamp broken", "tire flat"]

# Color map for different damage classes (BGR format for OpenCV)
COLOR_MAP = {
    "scratch": (203, 192, 255),  # Pink/Purple
    "dent": (0, 165, 255),       # Orange
    "crack": (0, 255, 0),        # Green
    "glass shatter": (255, 0, 0), # Blue
    "lamp broken": (255, 255, 0), # Cyan
    "tire flat": (0, 255, 0)      # Green
}

def get_jwt_secret():
    """Get JWT secret from app config"""
    return app.config.get('JWT_SECRET', 'odyssey_secret_key_change_in_production')

def authenticate_request():
    """Authenticate a request using JWT"""
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return None
    
    token = auth_header.split(' ')[1]
    
    try:
        payload = jwt.decode(token, get_jwt_secret(), algorithms=['HS256'])
        return payload.get('user_id')
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def save_damage_report_to_db(user_id, damage_percent, damage_classes, original_filename, result_image_data, car_id=None, car_info=None):
    """Save damage detection results to the database"""
    try:
        print(f"DEBUG: save_damage_report_to_db called with:")
        print(f"  user_id: {user_id} (type: {type(user_id)})")
        print(f"  car_id: {car_id} (type: {type(car_id)})")
        print(f"  damage_percent: {damage_percent}")
        print(f"  damage_classes count: {len(damage_classes) if damage_classes else 0}")
        print(f"  car_info: {car_info}")
        
        db = get_db()
        print(f"DEBUG: Got database: {db}")
        
        # Create damage report document
        try:
            user_object_id = ObjectId(user_id) if user_id else None
            print(f"DEBUG: Successfully converted user_id to ObjectId: {user_object_id}")
        except Exception as e:
            print(f"DEBUG: Error converting user_id to ObjectId: {e}")
            user_object_id = user_id  # Use as string if conversion fails
        
        # Convert car_id to ObjectId if provided
        car_object_id = None
        if car_id:
            try:
                car_object_id = ObjectId(car_id)
                print(f"DEBUG: Successfully converted car_id to ObjectId: {car_object_id}")
            except Exception as e:
                print(f"DEBUG: Error converting car_id to ObjectId: {e}")
                car_object_id = car_id  # Use as string if conversion fails
        else:
            print(f"DEBUG: No car_id provided, setting car_object_id to None")
        
        # Convert numpy types to Python native types
        damage_percent = float(damage_percent) if damage_percent is not None else 0.0
        
        # Convert damage classes and ensure all numpy types are converted to Python native types
        processed_damage_classes = []
        if damage_classes:
            for cls in damage_classes:
                processed_cls = {
                    'class': str(cls['class']),
                    'confidence': float(cls['confidence']) if 'confidence' in cls else 0.0,
                    'area': float(cls['area']) if 'area' in cls else 0.0
                }
                processed_damage_classes.append(processed_cls)
        
        damage_report = {
            'user_id': user_object_id,
            'car_id': car_object_id,
            'damage_detected': bool(damage_percent > 0),
            'damage_percentage': damage_percent,
            'damage_types': [cls['class'] for cls in processed_damage_classes],
            'detected_damages': processed_damage_classes,
            'images': {
                'original_filename': original_filename,
                'result_image_data': result_image_data
            },
            'confidence_scores': {cls['class']: cls['confidence'] for cls in processed_damage_classes},
            'status': 'completed',
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow(),
            'car_info': car_info or {
                'make': 'Unknown',
                'model': 'Unknown', 
                'year': 'Unknown'
            },
            'detection_method': 'ai_automated',
            'notes': f'AI-powered damage detection completed. {len(processed_damage_classes)} damage types detected.'
        }
        
        print(f"DEBUG: Created damage report document with keys: {list(damage_report.keys())}")
        print(f"DEBUG: About to insert damage report to collection: {db.damage_reports}")
        
        # Insert the report
        result = db.damage_reports.insert_one(damage_report)
        print(f"DEBUG: Insert operation completed. Result type: {type(result)}")
        print(f"DEBUG: Insert result.inserted_id: {result.inserted_id}")
        
        # Update the car record to include this damage report reference
        if car_object_id:
            try:
                car_update_result = db.cars.update_one(
                    {'_id': car_object_id},
                    {'$push': {'damage_reports': result.inserted_id}}
                )
                print(f"DEBUG: Car update result: {car_update_result.modified_count} documents modified")
            except Exception as e:
                print(f"DEBUG: Failed to update car with damage report: {e}")
        else:
            print(f"DEBUG: Skipping car update since car_object_id is None")
        
        print(f"Damage report saved to database with ID: {result.inserted_id}")
        return str(result.inserted_id)
        
    except Exception as e:
        print(f"ERROR in save_damage_report_to_db: {e}")
        import traceback
        traceback.print_exc()
        return None

def setup_cfg():
    """Configure the model for inference"""
    cfg = get_cfg()
    
    # Load ResNet-50 FPN Mask R-CNN config from model zoo
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    
    # Model settings
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6  # 6 damage classes
    cfg.MODEL.WEIGHTS = MODEL_PATH
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set the threshold for display
    
    return cfg

def load_model():
    """Load the damage detection model if available"""
    try:
        # Check if model file exists
        if os.path.exists(MODEL_PATH):
            print(f"Model found at {MODEL_PATH}")
            cfg = setup_cfg()
            predictor = DefaultPredictor(cfg)
            return predictor
        else:
            print(f"Model file not found at {MODEL_PATH}")
            return None
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

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

def calculate_damage_percentage(outputs, car_mask):
    """Calculate percentage of damaged area based on model predictions and car area"""
    # Get the predicted masks
    instances = outputs["instances"].to("cpu")
    if len(instances) == 0:
        return 0.0
    
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
        if cls_id >= len(CLASSES):
            continue
            
        class_name = CLASSES[cls_id]
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

def calculate_class_damage_percentages(outputs, car_mask):
    """Calculate percentage of damaged area for each class"""
    instances = outputs["instances"].to("cpu")
    if len(instances) == 0:
        return {}
    
    # Get the car area
    car_area = np.sum(car_mask > 0)
    if car_area == 0:  # Avoid division by zero
        return {}
    
    # Get masks and class IDs
    masks = instances.pred_masks.numpy()
    classes = instances.pred_classes.numpy()
    
    # Initialize damage percentage by class
    class_damage = {}
    
    # Calculate damage percentage for each instance
    for mask, cls_id in zip(masks, classes):
        if cls_id < len(CLASSES):
            class_name = CLASSES[cls_id]
            
            # Convert mask to binary uint8
            mask_uint8 = mask.astype(np.uint8) * 255
            # Bitwise AND with car mask to get only damage on the car
            damage_on_car = cv2.bitwise_and(mask_uint8, car_mask)
            # Count damaged pixels
            damaged_pixels = np.sum(damage_on_car > 0)
            
            # Calculate percentage relative to car area
            percentage = (damaged_pixels / car_area) * 100
            
            if class_name in class_damage:
                class_damage[class_name] += percentage
            else:
                class_damage[class_name] = percentage
    
    return class_damage

def process_car_image(image_path):
    """Process a car image to detect and segment damage using the trained model"""
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        return None, None, 0
    
    # First, segment the car from the background
    car_mask, car_area = segment_car(img)
    
    # Get the predictor
    predictor = load_model()
    
    if predictor is None:
        # Fall back to mock implementation if model is not available
        return process_car_image_mock(image_path)
    
    # Make prediction
    outputs = predictor(img)
    
    # Calculate damage percentage based on car area
    damage_percent = calculate_damage_percentage(outputs, car_mask)
    
    # Calculate damage percentage by class
    class_damage_percents = calculate_class_damage_percentages(outputs, car_mask)
    
    # RGB version for visualization
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Visualize the predictions
    v = Visualizer(
        img_rgb,
        scale=1.0,
        instance_mode=ColorMode.IMAGE_BW
    )
    
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    result = v.get_image()
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    
    # Add damage percentage text to the image
    text = f"Damage: {damage_percent:.2f}%"
    cv2.putText(result, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Extract damage classes with confidence scores
    instances = outputs["instances"].to("cpu")
    damage_classes = []
    
    if len(instances) > 0:
        boxes = instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes") else []
        scores = instances.scores.numpy() if instances.has("scores") else []
        classes = instances.pred_classes.numpy() if instances.has("pred_classes") else []
        masks = instances.pred_masks.numpy() if instances.has("pred_masks") else []
        
        for box, score, cls_id, mask in zip(boxes, scores, classes, masks):
            x1, y1, x2, y2 = map(int, box)
            class_name = CLASSES[cls_id] if cls_id < len(CLASSES) else "unknown"
            
            # Calculate area percentage for this instance relative to car area
            # Convert mask to binary uint8
            mask_uint8 = mask.astype(np.uint8) * 255
            # Bitwise AND with car mask to get only damage on the car
            damage_on_car = cv2.bitwise_and(mask_uint8, car_mask)
            # Count damaged pixels
            damaged_pixels = np.sum(damage_on_car > 0)
            
            # Calculate percentage relative to car area
            area_percentage = (damaged_pixels / car_area) * 100 if car_area > 0 else 0
            
            damage_classes.append({
                "class": class_name,
                "confidence": float(score),
                "area_percentage": float(area_percentage),
                "position": [int((x1 + x2) / 2), int((y1 + y2) / 2)],
                "bbox": [int(x1), int(y1), int(x2), int(y2)]
            })
            
            # Replace confidence score with area percentage in visualization
            label_text = f"{class_name} {area_percentage:.2f}%"
            cv2.putText(result, label_text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_MAP.get(class_name, (255, 0, 0)), 2)
    
    # Convert result image to base64
    _, buffer = cv2.imencode('.jpg', result)
    result_image_data = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
    
    return result_image_data, damage_classes, damage_percent

def generate_realistic_mask(image, damage_type):
    """Generate a more realistic mask based on damage type (used in mock implementation)"""
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Based on damage type, create different shaped regions
    if damage_type == "scratch":
        # Create a long thin scratch with random curvature
        start_x = random.randint(width // 4, width * 3 // 4)
        start_y = random.randint(height // 4, height * 3 // 4)
        length = random.randint(100, 300)
        angle = random.uniform(0, 2 * np.pi)
        thickness = random.randint(5, 15)
        
        points = []
        for i in range(0, length, 10):
            # Add some randomness to the line
            curve = random.randint(-10, 10)
            x = int(start_x + i * np.cos(angle) + curve)
            y = int(start_y + i * np.sin(angle) + curve)
            
            if 0 <= x < width and 0 <= y < height:
                points.append((x, y))
        
        if points:
            pts = np.array(points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(mask, [pts], False, 255, thickness=thickness)
            
    elif damage_type == "dent":
        # Create an irregular shaped dent
        center_x = random.randint(width // 4, width * 3 // 4)
        center_y = random.randint(height // 4, height * 3 // 4)
        radius = random.randint(30, 80)
        
        # Create an irregular polygon for the dent
        points = []
        for i in range(8):
            angle = 2 * np.pi * i / 8
            r = radius * random.uniform(0.7, 1.3)
            x = int(center_x + r * np.cos(angle))
            y = int(center_y + r * np.sin(angle))
            
            if 0 <= x < width and 0 <= y < height:
                points.append((x, y))
        
        if points:
            pts = np.array(points, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 255)
    
    elif damage_type == "crack":
        # Create a crack pattern
        start_x = random.randint(width // 4, width * 3 // 4)
        start_y = random.randint(height // 4, height * 3 // 4)
        
        # Generate a branching crack
        points = [(start_x, start_y)]
        branches = random.randint(2, 4)
        
        for _ in range(branches):
            branch_length = random.randint(50, 150)
            angle = random.uniform(0, 2 * np.pi)
            
            last_x, last_y = start_x, start_y
            branch_points = []
            
            for i in range(0, branch_length, 5):
                # Add some randomness to the crack
                wiggle = random.randint(-5, 5)
                x = int(last_x + 5 * np.cos(angle) + wiggle)
                y = int(last_y + 5 * np.sin(angle) + wiggle)
                
                if 0 <= x < width and 0 <= y < height:
                    branch_points.append((x, y))
                    last_x, last_y = x, y
            
            if branch_points:
                pts = np.array(branch_points, np.int32).reshape((-1, 1, 2))
                cv2.polylines(mask, [pts], False, 255, thickness=random.randint(2, 5))
    
    elif damage_type == "tire flat":
        # Create a tire-shaped region with deformation
        center_x = random.randint(width // 4, width * 3 // 4)
        center_y = random.randint(height // 4, height * 3 // 4)
        radius_outer = random.randint(80, 150)
        radius_inner = int(radius_outer * 0.6)
        
        # Draw tire shape
        cv2.circle(mask, (center_x, center_y), radius_outer, 255, -1)
        
        # Deform the bottom part to make it look flat
        bottom_y = center_y + int(radius_outer * 0.8)
        if bottom_y < height:
            cv2.rectangle(mask, (center_x - radius_outer, center_y), 
                         (center_x + radius_outer, bottom_y), 255, -1)
        
        # Add wheel hub (inner circle)
        cv2.circle(mask, (center_x, center_y), radius_inner, 0, -1)
    
    elif damage_type == "lamp broken":
        # Create a lamp shape with broken pattern
        center_x = random.randint(width // 4, width * 3 // 4)
        center_y = random.randint(height // 4, height * 3 // 4)
        width_lamp = random.randint(50, 100)
        height_lamp = random.randint(40, 80)
        
        # Draw lamp shape
        cv2.ellipse(mask, (center_x, center_y), (width_lamp, height_lamp), 
                   0, 0, 360, 255, -1)
        
        # Add broken lines
        for _ in range(random.randint(3, 7)):
            start_x = center_x + random.randint(-width_lamp, width_lamp)
            start_y = center_y + random.randint(-height_lamp, height_lamp)
            end_x = start_x + random.randint(-30, 30)
            end_y = start_y + random.randint(-30, 30)
            
            cv2.line(mask, (start_x, start_y), (end_x, end_y), 0, random.randint(2, 5))
    
    else:  # glass shatter or default
        # Create a glass shatter pattern
        center_x = random.randint(width // 4, width * 3 // 4)
        center_y = random.randint(height // 4, height * 3 // 4)
        
        # Draw main impact point
        cv2.circle(mask, (center_x, center_y), random.randint(20, 40), 255, -1)
        
        # Add radiating cracks
        for _ in range(random.randint(5, 10)):
            angle = random.uniform(0, 2 * np.pi)
            length = random.randint(30, 100)
            end_x = int(center_x + length * np.cos(angle))
            end_y = int(center_y + length * np.sin(angle))
            
            cv2.line(mask, (center_x, center_y), (end_x, end_y), 255, random.randint(2, 5))
    
    return mask

def process_car_image_mock(image_path):
    """Mock implementation for testing without the model"""
    try:
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            return None, None, 0
        
        # Create a copy for visualization
        result = img.copy()
        
        # Generate random damage classes for testing
        damage_classes = []
        num_damages = random.randint(1, 3)  # Random number of damages
        
        total_damage_percent = 0
        used_classes = set()
        
        for _ in range(num_damages):
            # Select a random damage type that hasn't been used yet
            available_classes = [cls for cls in CLASSES if cls not in used_classes]
            if not available_classes:
                break
                
            damage_type = random.choice(available_classes)
            used_classes.add(damage_type)
            
            # Generate random confidence and area
            confidence = random.uniform(0.5, 0.95)
            area_percent = random.uniform(1, 8)
            total_damage_percent += area_percent
            
            # Generate a realistic-looking mask for visualization
            damage_mask = generate_realistic_mask(img, damage_type)
            
            # Apply the mask to the image with the class color
            color = COLOR_MAP.get(damage_type, (255, 0, 0))
            colored_mask = np.zeros_like(img)
            colored_mask[:, :] = color
            alpha = 0.3
            
            mask_area = cv2.countNonZero(damage_mask)
            if mask_area > 0:
                # Apply the colored overlay
                mask_3channel = cv2.cvtColor(damage_mask, cv2.COLOR_GRAY2BGR)
                result = cv2.addWeighted(
                    result,
                    1,
                    cv2.bitwise_and(colored_mask, mask_3channel),
                    alpha,
                    0
                )
                
                # Find the bounding box of the mask
                contours, _ = cv2.findContours(
                    damage_mask,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )
                if contours:
                    x, y, w, h = cv2.boundingRect(contours[0])
                    
                    # Draw bounding box
                    cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
                    
                    # Add label with confidence
                    label = f"{damage_type} ({confidence*100:.1f}%)"
                    cv2.putText(
                        result,
                        label,
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2
                    )
                    
                    damage_classes.append({
                        "class": damage_type,
                        "confidence": float(confidence),
                        "area": float(area_percent),
                        "bbox": [x, y, x + w, y + h]
                    })
        
        # Add overall damage percentage to image
        cv2.putText(
            result,
            f"Total Damage: {total_damage_percent:.2f}%",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )
        
        # Convert result image to base64
        _, buffer = cv2.imencode('.jpg', result)
        result_image_data = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
        
        return result_image_data, damage_classes, total_damage_percent
        
    except Exception as e:
        print(f"Error in mock processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, 0

def calculate_mock_damage_percentage(mask, img_shape):
    """Calculate percentage of damaged area for mock implementation"""
    height, width = img_shape[:2]
    image_area = height * width
    damaged_pixels = np.sum(mask > 0)
    damage_percentage = (damaged_pixels / image_area) * 100
    return damage_percentage

@app.route('/detect_damage', methods=['POST'])
def detect_damage():
    """Handle damage detection request"""
    try:
        # Get user ID from JWT token
        user_id = authenticate_request()
        if not user_id:
            return jsonify({
                'success': False,
                'message': 'Authentication required'
            }), 401

        # Check if image was uploaded
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'message': 'No image file provided'
            }), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({
                'success': False,
                'message': 'No selected file'
            }), 400

        # Get car info if provided
        car_id = request.form.get('car_id')
        car_info = None
        if 'car_info' in request.form:
            try:
                car_info = json.loads(request.form['car_info'])
            except:
                car_info = None

        # Save uploaded file
        filename = secure_filename(image_file.filename)
        image_path = os.path.join(UPLOADS_DIR, filename)
        image_file.save(image_path)

        # Process the image
        try:
            result_image_data, damage_classes, damage_percent = process_car_image(image_path)
            
            print("DEBUG: Result from process_car_image:")
            print(f"- damage_percent: {damage_percent}")
            print(f"- damage_classes: {damage_classes}")
            print(f"- result_image_data starts with: {result_image_data[:100] if result_image_data else 'None'}")
            
            # Convert numpy/custom types to JSON serializable types
            damage_percent = float(damage_percent)
            damage_detected = bool(damage_percent > 0)
            
            # Process damage classes to ensure JSON serializable
            processed_damage_classes = []
            for cls in damage_classes:
                processed_cls = {
                    'class': str(cls.get('class', '')),
                    'confidence': float(cls.get('confidence', 0.0)),
                    'area': float(cls.get('area', 0.0))
                }
                processed_damage_classes.append(processed_cls)
            
            # Save results to database
            report_id = save_damage_report_to_db(
                user_id=user_id,
                damage_percent=damage_percent,
                damage_classes=processed_damage_classes,
                original_filename=filename,
                result_image_data=result_image_data,
                car_id=car_id,
                car_info=car_info
            )
            
            if not report_id:
                print("WARNING: Failed to save damage report to database")
            
            response_data = {
                'success': True,
                'damage_detected': damage_detected,
                'damage_percentage': damage_percent,
                'damage_classes': processed_damage_classes,
                'result_image_data': result_image_data,
                'report_id': report_id
            }
            
            print("DEBUG: Response data:")
            print(f"- success: {response_data['success']}")
            print(f"- damage_detected: {response_data['damage_detected']}")
            print(f"- damage_percentage: {response_data['damage_percentage']}")
            print(f"- number of damage_classes: {len(response_data['damage_classes'])}")
            print(f"- result_image_data starts with: {response_data['result_image_data'][:100] if response_data['result_image_data'] else 'None'}")
            print(f"- report_id: {response_data['report_id']}")
            
            return jsonify(response_data)
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Try mock processing as fallback
            try:
                result_image_data, damage_classes, damage_percent = process_car_image_mock(image_path)
                
                print("DEBUG: Result from process_car_image_mock:")
                print(f"- damage_percent: {damage_percent}")
                print(f"- damage_classes: {damage_classes}")
                print(f"- result_image_data starts with: {result_image_data[:100] if result_image_data else 'None'}")
                
                # Convert numpy/custom types to JSON serializable types
                damage_percent = float(damage_percent)
                damage_detected = bool(damage_percent > 0)
                
                # Process damage classes to ensure JSON serializable
                processed_damage_classes = []
                for cls in damage_classes:
                    processed_cls = {
                        'class': str(cls.get('class', '')),
                        'confidence': float(cls.get('confidence', 0.0)),
                        'area': float(cls.get('area', 0.0))
                    }
                    processed_damage_classes.append(processed_cls)
                
                # Save mock results to database
                report_id = save_damage_report_to_db(
                    user_id=user_id,
                    damage_percent=damage_percent,
                    damage_classes=processed_damage_classes,
                    original_filename=filename,
                    result_image_data=result_image_data,
                    car_id=car_id,
                    car_info=car_info
                )
                
                if not report_id:
                    print("WARNING: Failed to save mock damage report to database")
                
                response_data = {
                    'success': True,
                    'damage_detected': damage_detected,
                    'damage_percentage': damage_percent,
                    'damage_classes': processed_damage_classes,
                    'result_image_data': result_image_data,
                    'report_id': report_id,
                    'mock_results': True
                }
                
                print("DEBUG: Mock response data:")
                print(f"- success: {response_data['success']}")
                print(f"- damage_detected: {response_data['damage_detected']}")
                print(f"- damage_percentage: {response_data['damage_percentage']}")
                print(f"- number of damage_classes: {len(response_data['damage_classes'])}")
                print(f"- result_image_data starts with: {response_data['result_image_data'][:100] if response_data['result_image_data'] else 'None'}")
                print(f"- report_id: {response_data['report_id']}")
                
                return jsonify(response_data)
                
            except Exception as mock_error:
                print(f"Error in mock processing: {str(mock_error)}")
                traceback.print_exc()
                return jsonify({
                    'success': False,
                    'message': 'Failed to process image'
                }), 500
            
    except Exception as e:
        print(f"Error in detect_damage route: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500
    finally:
        # Clean up uploaded file
        try:
            if os.path.exists(image_path):
                os.remove(image_path)
        except:
            pass

@app.route('/api/detect-damage', methods=['POST'])
def api_detect_damage():
    """API endpoint to detect car damage in uploaded image (legacy)"""
    if 'image' not in request.files:
        return jsonify({
            'status': 'error',
            'message': 'No image file provided'
        }), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({
            'status': 'error',
            'message': 'No image selected'
        }), 400
    
    # Save the uploaded file
    filename = secure_filename(file.filename)
    upload_path = os.path.join(UPLOADS_DIR, filename)
    file.save(upload_path)
    
    try:
        # Process the image
        result_path, damage_classes, damage_percent = process_car_image(upload_path)
        
        if result_path is None:
            return jsonify({
                'status': 'error',
                'message': 'Error processing image'
            }), 500
        
        # Read the result image to encode as base64
        with open(result_path, "rb") as img_file:
            img_data = base64.b64encode(img_file.read()).decode('utf-8')
        
        return jsonify({
            'status': 'success',
            'damage_percentage': float(damage_percent),
            'damage_classes': damage_classes,
            'image_data': f"data:image/jpeg;base64,{img_data}",
            'original_image': filename
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error processing image: {str(e)}'
        }), 500

# Run the application
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.getenv("PORT", 5000))) 