import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import json
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Tuple, List, Union
import os
from ultralytics import YOLO
import requests

class StanfordCarsFeatureExtractor:
    """Feature extractor using InceptionV3 trained on Stanford Cars Dataset"""
    
    def __init__(self, weights_path: str = 'stanford_cars_feature_extractor.pth'):
        print("🚀 Initializing Stanford Cars Feature Extractor...")
        
        try:
            # Try to load Stanford Cars PyTorch weights
            weights_loaded = False
            if os.path.exists(weights_path):
                try:
                    print(f"📁 Found PyTorch weights file: {weights_path}")
                    
                    # Load complete model first
                    self.model = inception_v3(weights=None)
                    self.model.aux_logits = False
                    
                    # Load the state dict
                    state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
                    self.model.load_state_dict(state_dict)
                    
                    # Now replace fc layer with Identity for feature extraction
                    self.model.fc = nn.Identity()
                    
                    weights_loaded = True
                    print("✅ Loaded Stanford Cars feature extractor weights")
                except Exception as e:
                    print(f"⚠️ Failed to load PyTorch weights: {e}")
            
            # Fallback to ImageNet if Stanford Cars weights not available
            if not weights_loaded:
                print("🔄 Loading ImageNet pretrained weights as fallback")
                self.model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
                self.model.aux_logits = False
                self.model.fc = nn.Identity()
                print("✅ Loaded ImageNet weights (still effective for feature extraction)")
            
            self.model.eval()
            
            # Set device
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            print(f"✅ Model initialized on {self.device}")
            
        except Exception as e:
            print(f"⚠️ Error initializing model: {e}")
            self.model = None
        
        # Image preprocessing pipeline (matches InceptionV3 training)
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),  # InceptionV3 requires 299x299 input
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def extract_features(self, image_array: np.ndarray) -> np.ndarray:
        """Extract features from an image using InceptionV3"""
        try:
            if self.model is None:
                return None
            
            # Convert BGR to RGB if needed (OpenCV loads as BGR)
            if len(image_array.shape) == 3:
                image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image_array
            
            # Convert to PIL Image
            pil_image = Image.fromarray(image_rgb.astype(np.uint8))
            
            # Apply transformations
            input_tensor = self.transform(pil_image).unsqueeze(0)
            input_tensor = input_tensor.to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.model(input_tensor)
                features = features.flatten().cpu().numpy()
            
            # Normalize features
            features = features / np.linalg.norm(features)
            
            print(f"🧠 Extracted {len(features)}-dimensional feature vector")
            return features
            
        except Exception as e:
            print(f"Error extracting features: {str(e)}")
            return None

class CarImageVerifier:
    def __init__(self):
        print("🚀 Initializing Car Image Verifier with YOLOv11 and Stanford Cars InceptionV3...")
        
        # Initialize YOLOv11 for car detection
        try:
            self.yolo_model = YOLO('yolo11x.pt')  # Using YOLOv11 nano for speed
            print("✅ YOLOv11 model loaded successfully")
        except Exception as e:
            print(f"⚠️ Error loading YOLOv11: {e}")
            self.yolo_model = None
        
        # Initialize Stanford Cars feature extractor
        self.feature_extractor = StanfordCarsFeatureExtractor()
        
        # Similarity thresholds
        self.SIMILARITY_THRESHOLD = 0.80  # Base threshold for deep learning features
        self.BACK_SIMILARITY_THRESHOLD = 0.75  # Lower threshold for back view (more variation)
        self.CAR_CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for car detection
        
        # Car class IDs in COCO dataset (used by YOLO)
        self.CAR_CLASS_IDS = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        
        # Initialize requests session for URL downloads
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
    def download_image_from_url(self, url: str, save_path: str = None) -> Union[str, np.ndarray]:
        """Download image from URL and optionally save it to disk"""
        try:
            print(f"📥 Downloading image from URL: {url}")
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            # Convert to numpy array
            image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            if image is None:
                raise ValueError("Failed to decode image from URL")

            if save_path:
                # Ensure directory exists
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                # Save the image
                cv2.imwrite(save_path, image)
                print(f"💾 Saved downloaded image to: {save_path}")
                return save_path
            
            return image

        except Exception as e:
            print(f"⚠️ Error downloading image from URL: {str(e)}")
            return None
    
    def detect_cars(self, image_path: str) -> List[Dict]:
        """Detect cars in the image using YOLOv11"""
        try:
            if self.yolo_model is None:
                return []
            
            # Run YOLO detection
            results = self.yolo_model(image_path, verbose=False)
            
            car_detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Check if detection is a vehicle
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        if class_id in self.CAR_CLASS_IDS and confidence >= self.CAR_CONFIDENCE_THRESHOLD:
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            car_detections.append({
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': confidence,
                                'class_id': class_id,
                                'area': (x2 - x1) * (y2 - y1)
                            })
            
            # Sort by confidence and area (prefer larger, more confident detections)
            car_detections.sort(key=lambda x: x['confidence'] * (x['area'] ** 0.5), reverse=True)
            return car_detections
            
        except Exception as e:
            print(f"Error in car detection: {str(e)}")
            return []
    
    def crop_car_region(self, image_path: str, bbox: List[int]) -> np.ndarray:
        """Crop the car region from the image exactly at the bounding box (NO padding)"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            x1, y1, x2, y2 = bbox
            h, w = image.shape[:2]
            
            # Ensure coordinates are within image bounds (no padding added)
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            
            # Ensure we have a valid crop region
            if x2 <= x1 or y2 <= y1:
                print(f"⚠️ Invalid bounding box: ({x1}, {y1}, {x2}, {y2})")
                return None
            
            # Crop exactly at the bounding box
            cropped = image[y1:y2, x1:x2]
            
            print(f"✂️ Cropped car region: ({x1}, {y1}, {x2}, {y2}) -> {x2-x1}x{y2-y1} (NO PADDING)")
            return cropped
            
        except Exception as e:
            print(f"Error cropping car region: {str(e)}")
            return None
    
    def save_processed_image(self, image_array: np.ndarray, save_path: str) -> str:
        """Save processed image for visualization"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Convert RGB to BGR for OpenCV saving
            if len(image_array.shape) == 3:
                bgr_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            else:
                bgr_image = image_array
            
            # Save image
            cv2.imwrite(save_path, bgr_image)
            return save_path
        except Exception as e:
            print(f"⚠️ Failed to save processed image: {str(e)}")
            return None
    
    def draw_detection_boxes(self, image_path: str, detections: List[Dict], save_path: str) -> str:
        """Draw bounding boxes on detected cars and save visualization"""
        try:
            # Read original image
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Draw bounding boxes
            for i, detection in enumerate(detections):
                bbox = detection['bbox']
                confidence = detection['confidence']
                class_id = detection['class_id']
                
                # Get class name
                class_names = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
                class_name = class_names.get(class_id, 'vehicle')
                
                # Draw bounding box
                x1, y1, x2, y2 = bbox
                color = (0, 255, 0) if i == 0 else (0, 165, 255)  # Green for best detection, orange for others
                thickness = 3 if i == 0 else 2
                
                cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
                
                # Add label
                label = f"{class_name}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # Background for text
                cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0], y1), color, -1)
                
                # Text
                cv2.putText(image, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Add selection indicator for best detection
                if i == 0:
                    cv2.putText(image, "SELECTED", (x1, y2 + 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add summary text
            summary = f"Detected: {len(detections)} vehicle(s)"
            cv2.putText(image, summary, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(image, summary, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
            
            # Save visualization
            cv2.imwrite(save_path, image)
            return save_path
            
        except Exception as e:
            print(f"⚠️ Failed to create detection visualization: {str(e)}")
            return None
    
    def create_no_detection_image(self, image_path: str, save_path: str) -> str:
        """Create visualization for images with no car detection"""
        try:
            # Read original image
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Add "No car detected" overlay
            height, width = image.shape[:2]
            
            # Create semi-transparent overlay
            overlay = image.copy()
            cv2.rectangle(overlay, (0, 0), (width, 80), (0, 0, 255), -1)
            alpha = 0.7
            image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
            
            # Add text
            cv2.putText(image, "NO CAR DETECTED", (10, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            cv2.putText(image, "Using full image for processing", (10, 65), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Save visualization
            cv2.imwrite(save_path, image)
            return save_path
            
        except Exception as e:
            print(f"⚠️ Failed to create no detection image: {str(e)}")
            return None
    
    def extract_features(self, image_array: np.ndarray) -> np.ndarray:
        """Extract features using Stanford Cars trained InceptionV3"""
        return self.feature_extractor.extract_features(image_array)
    
    def extract_features_from_preprocessed(self, preprocessed_image: np.ndarray) -> np.ndarray:
        """Extract features from already preprocessed image"""
        return self.feature_extractor.extract_features(preprocessed_image)
    
    def preprocess_image(self, image_array: np.ndarray, is_reference: bool = False, view_type: str = None) -> np.ndarray:
        """
        Enhanced image preprocessing that preserves original colors.
        For reference (database) images:
            - Applies basic resizing and normalization
            - No color modifications
        For uploaded images:
            - Only converts BGR to RGB if needed
        """
        try:
            print("📷 Applying preprocessing...")
            
            # Convert BGR to RGB if needed (OpenCV loads as BGR)
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
                print("🔄 Converted BGR to RGB format")
            else:
                image_rgb = image_array
                print("⚠️ Image is not in BGR format, using as is")
            
            # Basic preprocessing without color modifications
            if is_reference:
                # Ensure image is float32 and normalized to [0, 1]
                image_processed = image_rgb.astype(np.float32) / 255.0
                
                # Apply very mild Gaussian blur only to reference images (0.5 sigma)
                if view_type != "front":  # Skip blur for front view
                    image_processed = cv2.GaussianBlur(image_processed, (3, 3), 0.5)
                
                # Convert back to uint8 for visualization
                image_processed = (image_processed * 255).astype(np.uint8)
            else:
                # For uploaded images, just normalize
                image_processed = image_rgb.astype(np.float32) / 255.0
                image_processed = (image_processed * 255).astype(np.uint8)
            
            print("✅ Basic preprocessing completed")
            return image_processed
            
        except Exception as e:
            print(f"⚠️ Error in preprocessing: {str(e)}")
            return image_array  # Return original image if preprocessing fails
    
    def process_image(self, image_path: str, save_visualizations: bool = False, vis_dir: str = None, skip_preprocessing: bool = False, is_reference: bool = False, view_type: str = None) -> Dict:
        """Process image: detect car and extract features with optional visualizations"""
        try:
            print(f"🔍 Processing: {os.path.basename(image_path)}")
            
            # Try to infer view type from filename if not provided
            if view_type is None:
                filename = os.path.basename(image_path).lower()
                if "front" in filename:
                    view_type = "front"
                elif "back" in filename:
                    view_type = "back"
                elif "left" in filename:
                    view_type = "left"
                elif "right" in filename:
                    view_type = "right"
            
            # Initialize visualization paths
            detection_vis_path = None
            preprocessed_vis_path = None
            
            # First detect cars in the image
            car_detections = self.detect_cars(image_path)
            
            # Save detection visualization if requested
            if save_visualizations and vis_dir:
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                detection_vis_path = os.path.join(vis_dir, f"{base_name}_detection.jpg")
                if car_detections:
                    self.draw_detection_boxes(image_path, car_detections, detection_vis_path)
                else:
                    # Save original image with "No car detected" text
                    self.create_no_detection_image(image_path, detection_vis_path)
            
            if not car_detections:
                # If no car detected, use the whole image
                print(f"⚠️ No car detected in {os.path.basename(image_path)}, using full image")
                image = cv2.imread(image_path)
                if image is None:
                    return {
                        'features': None,
                        'error': 'Could not read image',
                        'car_detected': False,
                        'detection_confidence': 0.0,
                        'detection_visualization': detection_vis_path,
                        'preprocessed_visualization': None
                    }
                
                # Apply preprocessing and save if requested
                preprocessed_image = self.preprocess_image(image, is_reference=is_reference, view_type=view_type)
                if save_visualizations and vis_dir:
                    base_name = os.path.splitext(os.path.basename(image_path))[0]
                    preprocessed_vis_path = os.path.join(vis_dir, f"{base_name}_preprocessed.jpg")
                    self.save_processed_image(preprocessed_image, preprocessed_vis_path)
                else:
                    preprocessed_vis_path = None
                
                # Extract features from image
                features = self.extract_features_from_preprocessed(preprocessed_image)
                
                processing_method = 'InceptionV3 + Enhanced Preprocessing on full image (no car detected)' if is_reference else 'InceptionV3 + Basic Preprocessing on full image (no car detected)'
                
                return {
                    'features': features,
                    'car_detected': False,
                    'detection_confidence': 0.0,
                    'bbox': None,
                    'processing_method': processing_method,
                    'detection_visualization': detection_vis_path,
                    'preprocessed_visualization': preprocessed_vis_path
                }
            
            # Use the car detection with highest confidence
            best_detection = car_detections[0]
            print(f"🎯 Car detected with {best_detection['confidence']:.3f} confidence")
            
            # Crop the car region
            cropped_car = self.crop_car_region(image_path, best_detection['bbox'])
            
            if cropped_car is None:
                return {
                    'features': None,
                    'error': 'Could not crop car region',
                    'car_detected': True,
                    'detection_confidence': best_detection['confidence'],
                    'detection_visualization': detection_vis_path,
                    'preprocessed_visualization': None
                }
            
            # Apply preprocessing and save if requested
            preprocessed_car = self.preprocess_image(cropped_car, is_reference=is_reference, view_type=view_type)
            if save_visualizations and vis_dir:
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                preprocessed_vis_path = os.path.join(vis_dir, f"{base_name}_preprocessed.jpg")
                self.save_processed_image(preprocessed_car, preprocessed_vis_path)
            else:
                preprocessed_vis_path = None
            
            # Extract features from cropped car
            features = self.extract_features_from_preprocessed(preprocessed_car)
            
            processing_method = 'InceptionV3 + Enhanced Preprocessing on detected car region' if is_reference else 'InceptionV3 + Basic Preprocessing on detected car region'
            
            return {
                'features': features,
                'car_detected': True,
                'detection_confidence': best_detection['confidence'],
                'bbox': best_detection['bbox'],
                'num_detections': len(car_detections),
                'processing_method': processing_method,
                'detection_visualization': detection_vis_path,
                'preprocessed_visualization': preprocessed_vis_path
            }
            
        except Exception as e:
            return {
                'features': None,
                'error': f'Processing failed: {str(e)}',
                'car_detected': False,
                'detection_confidence': 0.0,
                'detection_visualization': detection_vis_path,
                'preprocessed_visualization': preprocessed_vis_path
            }
    
    def calculate_cosine_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Calculate cosine similarity between two feature vectors"""
        try:
            if features1 is None or features2 is None:
                return 0.0
            
            # Reshape for sklearn
            features1 = features1.reshape(1, -1)
            features2 = features2.reshape(1, -1)
            
            similarity = cosine_similarity(features1, features2)[0][0]
            return float(similarity)
            
        except Exception as e:
            print(f"Error calculating cosine similarity: {str(e)}")
            return 0.0
    
    def compare_images(self, reference_path: str, uploaded_path: str) -> Dict:
        """Compare two images using YOLOv11 detection and InceptionV3 features with advanced preprocessing"""
        try:
            print(f"🔬 Comparing: {os.path.basename(reference_path)} ↔ {os.path.basename(uploaded_path)}")
            
            # Process both images
            ref_result = self.process_image(reference_path)
            upload_result = self.process_image(uploaded_path)
            
            # Check for errors
            if ref_result.get('error') or upload_result.get('error'):
                return {
                    'cosine_similarity': 0.0,
                    'is_match': False,
                    'confidence': 'Low',
                    'reference_car_detected': ref_result.get('car_detected', False),
                    'upload_car_detected': upload_result.get('car_detected', False),
                    'reference_detection_confidence': ref_result.get('detection_confidence', 0.0),
                    'upload_detection_confidence': upload_result.get('detection_confidence', 0.0),
                    'error': ref_result.get('error') or upload_result.get('error'),
                    'model_used': 'InceptionV3 + Advanced Preprocessing + YOLOv11'
                }
            
            # Calculate cosine similarity
            similarity = self.calculate_cosine_similarity(
                ref_result['features'],
                upload_result['features']
            )
            
            # Get the filename to determine if it's a back view
            filename = os.path.basename(uploaded_path).lower()
            is_back_view = "back" in filename
            
            # Use different threshold for back view
            threshold = self.BACK_SIMILARITY_THRESHOLD if is_back_view else self.SIMILARITY_THRESHOLD
            is_match = similarity >= threshold
            
            # Determine confidence level
            if similarity >= 0.8:
                confidence = 'High'
            elif similarity >= 0.75:
                confidence = 'Medium'
            else:
                confidence = 'Low'
            
            # Boost confidence if both cars are well-detected
            both_detected = ref_result.get('car_detected', False) and upload_result.get('car_detected', False)
            if both_detected and similarity >= 0.85:
                if confidence == 'Medium':
                    confidence = 'High'
            
            return {
                'cosine_similarity': round(similarity, 4),
                'is_match': is_match,
                'confidence': confidence,
                'reference_car_detected': ref_result.get('car_detected', False),
                'upload_car_detected': upload_result.get('car_detected', False),
                'reference_detection_confidence': round(ref_result.get('detection_confidence', 0.0), 3),
                'upload_detection_confidence': round(upload_result.get('detection_confidence', 0.0), 3),
                'reference_detections': ref_result.get('num_detections', 0),
                'upload_detections': upload_result.get('num_detections', 0),
                'model_used': 'InceptionV3 + Advanced Preprocessing + YOLOv11',
                'processing_method': {
                    'reference': ref_result.get('processing_method', 'Unknown'),
                    'upload': upload_result.get('processing_method', 'Unknown')
                }
            }
            
        except Exception as e:
            return {
                'cosine_similarity': 0.0,
                'is_match': False,
                'confidence': 'Low',
                'error': f'Comparison failed: {str(e)}',
                'reference_car_detected': False,
                'upload_car_detected': False,
                'reference_detection_confidence': 0.0,
                'upload_detection_confidence': 0.0,
                'model_used': 'InceptionV3 + Advanced Preprocessing + YOLOv11'
            }
    
    def verify_car_images(self, reference_images: Dict[str, str], uploaded_images: Dict[str, str]) -> Dict:
        """Verify all four sides of the car using deep learning models"""
        results = {}
        sides = ["front", "back", "left", "right"]
        
        all_matches = True
        total_similarity = 0.0
        valid_comparisons = 0
        
        print("🚀 Starting car verification with YOLOv11 + InceptionV3 + Basic Processing...")
        
        for side in sides:
            print(f"📷 Processing {side} view...")
            
            if side in reference_images and side in uploaded_images:
                if os.path.exists(reference_images[side]) and os.path.exists(uploaded_images[side]):
                    comparison = self.compare_images(reference_images[side], uploaded_images[side])
                    results[side] = comparison
                    
                    if not comparison["is_match"]:
                        all_matches = False
                    
                    total_similarity += comparison["cosine_similarity"]
                    valid_comparisons += 1
                else:
                    results[side] = {
                        "error": f"Image file not found for {side}",
                        "is_match": False,
                        "cosine_similarity": 0.0,
                        "confidence": "Low",
                        "model_used": "InceptionV3 + Basic Processing + YOLOv11"
                    }
                    all_matches = False
            else:
                results[side] = {
                    "error": f"Missing {side} image",
                    "is_match": False,
                    "cosine_similarity": 0.0,
                    "confidence": "Low",
                    "model_used": "InceptionV3 + Basic Processing + YOLOv11"
                }
                all_matches = False
        
        # Calculate overall verification result
        average_similarity = total_similarity / valid_comparisons if valid_comparisons > 0 else 0.0
        
        # Enhanced overall confidence
        if average_similarity >= 0.85 and all_matches:
            overall_confidence = "Very High"
        elif average_similarity >= self.SIMILARITY_THRESHOLD and all_matches:
            overall_confidence = "High"
        elif average_similarity >= 0.75:
            overall_confidence = "Medium"
        else:
            overall_confidence = "Low"
        
        results["overall_result"] = {
            "is_same_car": all_matches and average_similarity >= self.SIMILARITY_THRESHOLD,
            "average_similarity": round(average_similarity, 4),
            "confidence": overall_confidence,
            "matched_sides": sum(1 for side in sides if results.get(side, {}).get("is_match", False)),
            "valid_comparisons": valid_comparisons,
            "detection_method": "YOLOv11 + InceptionV3 + Basic Processing + Cosine Similarity",
            "similarity_thresholds": {
                "default": self.SIMILARITY_THRESHOLD,
                "back": self.BACK_SIMILARITY_THRESHOLD
            },
            "model_architecture": "InceptionV3 with Global Average Pooling for feature extraction"
        }
        
        print(f"✅ Verification complete! Average similarity: {average_similarity:.4f}")
        
        return results
    
    def verify_with_stored_features(self, stored_reference_features: Dict[str, np.ndarray], uploaded_images: Dict[str, str], save_visualizations: bool = False, vis_dir: str = None) -> Dict:
        """
        Verify uploaded images against pre-computed reference features stored in database.
        This is much faster as it avoids recomputing reference features.
        """
        results = {}
        sides = ["front", "back", "left", "right"]
        
        all_matches = True
        total_similarity = 0.0
        valid_comparisons = 0
        
        print("🚀 Starting car verification with stored reference features...")
        
        for side in sides:
            print(f"📷 Processing {side} view...")
            
            if side in stored_reference_features and side in uploaded_images:
                ref_features = stored_reference_features[side]
                upload_path = uploaded_images[side]
                
                if ref_features is not None and os.path.exists(upload_path):
                    # Process only the uploaded image (reference features are already computed)
                    # Apply basic processing to all sides consistently
                    upload_result = self.process_image(upload_path, save_visualizations, vis_dir, skip_preprocessing=False, is_reference=False, view_type=side)
                    
                    if upload_result.get('error') or upload_result['features'] is None:
                        results[side] = {
                            'cosine_similarity': 0.0,
                            'is_match': False,
                            'confidence': 'Low',
                            'upload_car_detected': upload_result.get('car_detected', False),
                            'upload_detection_confidence': upload_result.get('detection_confidence', 0.0),
                            'error': upload_result.get('error', 'Failed to extract features'),
                            'model_used': 'InceptionV3 + Basic Processing + YOLOv11 (Stored Features)',
                            # Add visualization paths even for errors
                            'upload_detection_visualization': upload_result.get('detection_visualization'),
                            'upload_preprocessed_visualization': upload_result.get('preprocessed_visualization')
                        }
                        all_matches = False
                        continue
                    
                    # Calculate similarity between stored reference features and uploaded image features
                    similarity = self.calculate_cosine_similarity(ref_features, upload_result['features'])
                    
                    # Use different threshold for back view
                    threshold = self.BACK_SIMILARITY_THRESHOLD if side == "back" else self.SIMILARITY_THRESHOLD
                    is_match = similarity >= threshold
                    
                    # Determine confidence level
                    if similarity >= 0.8:
                        confidence = 'High'
                    elif similarity >= 0.7:
                        confidence = 'Medium'
                    else:
                        confidence = 'Low'
                    
                    # Boost confidence if uploaded car is well-detected
                    if upload_result.get('car_detected', False) and similarity >= 0.85:
                        if confidence == 'Medium':
                            confidence = 'High'
                    
                    results[side] = {
                        'cosine_similarity': round(similarity, 4),
                        'is_match': is_match,
                        'confidence': confidence,
                        'upload_car_detected': upload_result.get('car_detected', False),
                        'upload_detection_confidence': round(upload_result.get('detection_confidence', 0.0), 3),
                        'upload_detections': upload_result.get('num_detections', 0),
                        'model_used': 'InceptionV3 + Basic Processing + YOLOv11 (Stored Features)',
                        'processing_method': {
                            'reference': 'Pre-computed stored features',
                            'upload': upload_result.get('processing_method', 'Unknown')
                        },
                        # Add visualization paths
                        'upload_detection_visualization': upload_result.get('detection_visualization'),
                        'upload_preprocessed_visualization': upload_result.get('preprocessed_visualization')
                    }
                    
                    if not is_match:
                        all_matches = False
                    
                    total_similarity += similarity
                    valid_comparisons += 1
                    
                else:
                    error_msg = "Missing stored reference features" if ref_features is None else f"Upload image file not found for {side}"
                    results[side] = {
                        "error": error_msg,
                        "is_match": False,
                        "cosine_similarity": 0.0,
                        "confidence": "Low",
                        "model_used": "InceptionV3 + Basic Processing + YOLOv11 (Stored Features)"
                    }
                    all_matches = False
            else:
                results[side] = {
                    "error": f"Missing reference features or upload image for {side}",
                    "is_match": False,
                    "cosine_similarity": 0.0,
                    "confidence": "Low",
                    "model_used": "InceptionV3 + Basic Processing + YOLOv11 (Stored Features)"
                }
                all_matches = False
        
        # Calculate overall verification result
        average_similarity = total_similarity / valid_comparisons if valid_comparisons > 0 else 0.0
        
        # Enhanced overall confidence 
        if average_similarity >= 0.85 and all_matches:
            overall_confidence = "Very High"
        elif average_similarity >= self.SIMILARITY_THRESHOLD and all_matches:
            overall_confidence = "High"
        elif average_similarity >= 0.7:
            overall_confidence = "Medium"
        else:
            overall_confidence = "Low"
        
        results["overall_result"] = {
            "is_same_car": all_matches and average_similarity >= self.SIMILARITY_THRESHOLD,
            "average_similarity": round(average_similarity, 4),
            "confidence": overall_confidence,
            "matched_sides": sum(1 for side in sides if results.get(side, {}).get("is_match", False)),
            "valid_comparisons": valid_comparisons,
            "detection_method": "YOLOv11 + InceptionV3 + Basic Processing + Stored Features",
            "similarity_thresholds": {
                "default": self.SIMILARITY_THRESHOLD,
                "back": self.BACK_SIMILARITY_THRESHOLD
            },
            "model_architecture": "InceptionV3 with Basic Processing and pre-computed reference features",
            "optimization": "Reference features pre-computed and stored in database for faster verification"
        }
        
        print(f"✅ Verification complete! Average similarity: {average_similarity:.4f}")
        print("⚡ Used stored reference features for faster processing")
        
        return results 