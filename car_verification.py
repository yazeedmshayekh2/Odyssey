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
import random

class ImageAugmenter:
    """Handles realistic image augmentation for car verification"""
    
    @staticmethod
    def add_gaussian_noise(image: np.ndarray, mean: float = 0, sigma: float = 15) -> np.ndarray:
        """Add Gaussian noise to simulate sensor noise"""
        if image.dtype != np.float32:
            image = image.astype(np.float32) / 255.0
        
        noise = np.random.normal(mean, sigma / 255.0, image.shape)
        noisy_image = image + noise
        noisy_image = np.clip(noisy_image, 0, 1)
        return (noisy_image * 255).astype(np.uint8)
    
    @staticmethod
    def adjust_brightness(image: np.ndarray, factor: float) -> np.ndarray:
        """Adjust image brightness to simulate different lighting conditions"""
        pil_image = Image.fromarray(image)
        enhancer = ImageEnhance.Brightness(pil_image)
        adjusted = enhancer.enhance(factor)
        return np.array(adjusted)
    
    @staticmethod
    def adjust_contrast(image: np.ndarray, factor: float) -> np.ndarray:
        """Adjust image contrast to simulate different lighting conditions"""
        pil_image = Image.fromarray(image)
        enhancer = ImageEnhance.Contrast(pil_image)
        adjusted = enhancer.enhance(factor)
        return np.array(adjusted)
    
    @staticmethod
    def add_motion_blur(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """Add motion blur to simulate camera movement"""
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        blurred = cv2.filter2D(image, -1, kernel)
        return blurred
    
    @staticmethod
    def add_jpeg_compression(image: np.ndarray, quality: int = 90) -> np.ndarray:
        """Simulate JPEG compression artifacts"""
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode('.jpg', image, encode_param)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        return decoded
    
    @staticmethod
    def apply_realistic_augmentation(image: np.ndarray, is_reference: bool = False) -> np.ndarray:
        """Apply a combination of realistic augmentations"""
        if is_reference:
            # For reference images, apply mild augmentations
            if random.random() < 0.5:
                image = ImageAugmenter.add_gaussian_noise(image, sigma=10)
            if random.random() < 0.3:
                image = ImageAugmenter.adjust_brightness(image, random.uniform(0.9, 1.1))
            if random.random() < 0.3:
                image = ImageAugmenter.adjust_contrast(image, random.uniform(0.9, 1.1))
        else:
            # For uploaded images, apply more aggressive augmentations
            if random.random() < 0.7:
                image = ImageAugmenter.add_gaussian_noise(image, sigma=15)
            if random.random() < 0.5:
                image = ImageAugmenter.adjust_brightness(image, random.uniform(0.8, 1.2))
            if random.random() < 0.5:
                image = ImageAugmenter.adjust_contrast(image, random.uniform(0.8, 1.2))
            if random.random() < 0.3:
                image = ImageAugmenter.add_motion_blur(image)
            if random.random() < 0.4:
                image = ImageAugmenter.add_jpeg_compression(image, quality=random.randint(85, 95))
        
        return image

class StanfordCarsFeatureExtractor:
    """Feature extractor using InceptionV3 trained on Stanford Cars Dataset"""
    
    def __init__(self, weights_path: str = 'stanford_cars_feature_extractor.pth'):
        """Initialize the feature extractor with Stanford Cars weights"""
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

        # Define scales for multi-scale testing
        self.scales = [
            (256, 256),  # Smaller scale for fine details
            (299, 299),  # Native InceptionV3 resolution
            (320, 320),  # Larger scale for context
        ]
        
        # Define geometric augmentations
        self.geometric_augmentations = [
            {'flip': False, 'rotate': 0, 'crop': 1.0},  # Original
            {'flip': True, 'rotate': 0, 'crop': 1.0},   # Horizontal flip
            {'flip': False, 'rotate': 5, 'crop': 0.95}, # Slight clockwise rotation + crop
            {'flip': False, 'rotate': -5, 'crop': 0.95},# Slight counter-clockwise rotation + crop
            {'flip': False, 'rotate': 0, 'crop': 0.9},  # Center crop
        ]

        # Define color augmentations
        self.color_augmentations = [
            {'brightness': 1.0, 'contrast': 1.0, 'saturation': 1.0, 'channel_order': 'RGB'},  # Original
            {'brightness': 1.1, 'contrast': 1.0, 'saturation': 1.0, 'channel_order': 'RGB'},  # Brighter
            {'brightness': 0.9, 'contrast': 1.0, 'saturation': 1.0, 'channel_order': 'RGB'},  # Darker
            {'brightness': 1.0, 'contrast': 1.1, 'saturation': 1.0, 'channel_order': 'RGB'},  # Higher contrast
            {'brightness': 1.0, 'contrast': 1.0, 'saturation': 1.2, 'channel_order': 'RGB'},  # More saturated
            {'brightness': 1.0, 'contrast': 1.0, 'saturation': 0.8, 'channel_order': 'RGB'},  # Less saturated
            {'brightness': 1.0, 'contrast': 1.0, 'saturation': 1.0, 'channel_order': 'BGR'},  # BGR channel order
        ]
        
        # Base transform without resize or color normalization
        self.base_transform = transforms.ToTensor()

        # Color normalization as separate step (after augmentations)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )

    def apply_geometric_augmentation(self, pil_image: Image.Image, aug_params: Dict) -> Image.Image:
        """Apply geometric augmentations to an image"""
        try:
            img = pil_image
            
            # Get original size
            w, h = img.size
            
            # Apply center crop if specified
            if aug_params['crop'] < 1.0:
                crop_size = (int(w * aug_params['crop']), int(h * aug_params['crop']))
                left = (w - crop_size[0]) // 2
                top = (h - crop_size[1]) // 2
                right = left + crop_size[0]
                bottom = top + crop_size[1]
                img = img.crop((left, top, right, bottom))
            
            # Apply rotation if specified
            if aug_params['rotate'] != 0:
                img = img.rotate(aug_params['rotate'], expand=True, resample=Image.BILINEAR)
            
            # Apply horizontal flip if specified
            if aug_params['flip']:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            
            return img
            
        except Exception as e:
            print(f"⚠️ Error applying geometric augmentation: {str(e)}")
            return pil_image

    def apply_color_augmentation(self, image: torch.Tensor, aug_params: Dict) -> torch.Tensor:
        """Apply color space augmentations to a tensor image"""
        try:
            # Start with channel order adjustment
            if aug_params['channel_order'] == 'BGR':
                image = image.flip(0)  # Flip RGB to BGR
            
            # Convert to PIL for color adjustments
            img_pil = transforms.ToPILImage()(image)
            
            # Apply color adjustments
            if aug_params['brightness'] != 1.0:
                enhancer = ImageEnhance.Brightness(img_pil)
                img_pil = enhancer.enhance(aug_params['brightness'])
            
            if aug_params['contrast'] != 1.0:
                enhancer = ImageEnhance.Contrast(img_pil)
                img_pil = enhancer.enhance(aug_params['contrast'])
            
            if aug_params['saturation'] != 1.0:
                enhancer = ImageEnhance.Color(img_pil)
                img_pil = enhancer.enhance(aug_params['saturation'])
            
            # Convert back to tensor
            return transforms.ToTensor()(img_pil)
            
        except Exception as e:
            print(f"⚠️ Error applying color augmentation: {str(e)}")
            return image

    def extract_features_single_scale(self, image_tensor: torch.Tensor) -> np.ndarray:
        """Extract features at a single scale"""
        try:
            with torch.no_grad():
                features = self.model(image_tensor.unsqueeze(0).to(self.device))
                features = features.flatten().cpu().numpy()
                # Normalize features
                features = features / np.linalg.norm(features)
                return features
        except Exception as e:
            print(f"Error extracting features at scale: {str(e)}")
            return None

    def extract_features(self, image_array: np.ndarray) -> np.ndarray:
        """Extract features using multi-scale, multi-augmentation approach with color variations"""
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
            
            # Store all features from different scales and augmentations
            all_features = []
            
            # For each scale
            for scale_idx, scale in enumerate(self.scales):
                print(f"📏 Processing scale {scale_idx + 1}/{len(self.scales)}: {scale}")
                
                # For each geometric augmentation
                for geo_idx, geo_params in enumerate(self.geometric_augmentations):
                    # Apply geometric augmentation
                    geo_augmented = self.apply_geometric_augmentation(pil_image, geo_params)
                    
                    # Create transform for this scale
                    scale_transform = transforms.Resize(scale)
                    
                    # Apply scaling
                    scaled_image = scale_transform(geo_augmented)
                    
                    # Convert to tensor (but don't normalize yet)
                    image_tensor = self.base_transform(scaled_image)
                    
                    # For each color augmentation
                    for color_idx, color_params in enumerate(self.color_augmentations):
                        # Apply color augmentation
                        augmented_tensor = self.apply_color_augmentation(image_tensor, color_params)
                        
                        # Apply normalization
                        normalized_tensor = self.normalize(augmented_tensor)
                        
                        # Extract features
                        features = self.extract_features_single_scale(normalized_tensor)
                        if features is not None:
                            all_features.append(features)
                            
                            # Build augmentation description
                            geo_desc = []
                            if geo_params['flip']: geo_desc.append('flipped')
                            if geo_params['rotate'] != 0: geo_desc.append(f'rotated {geo_params["rotate"]}°')
                            if geo_params['crop'] < 1.0: geo_desc.append(f'cropped {int(geo_params["crop"]*100)}%')
                            geo_desc = ' + '.join(geo_desc) if geo_desc else 'original geometry'
                            
                            color_desc = []
                            if color_params['brightness'] != 1.0: 
                                color_desc.append(f'brightness {int(color_params["brightness"]*100)}%')
                            if color_params['contrast'] != 1.0: 
                                color_desc.append(f'contrast {int(color_params["contrast"]*100)}%')
                            if color_params['saturation'] != 1.0: 
                                color_desc.append(f'saturation {int(color_params["saturation"]*100)}%')
                            if color_params['channel_order'] != 'RGB':
                                color_desc.append(f'{color_params["channel_order"]} channels')
                            color_desc = ' + '.join(color_desc) if color_desc else 'original colors'
                            
                            print(f"  ✨ Variant {len(all_features)}: {geo_desc} | {color_desc}")
            
            if not all_features:
                return None
            
            # Average features from all variants
            avg_features = np.mean(all_features, axis=0)
            # Normalize averaged features
            avg_features = avg_features / np.linalg.norm(avg_features)
            
            total_variants = len(self.scales) * len(self.geometric_augmentations) * len(self.color_augmentations)
            print(f"🧠 Extracted {len(avg_features)}-dimensional feature vector (averaged across {total_variants} variants)")
            return avg_features
            
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
        self.SIMILARITY_THRESHOLD = 0.85  # Base threshold for deep learning features
        self.BACK_SIMILARITY_THRESHOLD = 0.80  # Lower threshold for back view (more variation)
        self.CAR_CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for car detection
        
        # Car class IDs in COCO dataset (used by YOLO)
        self.CAR_CLASS_IDS = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        
        # View weights for weighted similarity calculation
        # Front and back views are more distinctive, left and right are more similar across cars
        self.VIEW_WEIGHTS = {
            "front": 0.35,  # 35% weight - most distinctive view
            "back": 0.35,   # 35% weight - second most distinctive view
            "left": 0.15,   # 15% weight - side views are more similar across cars
            "right": 0.15   # 15% weight - side views are more similar across cars
        }
        # Total weights sum to 1.0: 0.35 + 0.35 + 0.15 + 0.15 = 1.0
        
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
        Enhanced image preprocessing that preserves original colors and adds realistic noise.
        For reference (database) images:
            - Applies basic resizing and normalization
            - Adds slight noise (except for front view)
            - Applies slight blur
            - Matches histogram to a template
        For uploaded images:
            - Only converts BGR to RGB
        """
        try:
            print("📷 Applying preprocessing...")
            
            # Convert BGR to RGB if needed (OpenCV loads as BGR)
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
                print("🔄 Converted BGR to RGB format")
            else:
                image_rgb = image_array
            
            if is_reference:
                print("🎨 Applying enhanced preprocessing for reference image...")
                
                # Convert to LAB color space
                lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                
                # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                l_eq = clahe.apply(l)
                
                # Merge channels back
                lab_eq = cv2.merge([l_eq, a, b])
                image_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
                
                # Apply noise reduction to front view
                if view_type == "front":
                    # Apply denoising to preserve front details
                    image_processed = cv2.fastNlMeansDenoisingColored(image_eq, None, 10, 10, 7, 21)
                    print("✅ Applied histogram equalization and denoising for front view")
                else:
                    # Add slight Gaussian noise for other views
                    noise = np.random.normal(0, 1, image_eq.shape).astype(np.uint8)  # Reduced noise intensity
                    image_noisy = cv2.add(image_eq, noise)
            
                    # Apply slight Gaussian blur
                    image_processed = cv2.GaussianBlur(image_noisy, (3,3), 0.5)
                    print("✅ Applied histogram equalization, noise, and blur")
                
                return image_processed
            
            print("✅ Basic preprocessing completed")
            return image_rgb
            
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
    
    def calculate_weighted_similarity(self, similarities: Dict[str, float], valid_sides: List[str]) -> float:
        """Calculate weighted similarity score based on available views"""
        try:
            if not similarities or not valid_sides:
                return 0.0
            
            # Normalize weights for available sides
            total_weight = sum(self.VIEW_WEIGHTS[side] for side in valid_sides)
            if total_weight == 0:
                return 0.0
            
            # Calculate weighted sum
            weighted_sum = sum(similarities[side] * (self.VIEW_WEIGHTS[side] / total_weight) 
                             for side in valid_sides)
            
            return weighted_sum
            
        except Exception as e:
            print(f"Error calculating weighted similarity: {str(e)}")
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
            if similarity >= 0.85:
                confidence = 'High'
            elif similarity >= 0.80:
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
        """Verify if uploaded car images match the reference car images"""
        try:
            print("🚀 Starting car verification with weighted processing...")
            
            # Process each view
            similarities = {}
            processing_results = {}
            
            for view in ['front', 'back', 'left', 'right']:
                if view in reference_images and view in uploaded_images:
                    result = self.compare_images(reference_images[view], uploaded_images[view])
                    if result['similarity'] is not None:
                        similarities[view] = result['similarity']
                        processing_results[view] = result
            
            if not similarities:
                return {
                    'match': False,
                    'weighted_similarity': 0.0,
                    'simple_average': 0.0,
                    'confidence': 'Low',
                    'details': {},
                    'error': 'No valid similarity scores computed'
                }
            
            # Calculate weighted similarity
            weighted_similarity = self.calculate_weighted_similarity(similarities, list(similarities.keys()))
            
            # Calculate simple average for comparison
            simple_average = sum(similarities.values()) / len(similarities)
            
            # Determine confidence level based on weighted similarity
            if weighted_similarity >= 0.85:
                confidence = 'High'
            elif weighted_similarity >= 0.80:
                confidence = 'Medium'
            else:
                confidence = 'Low'
            
            # Determine if it's a match based on weighted similarity and confidence
            is_match = weighted_similarity >= self.SIMILARITY_THRESHOLD
            
            print(f"✅ Verification complete! Weighted similarity: {weighted_similarity:.4f}, Simple average: {simple_average:.4f}")
            
            return {
                'match': is_match,
                'weighted_similarity': weighted_similarity,
                'simple_average': simple_average,
                'confidence': confidence,
                'details': processing_results
            }
            
        except Exception as e:
            print(f"Error in car verification: {str(e)}")
            return {
                'match': False,
                'weighted_similarity': 0.0,
                'simple_average': 0.0,
                'confidence': 'Low',
                'details': {},
                'error': str(e)
            }
    
    def verify_with_stored_features(self, stored_reference_features: Dict[str, np.ndarray], uploaded_images: Dict[str, str], save_visualizations: bool = False, vis_dir: str = None) -> Dict:
        """
        Verify uploaded images against pre-computed reference features stored in database.
        This is much faster as it avoids recomputing reference features.
        """
        results = {}
        sides = ["front", "back", "left", "right"]
        
        all_matches = True
        similarities = {}
        valid_sides = []
        
        print("🚀 Starting car verification with stored reference features and weighted processing...")
        
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
                            'model_used': 'InceptionV3 + Weighted Processing + YOLOv11 (Stored Features)',
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
                        'model_used': 'InceptionV3 + Weighted Processing + YOLOv11 (Stored Features)',
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
                    
                    # Store similarity for weighted calculation
                    similarities[side] = similarity
                    valid_sides.append(side)
                    
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
        
        # Calculate weighted similarity instead of simple average
        weighted_similarity = self.calculate_weighted_similarity(similarities, valid_sides)
        
        # Also calculate simple average for comparison
        simple_average = sum(similarities.values()) / len(similarities) if similarities else 0.0
        
        # Enhanced overall confidence based on weighted similarity
        if weighted_similarity >= 0.85 and all_matches:
            overall_confidence = "Very High"
        elif weighted_similarity >= self.SIMILARITY_THRESHOLD and all_matches:
            overall_confidence = "High"
        elif weighted_similarity >= 0.7:
            overall_confidence = "Medium"
        else:
            overall_confidence = "Low"
        
        # Enhanced matching logic: Consider same car if weighted similarity >= 80% even if some sides fail
        weighted_threshold = 0.85
        is_same_car_weighted = weighted_similarity >= weighted_threshold
        is_same_car_traditional = all_matches and weighted_similarity >= self.SIMILARITY_THRESHOLD
        
        # Use weighted approach as primary decision
        final_is_same_car = is_same_car_weighted
        
        results["overall_result"] = {
            "is_same_car": final_is_same_car,
            "weighted_similarity": round(weighted_similarity, 4),
            "average_similarity": round(simple_average, 4),  # Keep simple average for comparison
            "confidence": overall_confidence,
            "matched_sides": sum(1 for side in sides if results.get(side, {}).get("is_match", False)),
            "valid_comparisons": len(valid_sides),
            "detection_method": "YOLOv11 + InceptionV3 + Weighted Processing + Cosine Similarity",
            "similarity_thresholds": {
                "default": self.SIMILARITY_THRESHOLD,
                "back": self.BACK_SIMILARITY_THRESHOLD,
                "weighted_threshold": weighted_threshold
            },
            "model_architecture": "InceptionV3 with Global Average Pooling for feature extraction",
            "view_weights": self.VIEW_WEIGHTS,
            "weighting_explanation": "Front and back views weighted more heavily (35% each) as they are more distinctive than side views (15% each)",
            "matching_logic": f"Same car if weighted similarity >= {weighted_threshold*100}% even if individual sides fail their thresholds"
        }
        
        print(f"✅ Verification complete! Weighted similarity: {weighted_similarity:.4f}, Simple average: {simple_average:.4f}")
        print("⚡ Used stored reference features for faster processing")
        
        return results 