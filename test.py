import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from collections import Counter
import webcolors
from PIL import Image
from ultralytics import YOLO

class CarColorIdentifier:
    def __init__(self, model_path='yolo11n-seg.pt'):
        # Load YOLOv11 segmentation model
        self.yolo_model = YOLO(model_path)
        
        # Define common car colors with their RGB values
        self.car_colors = {
            'white': (255, 255, 255),
            'black': (0, 0, 0),
            'silver': (192, 192, 192),
            'gray': (128, 128, 128),
            'red': (255, 0, 0),
            'blue': (0, 0, 255),
            'green': (0, 128, 0),
            'yellow': (255, 255, 0),
            'orange': (255, 165, 0),
            'brown': (165, 42, 42),
            'beige': (245, 245, 220),
            'navy': (0, 0, 128),
            'maroon': (128, 0, 0),
            'purple': (128, 0, 128),
            'gold': (255, 215, 0),
            'pink': (255, 192, 203)
        }
        
        # Car class ID in COCO dataset
        self.car_class_id = 2
    
    def load_image(self, image_path):
        """Load and preprocess image"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def detect_car_region(self, image, confidence_threshold=0.5):
        """
        Detect car regions using YOLOv11 segmentation and return cropped car images with masks
        """
        # Run YOLOv11 segmentation
        results = self.yolo_model(image, conf=confidence_threshold, verbose=False)
        
        car_regions = []
        
        for result in results:
            boxes = result.boxes
            masks = result.masks
            
            if boxes is not None:
                for i, box in enumerate(boxes):
                    # Check if detected object is a car (class 2 in COCO)
                    if int(box.cls) == self.car_class_id:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Add padding around the car (5% of width/height)
                        height, width = image.shape[:2]
                        padding_x = int((x2 - x1) * 0.05)
                        padding_y = int((y2 - y1) * 0.05)
                        
                        # Apply padding with bounds checking
                        x1 = max(0, x1 - padding_x)
                        y1 = max(0, y1 - padding_y)
                        x2 = min(width, x2 + padding_x)
                        y2 = min(height, y2 + padding_y)
                        
                        # Crop the car region
                        car_crop = image[y1:y2, x1:x2]
                        
                        # Extract segmentation mask if available
                        mask = None
                        if masks is not None and i < len(masks.data):
                            # Get the mask for this detection
                            full_mask = masks.data[i].cpu().numpy()
                            # Crop the mask to match the car crop
                            mask_crop = full_mask[y1:y2, x1:x2]
                            mask = mask_crop
                        
                        # Check if crop is valid and has minimum size
                        if car_crop.size > 0 and car_crop.shape[0] > 50 and car_crop.shape[1] > 50:
                            confidence = float(box.conf)
                            car_regions.append({
                                'image': car_crop,
                                'mask': mask,
                                'bbox': (x1, y1, x2, y2),
                                'confidence': confidence,
                                'area': (x2 - x1) * (y2 - y1)
                            })
        
        # Sort by area (largest first) and confidence
        car_regions.sort(key=lambda x: (x['area'], x['confidence']), reverse=True)
        
        return car_regions
    
    def apply_background_removal(self, image, mask, blur_edges=True):
        """
        Apply background removal using segmentation mask
        """
        if mask is None:
            return image
        
        # Ensure mask is in the right format (0-1 range)
        if mask.max() > 1:
            mask = mask / 255.0
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        clean_mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel)
        
        # Convert back to float
        clean_mask = clean_mask.astype(np.float32)
        
        # Apply Gaussian blur to mask edges for smoother transitions
        if blur_edges:
            clean_mask = cv2.GaussianBlur(clean_mask, (5, 5), 1.0)
        
        # Apply mask to image
        if len(image.shape) == 3:
            # For color images, expand mask to 3 channels
            mask_3d = np.stack([clean_mask] * 3, axis=2)
            masked_image = image * mask_3d
        else:
            masked_image = image * clean_mask
        
        return masked_image, clean_mask
    def extract_dominant_colors(self, image, mask=None, k=5):
        """Extract dominant colors using K-means clustering with optional mask"""
        # Apply mask if provided
        if mask is not None:
            # Get only the pixels that belong to the car (mask > threshold)
            mask_threshold = 0.5
            car_pixels_mask = mask > mask_threshold
            
            if len(image.shape) == 3:
                # For color images
                pixels = image[car_pixels_mask]
            else:
                # For grayscale images
                pixels = image[car_pixels_mask].reshape(-1, 1)
        else:
            # Reshape image to be a list of pixels
            pixels = image.reshape(-1, 3)
        
        # If no valid pixels after masking, fall back to all pixels
        if len(pixels) == 0:
            pixels = image.reshape(-1, 3)
        
        # Remove very dark pixels (shadows) and very bright pixels (reflections)
        if len(pixels.shape) == 2 and pixels.shape[1] == 3:
            mask_filter = np.all(pixels > [30, 30, 30], axis=1) & np.all(pixels < [240, 240, 240], axis=1)
            filtered_pixels = pixels[mask_filter]
        else:
            filtered_pixels = pixels
        
        if len(filtered_pixels) == 0:
            filtered_pixels = pixels
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=min(k, len(filtered_pixels)), random_state=42, n_init=10)
        kmeans.fit(filtered_pixels)
        
        # Get cluster centers (dominant colors)
        colors = kmeans.cluster_centers_
        
        # Get color counts
        labels = kmeans.labels_
        color_counts = Counter(labels)
        
        # Sort colors by frequency
        sorted_colors = []
        for label in color_counts.most_common():
            color = colors[label[0]]
            percentage = (label[1] / len(labels)) * 100
            sorted_colors.append((color, percentage))
        
        return sorted_colors
    
    def closest_color(self, rgb):
        """Find the closest predefined car color"""
        min_distance = float('inf')
        closest_color_name = None
        
        r, g, b = rgb
        
        for color_name, color_rgb in self.car_colors.items():
            # Calculate Euclidean distance
            distance = np.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(rgb, color_rgb)))
            
            if distance < min_distance:
                min_distance = distance
                closest_color_name = color_name
        
        return closest_color_name, min_distance
    
    def identify_car_color(self, image_path, show_analysis=False, return_all_cars=False):
        """Main method to identify car color"""
        # Load image
        image = self.load_image(image_path)
        
        # Detect car regions using YOLOv11
        car_regions = self.detect_car_region(image)
        
        if not car_regions:
            print("No cars detected in the image.")
            return []
        
        all_results = []
        
        # Process each detected car
        for i, car_data in enumerate(car_regions):
            car_image = car_data['image']
            car_mask = car_data['mask']
            car_bbox = car_data['bbox']
            detection_confidence = car_data['confidence']
            
            # Apply background removal if mask is available
            if car_mask is not None:
                masked_car_image, processed_mask = self.apply_background_removal(car_image, car_mask)
            else:
                masked_car_image = car_image
                processed_mask = None
            
            # Extract dominant colors from the masked car region
            dominant_colors = self.extract_dominant_colors(masked_car_image, processed_mask, k=3)
            
            # Find the most likely car color
            color_results = []
            for color_rgb, percentage in dominant_colors:
                color_name, distance = self.closest_color(color_rgb)
                color_results.append({
                    'color': color_name,
                    'rgb': color_rgb,
                    'percentage': percentage,
                    'confidence': max(0, 100 - (distance / 255 * 100))
                })
            
            car_result = {
                'car_id': i + 1,
                'bbox': car_bbox,
                'detection_confidence': detection_confidence,
                'colors': color_results,
                'car_image': car_image,
                'masked_image': masked_car_image if car_mask is not None else None,
                'has_mask': car_mask is not None
            }
            
            all_results.append(car_result)
            
            # Show analysis if requested
            if show_analysis:
                if car_result['has_mask']:
                    self.show_color_analysis_with_mask(car_image, masked_car_image, color_results, f"Car {i+1}")
                else:
                    self.show_color_analysis(car_image, color_results, f"Car {i+1}")
            
            # If only processing the first (largest) car
            if not return_all_cars:
                break
        
        return all_results
    
    def show_color_analysis_with_mask(self, original_image, masked_image, results, title="Car Color Analysis"):
        """Display the analysis results with original and masked images"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Show original image
        axes[0].imshow(original_image)
        axes[0].set_title(f'{title} - Original')
        axes[0].axis('off')
        
        # Show masked image
        axes[1].imshow(masked_image.astype(np.uint8))
        axes[1].set_title(f'{title} - Background Removed')
        axes[1].axis('off')
        
        # Show color palette
        colors = [result['rgb']/255 for result in results]
        labels = [f"{result['color']}\n({result['percentage']:.1f}%)" for result in results]
        
        axes[2].pie([result['percentage'] for result in results], 
                   colors=colors, labels=labels, autopct='%1.1f%%')
        axes[2].set_title('Dominant Colors')
        
        plt.tight_layout()
        plt.show()

    def show_color_analysis(self, image, results, title="Car Color Analysis"):
        """Display the analysis results"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Show original image
        axes[0].imshow(image)
        axes[0].set_title(f'{title} - Original Image')
        axes[0].axis('off')
        
        # Show color palette
        colors = [result['rgb']/255 for result in results]
        labels = [f"{result['color']}\n({result['percentage']:.1f}%)" for result in results]
        
        axes[1].pie([result['percentage'] for result in results], 
                   colors=colors, labels=labels, autopct='%1.1f%%')
        axes[1].set_title('Dominant Colors')
        
        plt.tight_layout()
        plt.show()
    
    def predict_single_color(self, image_path):
        """Get the single most likely car color"""
        results = self.identify_car_color(image_path)
        if results and results[0]['colors']:
            best_result = max(results[0]['colors'], key=lambda x: x['confidence'])
            return best_result['color']
        return "unknown"
    
    def get_all_car_colors(self, image_path):
        """Get colors for all detected cars"""
        results = self.identify_car_color(image_path, return_all_cars=True)
        car_colors = []
        
        for car_result in results:
            if car_result['colors']:
                best_color = max(car_result['colors'], key=lambda x: x['confidence'])
                car_colors.append({
                    'car_id': car_result['car_id'],
                    'color': best_color['color'],
                    'confidence': best_color['confidence'],
                    'detection_confidence': car_result['detection_confidence'],
                    'bbox': car_result['bbox']
                })
        
        return car_colors

# Example usage and testing
def main():
    # Initialize the car color identifier
    identifier = CarColorIdentifier()
    
    # Example usage (you'll need to provide actual image paths)
    image_path = "uploads/reference/Toyota Supra_2000/back_184992f7b9d14056a5f033bc2a7035ce.png"
    
    try:
        # Get detailed analysis for the main car
        results = identifier.identify_car_color(image_path, show_analysis=True)
        
        if results:
            print("Car Color Analysis Results:")
            print("-" * 50)
            
            for car_result in results:
                print(f"Car {car_result['car_id']}:")
                print(f"  Detection Confidence: {car_result['detection_confidence']:.2f}")
                print(f"  Bounding Box: {car_result['bbox']}")
                print("  Colors:")
                
                for i, color_result in enumerate(car_result['colors'], 1):
                    print(f"    {i}. Color: {color_result['color']}")
                    print(f"       RGB: {color_result['rgb']}")
                    print(f"       Percentage: {color_result['percentage']:.1f}%")
                    print(f"       Confidence: {color_result['confidence']:.1f}%")
                print()
        
        # Get single prediction for the main car
        predicted_color = identifier.predict_single_color(image_path)
        print(f"Main car predicted color: {predicted_color}")
        
        # Get all car colors in the image
        all_car_colors = identifier.get_all_car_colors(image_path)
        print(f"\nAll detected cars:")
        for car_color in all_car_colors:
            print(f"  Car {car_color['car_id']}: {car_color['color']} "
                  f"(confidence: {car_color['confidence']:.1f}%)")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have installed: pip install ultralytics")

# Advanced version with deep learning (optional)
class DeepLearningCarColorIdentifier:
    """
    Advanced version using deep learning
    Requires tensorflow/keras and a pre-trained model
    """
    
    def __init__(self):
        # This would load a pre-trained model
        # For example, a CNN trained on car images with color labels
        pass
    
    def preprocess_image(self, image_path, target_size=(224, 224)):
        """Preprocess image for deep learning model"""
        from tensorflow.keras.preprocessing import image
        from tensorflow.keras.applications.resnet50 import preprocess_input
        
        img = image.load_img(image_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        return img_array
    
    def predict_color(self, image_path):
        """Predict color using deep learning model"""
        # This would use a trained model to predict color
        # preprocessed_image = self.preprocess_image(image_path)
        # prediction = self.model.predict(preprocessed_image)
        # return self.decode_prediction(prediction)
        pass

if __name__ == "__main__":
    main()