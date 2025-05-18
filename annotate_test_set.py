#!/usr/bin/env python
import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import argparse
import torchvision.transforms as transforms
from train_fewshot_model import FourChannelResNet, DAMAGE_CLASSES

class TestSetAnnotator:
    """Class for annotating test set images with help from the trained model"""
    def __init__(self, model_path, image_dir, mask_dir, output_csv, confidence_threshold=0.3):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.output_csv = output_csv
        self.confidence_threshold = confidence_threshold
        self.window_name = "Car Damage Annotation Tool (Test Set)"
        self.current_image_idx = 0
        self.annotations = []
        self.current_damage_class = 0
        self.damage_classes = DAMAGE_CLASSES
        self.class_colors = [
            (0, 0, 255),    # dent: red
            (0, 255, 0),    # scratch: green
            (255, 0, 0),    # crack: blue
            (255, 255, 0),  # glass shatter: cyan
            (255, 0, 255),  # lamp broken: magenta
            (0, 255, 255),  # tire flat: yellow
            (128, 128, 128) # no_damage: gray
        ]
        
        # Get all image files
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        self.image_files.sort()
        
        # Load existing annotations if file exists
        if os.path.exists(output_csv):
            self.load_annotations(output_csv)
        
        # Load model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(model_path)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406, 0.5], [0.229, 0.224, 0.225, 0.5])
        ])
        
    def load_model(self, model_path):
        """Load the trained model"""
        model = FourChannelResNet(num_classes=len(DAMAGE_CLASSES), multi_class=True)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model
    
    def load_annotations(self, csv_path):
        """Load existing annotations"""
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            self.annotations.append({
                'image_file': row['image_file'],
                'mask_file': row['mask_file'],
                'damage_class': row['damage_class'],
                'annotation_idx': row['annotation_idx']
            })
        print(f"Loaded {len(self.annotations)} existing annotations")
        
        # Find the first image that hasn't been fully annotated
        annotated_images = set()
        for ann in self.annotations:
            annotated_images.add(ann['image_file'])
        
        # Start from the first unannotated image
        for i, img_file in enumerate(self.image_files):
            if img_file not in annotated_images:
                self.current_image_idx = i
                break
    
    def save_annotations(self):
        """Save annotations to CSV"""
        if self.annotations:
            df = pd.DataFrame(self.annotations)
            df.to_csv(self.output_csv, index=False)
            print(f"Saved {len(self.annotations)} annotations to {self.output_csv}")
        else:
            print("No annotations to save")
    
    def get_model_predictions(self, image, mask):
        """Get model predictions for the current image"""
        # Resize for model input
        image_size = (224, 224)
        img_resized = cv2.resize(image, image_size)
        mask_resized = cv2.resize(mask, image_size)
        
        # Create 4-channel input
        mask_normalized = mask_resized / 255.0 if mask_resized.max() > 1 else mask_resized
        image_with_mask = np.dstack((img_resized, mask_normalized))
        
        # Convert to tensor and normalize
        x = self.transform(image_with_mask).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(x)
            probs = outputs.cpu().numpy()[0]
            preds = (probs > self.confidence_threshold).astype(int)
        
        # Return classes and confidences
        results = []
        for cls_idx, is_present in enumerate(preds):
            if is_present:
                results.append({
                    'class_idx': cls_idx,
                    'class_name': self.damage_classes[cls_idx],
                    'confidence': float(probs[cls_idx])
                })
        
        return results
    
    def draw_annotations(self, image, mask, filename):
        """Draw existing annotations on the image"""
        # Create a copy to draw on
        img_with_annotations = image.copy()
        
        # Draw mask overlay with 50% opacity
        if mask is not None:
            mask_overlay = np.zeros_like(image)
            mask_overlay[mask > 0] = [0, 0, 200]  # Red overlay for mask
            img_with_annotations = cv2.addWeighted(img_with_annotations, 1.0, mask_overlay, 0.5, 0)
        
        # Get annotations for this image
        annotations_for_image = [a for a in self.annotations if a['image_file'] == filename]
        
        # Draw class labels
        for i, ann in enumerate(annotations_for_image):
            class_idx = self.damage_classes.index(ann['damage_class'])
            color = self.class_colors[class_idx]
            text = f"#{i+1}: {ann['damage_class']}"
            cv2.putText(img_with_annotations, text, (10, 30 + i*30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Show current selected class
        cv2.putText(img_with_annotations, f"Selected: {self.damage_classes[self.current_damage_class]}", 
                   (10, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                   self.class_colors[self.current_damage_class], 2)
        
        return img_with_annotations
    
    def display_model_predictions(self, image, predictions):
        """Display model predictions on the image"""
        # Create a copy to draw on
        prediction_display = image.copy()
        
        # Draw prediction text
        cv2.putText(prediction_display, "MODEL PREDICTIONS:", 
                   (image.shape[1] - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        for i, pred in enumerate(predictions):
            text = f"{pred['class_name']} ({pred['confidence']:.2f})"
            cv2.putText(prediction_display, text, 
                       (image.shape[1] - 300, 60 + i*30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.class_colors[pred['class_idx']], 2)
        
        return prediction_display
    
    def add_annotation(self, image_file, mask_file):
        """Add an annotation for the current image"""
        # Get annotation index
        annotations_for_image = [a for a in self.annotations if a['image_file'] == image_file]
        annotation_idx = len(annotations_for_image)
        
        # Add annotation
        self.annotations.append({
            'image_file': image_file,
            'mask_file': mask_file,
            'damage_class': self.damage_classes[self.current_damage_class],
            'annotation_idx': annotation_idx
        })
        
        print(f"Added {self.damage_classes[self.current_damage_class]} annotation to {image_file}")
    
    def remove_annotation(self, image_file):
        """Remove the last annotation for this image"""
        annotations_for_image = [a for a in self.annotations if a['image_file'] == image_file]
        if annotations_for_image:
            last_annotation = annotations_for_image[-1]
            self.annotations.remove(last_annotation)
            print(f"Removed {last_annotation['damage_class']} annotation from {image_file}")
    
    def handle_key(self, key, image_file, mask_file):
        """Handle key presses"""
        if key == ord('q'):  # Quit
            return False
        elif key == ord('a'):  # Previous image
            self.current_image_idx = max(0, self.current_image_idx - 1)
        elif key == ord('d'):  # Next image
            self.current_image_idx = min(len(self.image_files) - 1, self.current_image_idx + 1)
        elif key == ord('w'):  # Previous damage class
            self.current_damage_class = (self.current_damage_class - 1) % len(self.damage_classes)
        elif key == ord('s'):  # Next damage class
            self.current_damage_class = (self.current_damage_class + 1) % len(self.damage_classes)
        elif key == ord(' '):  # Add annotation
            self.add_annotation(image_file, mask_file)
        elif key == ord('r'):  # Remove last annotation
            self.remove_annotation(image_file)
        elif key == ord('p'):  # Accept model predictions
            self.accept_model_predictions(image_file, mask_file)
        
        return True
    
    def accept_model_predictions(self, image_file, mask_file):
        """Accept all model predictions for the current image"""
        # Get the image and mask
        img_path = os.path.join(self.image_dir, image_file)
        mask_path = os.path.join(self.mask_dir, mask_file)
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Get model predictions
        predictions = self.get_model_predictions(image, mask)
        
        # Add annotations for all predictions
        annotations_for_image = [a for a in self.annotations if a['image_file'] == image_file]
        annotation_idx = len(annotations_for_image)
        
        for pred in predictions:
            self.annotations.append({
                'image_file': image_file,
                'mask_file': mask_file,
                'damage_class': pred['class_name'],
                'annotation_idx': annotation_idx,
                'confidence': pred['confidence']
            })
            annotation_idx += 1
        
        print(f"Added {len(predictions)} predictions to {image_file}")
    
    def run(self):
        """Run the annotation tool"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        running = True
        while running and self.current_image_idx < len(self.image_files):
            image_file = self.image_files[self.current_image_idx]
            mask_file = os.path.splitext(image_file)[0] + '.png'
            
            # Load image and mask
            img_path = os.path.join(self.image_dir, image_file)
            mask_path = os.path.join(self.mask_dir, mask_file)
            
            image = cv2.imread(img_path)
            if image is None:
                print(f"Error loading {image_file}")
                self.current_image_idx += 1
                continue
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            mask = None
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            # Get model predictions
            predictions = self.get_model_predictions(image, mask)
            
            # Draw annotations
            img_with_annotations = self.draw_annotations(image, mask, image_file)
            
            # Add model predictions
            if predictions:
                img_with_annotations = self.display_model_predictions(img_with_annotations, predictions)
            
            # Display instructions
            info_img = np.zeros((200, img_with_annotations.shape[1], 3), dtype=np.uint8)
            cv2.putText(info_img, f"Image: {image_file} ({self.current_image_idx + 1}/{len(self.image_files)})", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.putText(info_img, "Controls: A/D=Prev/Next image, W/S=Prev/Next class, SPACE=Add annotation", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.putText(info_img, "          R=Remove last, P=Accept predictions, Q=Quit and save", 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            # Combine images
            display_img = np.vstack((img_with_annotations, info_img))
            
            # Convert to BGR for display
            display_img = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)
            
            # Display image
            cv2.imshow(self.window_name, display_img)
            key = cv2.waitKey(1) & 0xFF
            
            # Handle key press
            running = self.handle_key(key, image_file, mask_file)
            
            # Save periodically
            if self.current_image_idx % 5 == 0:
                self.save_annotations()
        
        # Final save
        self.save_annotations()
        cv2.destroyAllWindows()
        print("Annotation complete!")

def main():
    parser = argparse.ArgumentParser(description="Annotate test set images with help from the trained model")
    parser.add_argument('--model_path', type=str, required=True,
                      help="Path to trained model")
    parser.add_argument('--image_dir', type=str, default="CarDD_release/CarDD_SOD/CarDD-TE/CarDD-TE-Image",
                      help="Directory containing test images")
    parser.add_argument('--mask_dir', type=str, default="CarDD_release/CarDD_SOD/CarDD-TE/CarDD-TE-Mask",
                      help="Directory containing test masks")
    parser.add_argument('--output_csv', type=str, default="cardd_sod/test_annotations.csv",
                      help="Output CSV file for annotations")
    parser.add_argument('--confidence', type=float, default=0.3,
                      help="Confidence threshold for model predictions")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    
    # Create and run annotator
    annotator = TestSetAnnotator(
        args.model_path,
        args.image_dir,
        args.mask_dir,
        args.output_csv,
        args.confidence
    )
    annotator.run()

if __name__ == "__main__":
    main() 