#!/usr/bin/env python
import os
import cv2
import numpy as np
import pandas as pd
import argparse
import random
from pathlib import Path
import json

# Damage classes from the project
DAMAGE_CLASSES = ["dent", "scratch", "crack", "glass shatter", "lamp broken", "tire flat", "no_damage"]

def create_labeled_image(image, mask):
    """Create an image with numbered labels for annotation"""
    # Create the mask overlay
    overlay = overlay_mask(image, mask)
    
    # Add class numbers to the image
    height, width = overlay.shape[:2]
    label_image = overlay.copy()
    
    # Add class labels at the bottom of the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    y_pos = height - 30  # Position at the bottom
    
    # Draw a semi-transparent background for the labels
    label_bg = np.zeros((100, width, 3), dtype=np.uint8)
    label_bg_with_alpha = cv2.addWeighted(label_image[height-100:height, :], 0.3, label_bg, 0.7, 0)
    label_image[height-100:height, :] = label_bg_with_alpha
    
    # Draw class labels with numbers
    for i, damage_class in enumerate(DAMAGE_CLASSES):
        text = f"{i}: {damage_class}"
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        
        # Calculate x position to spread labels across the width
        x_pos = int(width * (i + 0.5) / len(DAMAGE_CLASSES)) - text_size[0] // 2
        
        # Ensure label is fully visible
        x_pos = max(5, min(x_pos, width - text_size[0] - 5))
        
        # Draw the text
        cv2.putText(label_image, text, (x_pos, y_pos), font, font_scale, (255, 255, 255), thickness)
    
    return label_image

def overlay_mask(image, mask, alpha=0.5, color=(0, 0, 255)):
    """Overlay the mask on the image with given opacity and color"""
    # Ensure mask is binary and has the same dimensions as the image
    if len(mask.shape) == 3 and mask.shape[2] == 3:
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask.copy()
    
    # Resize mask if necessary
    if mask_gray.shape[0] != image.shape[0] or mask_gray.shape[1] != image.shape[1]:
        mask_gray = cv2.resize(mask_gray, (image.shape[1], image.shape[0]))
    
    # Create color overlay
    overlay = image.copy()
    mask_bool = mask_gray > 127  # Convert to binary mask
    
    # Apply the overlay - this is safer than the previous approach
    overlay_copy = overlay.copy()
    overlay_copy[mask_bool] = [color[0], color[1], color[2]]
    cv2.addWeighted(overlay_copy, alpha, overlay, 1 - alpha, 0, overlay)
    
    return overlay

def select_random_images(image_dir, mask_dir, num_images=100, seed=42):
    """Select random images from the dataset"""
    random.seed(seed)
    all_images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    valid_images = []
    
    # Verify corresponding mask exists
    for img in all_images:
        mask_name = os.path.splitext(img)[0] + '.png'
        if os.path.exists(os.path.join(mask_dir, mask_name)):
            valid_images.append(img)
    
    print(f"Found {len(valid_images)} valid image-mask pairs")
    if len(valid_images) < num_images:
        print(f"Warning: Only {len(valid_images)} valid pairs available")
        return valid_images
    
    return random.sample(valid_images, num_images)

def get_user_input_number(prompt, min_val=1, max_val=10, default=1):
    """Get a number from the user within the specified range"""
    while True:
        try:
            user_input = input(prompt)
            if not user_input:
                return default
            val = int(user_input)
            if min_val <= val <= max_val:
                return val
            else:
                print(f"Please enter a number between {min_val} and {max_val}")
        except ValueError:
            print("Please enter a valid number")

def annotate_images(image_dir, mask_dir, output_csv, num_images=100):
    """Interactive annotation tool for the CarDD_SOD dataset"""
    # Create output directory
    output_dir = os.path.dirname(output_csv)
    os.makedirs(output_dir, exist_ok=True)
    
    # Select images to annotate
    selected_images = select_random_images(image_dir, mask_dir, num_images)
    
    # Initialize dataframe for annotations
    annotations = []
    
    # Display instructions
    print("\n=== CarDD_SOD Manual Annotation Tool ===")
    print("Instructions:")
    print("1. For each image, you'll see the damage mask with class labels")
    print("2. Enter how many different damage types you want to annotate")
    print("3. For each annotation, enter the number (0-6) corresponding to the damage type")
    print("4. Press 'ESC' to exit annotation early")
    print("5. The tool will save progress after every image")
    print("\nDamage Classes:")
    for i, damage_class in enumerate(DAMAGE_CLASSES):
        print(f"{i}: {damage_class}")
    
    # Annotate each image
    for i, img_file in enumerate(selected_images):
        print(f"\nProcessing image {i+1}/{len(selected_images)}: {img_file}")
        
        # Load image and mask
        img_path = os.path.join(image_dir, img_file)
        mask_name = os.path.splitext(img_file)[0] + '.png'
        mask_path = os.path.join(mask_dir, mask_name)
        
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path)
        
        if image is None or mask is None:
            print(f"Error: Could not load image or mask for {img_file}")
            continue
        
        # Create labeled image for annotation
        labeled_image = create_labeled_image(image, mask)
        
        # Resize for display
        display_height = 800
        aspect_ratio = labeled_image.shape[1] / labeled_image.shape[0]
        display_width = int(display_height * aspect_ratio)
        display_img = cv2.resize(labeled_image, (display_width, display_height))
        
        # Show the image with labels
        window_name = "Annotate Damage (Press ESC to exit)"
        cv2.imshow(window_name, display_img)
        cv2.waitKey(100)  # Short delay to ensure the window is shown
        
        # Ask how many annotations for this image with the image already shown
        print("How many damage types do you want to annotate for this image? (1-5, default: 1)")
        annotation_count = get_user_input_number("Number of annotations: ", 1, 5, 1)
        
        # Collect annotations for this image
        image_annotations = []
        escape_pressed = False
        for j in range(annotation_count):
            print(f"Annotation {j+1}/{annotation_count}:")
            
            # Wait for class input
            valid_input = False
            damage_class = None
            
            while not valid_input:
                print("Enter damage class (0-6):", end=' ')
                key = cv2.waitKey(0) & 0xFF
                
                if key == 27:  # ESC key
                    print("Annotation canceled.")
                    cv2.destroyAllWindows()
                    escape_pressed = True
                    break
                
                try:
                    class_idx = int(chr(key))
                    if 0 <= class_idx < len(DAMAGE_CLASSES):
                        damage_class = DAMAGE_CLASSES[class_idx]
                        print(f"Selected: {damage_class}")
                        valid_input = True
                    else:
                        print("Invalid class index. Try again.")
                except ValueError:
                    print("Invalid input. Enter a number 0-6.")
            
            if escape_pressed:
                break
            
            # Add annotation to the list for this image
            image_annotations.append({
                'image_file': img_file,
                'mask_file': mask_name,
                'damage_class': damage_class,
                'annotation_idx': j
            })
        
        # Close the window when done with this image
        cv2.destroyAllWindows()
        
        if escape_pressed:
            break
        
        # Add all annotations for this image to the main list
        annotations.extend(image_annotations)
        
        # Save progress after each image
        df = pd.DataFrame(annotations)
        df.to_csv(output_csv, index=False)
        print(f"Progress saved to {output_csv}")
    
    # Save final annotations
    df = pd.DataFrame(annotations)
    df.to_csv(output_csv, index=False)
    
    # Create a summary file
    annotation_counts = df.groupby('image_file').size().reset_index(name='annotation_count')
    summary = {
        'total_annotated_images': len(annotation_counts),
        'total_annotations': len(annotations),
        'average_annotations_per_image': annotation_counts['annotation_count'].mean(),
        'class_distribution': df['damage_class'].value_counts().to_dict()
    }
    
    summary_path = os.path.join(output_dir, 'annotation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nAnnotation complete!")
    print(f"Annotated {len(annotation_counts)} images with {len(annotations)} total annotations")
    print(f"Average of {annotation_counts['annotation_count'].mean():.1f} annotations per image")
    print(f"Results saved to {output_csv}")
    print(f"Summary saved to {summary_path}")

def main():
    parser = argparse.ArgumentParser(description="Manual annotation tool for CarDD_SOD dataset")
    parser.add_argument('--image_dir', type=str, default="CarDD_release/CarDD_SOD/CarDD-TR/CarDD-TR-Image",
                        help="Directory containing images")
    parser.add_argument('--mask_dir', type=str, default="CarDD_release/CarDD_SOD/CarDD-TR/CarDD-TR-Mask",
                        help="Directory containing masks")
    parser.add_argument('--output', type=str, default="cardd_sod/manual_annotations.csv",
                        help="Output CSV file for annotations")
    parser.add_argument('--count', type=int, default=100,
                        help="Number of images to annotate")
    
    args = parser.parse_args()
    
    annotate_images(args.image_dir, args.mask_dir, args.output, args.count)

if __name__ == "__main__":
    main() 