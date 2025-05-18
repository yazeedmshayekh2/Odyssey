#!/usr/bin/env python
import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.model_selection import train_test_split
import argparse
import json
from datetime import datetime
from collections import Counter

# Damage classes from the project
DAMAGE_CLASSES = ["dent", "scratch", "crack", "glass shatter", "lamp broken", "tire flat", "no_damage"]

class CarDamageDataset(Dataset):
    """Car damage dataset loader"""
    def __init__(self, annotations_df, image_dir, mask_dir, transform=None, multi_class=True, image_size=(224, 224)):
        self.annotations = annotations_df
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.multi_class = multi_class
        self.image_size = image_size
        
        # If multi-class mode, group by image
        if self.multi_class:
            self.image_groups = annotations_df.groupby('image_file')
            self.unique_images = list(self.image_groups.groups.keys())
        
    def __len__(self):
        if self.multi_class:
            return len(self.unique_images)
        return len(self.annotations)
    
    def __getitem__(self, idx):
        if self.multi_class:
            # Get all annotations for this image
            img_file = self.unique_images[idx]
            img_annotations = self.annotations[self.annotations['image_file'] == img_file]
            
            # Get file paths
            mask_file = img_annotations.iloc[0]['mask_file']
            img_path = os.path.join(self.image_dir, img_file)
            mask_path = os.path.join(self.mask_dir, mask_file)
            
            # Load image and mask
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            # Resize image and mask to the same dimensions
            image = cv2.resize(image, self.image_size)
            mask = cv2.resize(mask, self.image_size)
            
            # Create multi-label target (one-hot encoding)
            target = torch.zeros(len(DAMAGE_CLASSES))
            for _, row in img_annotations.iterrows():
                class_idx = DAMAGE_CLASSES.index(row['damage_class'])
                target[class_idx] = 1.0
        else:
            # Single class mode - get file paths
            img_file = self.annotations.iloc[idx]['image_file']
            mask_file = self.annotations.iloc[idx]['mask_file']
            img_path = os.path.join(self.image_dir, img_file)
            mask_path = os.path.join(self.mask_dir, mask_file)
            
            # Load image and mask
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            # Resize image and mask to the same dimensions
            image = cv2.resize(image, self.image_size)
            mask = cv2.resize(mask, self.image_size)
            
            # Get class label
            class_name = self.annotations.iloc[idx]['damage_class']
            class_idx = DAMAGE_CLASSES.index(class_name)
            target = class_idx
        
        # Combine image and mask (4 channels)
        if mask is not None:
            # Create a 4-channel image (RGB + mask)
            mask_normalized = mask / 255.0 if mask.max() > 1 else mask
            image_with_mask = np.dstack((image, mask_normalized))
        else:
            # Use zeros for mask if it's not available
            image_with_mask = np.dstack((image, np.zeros((image.shape[0], image.shape[1]))))
        
        # Apply transforms
        if self.transform:
            # Convert to tensor and normalize
            image_tensor = self.transform(image_with_mask)
        else:
            # Default conversion
            image_tensor = torch.from_numpy(image_with_mask.transpose(2, 0, 1)).float() / 255.0
        
        return image_tensor, target

class FourChannelResNet(nn.Module):
    """Modified ResNet to handle 4-channel input (RGB + mask) with multi-label output"""
    def __init__(self, num_classes=len(DAMAGE_CLASSES), multi_class=True):
        super(FourChannelResNet, self).__init__()
        # Load pretrained ResNet
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # Replace first conv layer to accept 4 channels instead of 3
        original_layer = self.model.conv1
        self.model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Initialize the first 3 channels with pretrained weights
        with torch.no_grad():
            self.model.conv1.weight[:, :3] = original_layer.weight
            # Initialize the mask channel with random weights
            n = self.model.conv1.kernel_size[0] * self.model.conv1.kernel_size[1] * self.model.conv1.out_channels
            self.model.conv1.weight[:, 3] = torch.randn(64, 7, 7) * np.sqrt(2. / n)
        
        # Replace the final fully connected layer
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        
        # For multi-class (multi-label) classification, use sigmoid
        self.multi_class = multi_class
        
    def forward(self, x):
        features = self.model(x)
        if self.multi_class:
            return torch.sigmoid(features)  # For multi-label
        else:
            return features  # For single-label (use with CrossEntropyLoss)

def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=10):
    """Train the model"""
    model = model.to(device)
    
    best_loss = float('inf')
    best_model_weights = None
    
    # Training history
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            
            running_loss = 0.0
            
            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Track statistics
                running_loss += loss.item() * inputs.size(0)
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            history[f'{phase}_loss'].append(epoch_loss)
            
            print(f'{phase} Loss: {epoch_loss:.4f}')
            
            # Deep copy the model if it's the best validation loss
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_weights = model.state_dict().copy()
        
        print()
    
    # Load best model weights
    if best_model_weights:
        model.load_state_dict(best_model_weights)
    
    return model, history

def evaluate_model(model, dataloader, device, threshold=0.5):
    """Evaluate the multi-label model on a dataset"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.cpu().numpy()
            
            outputs = model(inputs)
            preds = (outputs.cpu().numpy() > threshold).astype(int)
            
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics per class
    results = {}
    for i, class_name in enumerate(DAMAGE_CLASSES):
        class_preds = all_preds[:, i]
        class_labels = all_labels[:, i]
        
        # Skip if no examples in validation set
        if np.sum(class_labels) == 0:
            continue
        
        # Calculate metrics
        tp = np.sum((class_preds == 1) & (class_labels == 1))
        fp = np.sum((class_preds == 1) & (class_labels == 0))
        tn = np.sum((class_preds == 0) & (class_labels == 0))
        fn = np.sum((class_preds == 0) & (class_labels == 1))
        
        # Precision, recall, f1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': int(np.sum(class_labels))
        }
    
    # Overall metrics
    exact_match_ratio = np.mean(np.all(all_preds == all_labels, axis=1))
    
    return {
        'class_metrics': results,
        'exact_match_ratio': exact_match_ratio
    }

def prepare_datasets(annotations_csv, image_dir, mask_dir, test_size=0.2):
    """Prepare training and validation datasets"""
    # Read annotations
    df = pd.read_csv(annotations_csv)
    
    # Check class distribution
    class_counts = df['damage_class'].value_counts()
    print("Class distribution in annotations:")
    for cls, count in class_counts.items():
        print(f"  {cls}: {count}")
    
    # Get unique images
    unique_images = df['image_file'].unique()
    
    # Split at the image level to avoid data leakage
    train_images, val_images = train_test_split(
        unique_images, test_size=test_size, random_state=42
    )
    
    # Create train and validation dataframes
    train_df = df[df['image_file'].isin(train_images)]
    val_df = df[df['image_file'].isin(val_images)]
    
    # Print dataset stats
    print(f"Total annotations: {len(df)}")
    print(f"Unique images: {len(unique_images)}")
    print(f"Training images: {len(train_images)}")
    print(f"Validation images: {len(val_images)}")
    print(f"Training annotations: {len(train_df)}")
    print(f"Validation annotations: {len(val_df)}")
    
    # Set a fixed image size for ResNet
    image_size = (224, 224)
    
    # Create transformations - simpler transforms for small dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406, 0.5], [0.229, 0.224, 0.225, 0.5])
    ])
    
    # Create datasets
    train_dataset = CarDamageDataset(train_df, image_dir, mask_dir, transform, multi_class=True, image_size=image_size)
    val_dataset = CarDamageDataset(val_df, image_dir, mask_dir, transform, multi_class=True, image_size=image_size)
    
    # Create dataloaders with smaller batch size for small dataset
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)
    
    return {
        'train': train_loader,
        'val': val_loader
    }

def auto_annotate(model, image_dir, mask_dir, output_csv, device, threshold=0.3, max_images=50):
    """Automatically annotate images using the trained model"""
    model.eval()
    
    # List all images in the directory
    all_images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if len(all_images) > max_images:
        print(f"Limiting to {max_images} images")
        all_images = all_images[:max_images]
    
    # Create transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406, 0.5], [0.229, 0.224, 0.225, 0.5])
    ])
    
    # Store annotations
    annotations = []
    
    # Fixed image size for model input (must match training)
    image_size = (224, 224)
    
    # Process each image
    print(f"Annotating {len(all_images)} images...")
    for i, img_file in enumerate(all_images):
        if i % 10 == 0:
            print(f"  Processing image {i+1}/{len(all_images)}")
        
        # Get file paths
        mask_name = os.path.splitext(img_file)[0] + '.png'
        img_path = os.path.join(image_dir, img_file)
        mask_path = os.path.join(mask_dir, mask_name)
        
        if not os.path.exists(mask_path):
            print(f"  Skipping {img_file} - no mask found")
            continue
        
        # Load image and mask
        image = cv2.imread(img_path)
        if image is None:
            print(f"  Error loading {img_file}")
            continue
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Resize to fixed dimensions
        image = cv2.resize(image, image_size)
        mask = cv2.resize(mask, image_size)
        
        # Create 4-channel input
        mask_normalized = mask / 255.0 if mask.max() > 1 else mask
        image_with_mask = np.dstack((image, mask_normalized))
        
        # Convert to tensor and normalize
        x = transform(image_with_mask).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(x)
            probs = outputs.cpu().numpy()[0]
            preds = (probs > threshold).astype(int)
        
        # Add annotations for detected classes
        annotation_idx = 0
        for cls_idx, is_present in enumerate(preds):
            if is_present:
                annotations.append({
                    'image_file': img_file,
                    'mask_file': mask_name,
                    'damage_class': DAMAGE_CLASSES[cls_idx],
                    'confidence': float(probs[cls_idx]),
                    'annotation_idx': annotation_idx
                })
                annotation_idx += 1
    
    # Create DataFrame and save
    if annotations:
        df = pd.DataFrame(annotations)
        df.to_csv(output_csv, index=False)
        print(f"Saved {len(annotations)} annotations across {df['image_file'].nunique()} images to {output_csv}")
        
        # Create class distribution summary
        class_counts = df['damage_class'].value_counts()
        print("\nClass distribution in auto-annotations:")
        for cls, count in class_counts.items():
            print(f"  {cls}: {count}")
        
        # Create a summary file
        output_dir = os.path.dirname(output_csv)
        annotation_counts = df.groupby('image_file').size().reset_index(name='annotation_count')
        summary = {
            'total_annotated_images': len(annotation_counts),
            'total_annotations': len(annotations),
            'average_annotations_per_image': annotation_counts['annotation_count'].mean(),
            'class_distribution': df['damage_class'].value_counts().to_dict()
        }
        
        summary_path = os.path.join(output_dir, 'auto_annotation_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
    else:
        print("No annotations generated!")

def main():
    parser = argparse.ArgumentParser(description="Train a few-shot learning model on car damage annotations")
    parser.add_argument('--train_csv', type=str, default="cardd_sod/manual_annotations.csv",
                        help="Path to the training annotations CSV file")
    parser.add_argument('--image_dir', type=str, default="CarDD_release/CarDD_SOD/CarDD-TR/CarDD-TR-Image",
                        help="Directory containing training images")
    parser.add_argument('--mask_dir', type=str, default="CarDD_release/CarDD_SOD/CarDD-TR/CarDD-TR-Mask",
                        help="Directory containing training masks")
    parser.add_argument('--auto_annotate_dir', type=str, default="CarDD_release/CarDD_SOD/CarDD-TE/CarDD-TE-Image",
                        help="Directory containing images to auto-annotate")
    parser.add_argument('--auto_annotate_mask_dir', type=str, default="CarDD_release/CarDD_SOD/CarDD-TE/CarDD-TE-Mask",
                        help="Directory containing masks for auto-annotation")
    parser.add_argument('--output_csv', type=str, default="cardd_sod/auto_annotations.csv",
                        help="Output CSV file for auto-annotations")
    parser.add_argument('--output_dir', type=str, default="model_output",
                        help="Output directory for model and results")
    parser.add_argument('--epochs', type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                        help="Learning rate")
    parser.add_argument('--threshold', type=float, default=0.3,
                        help="Confidence threshold for predictions")
    parser.add_argument('--max_images', type=int, default=50,
                        help="Maximum number of images to auto-annotate")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Prepare datasets
    dataloaders = prepare_datasets(args.train_csv, args.image_dir, args.mask_dir)
    
    # Create model
    model = FourChannelResNet(num_classes=len(DAMAGE_CLASSES), multi_class=True)
    criterion = nn.BCELoss()  # Binary Cross Entropy for multi-label
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    
    # Train model
    print("\nTraining model...")
    trained_model, history = train_model(
        model, dataloaders, criterion, optimizer, device, args.epochs
    )
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(args.output_dir, f"car_damage_model_{timestamp}.pth")
    torch.save(trained_model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_results = evaluate_model(trained_model, dataloaders['val'], device, args.threshold)
    
    # Print evaluation results
    print(f"Exact match ratio: {val_results['exact_match_ratio']:.4f}")
    print("Class metrics:")
    for cls, metrics in val_results['class_metrics'].items():
        print(f"  {cls}: F1={metrics['f1']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, Support={metrics['support']}")
    
    # Save validation results
    val_results_path = os.path.join(args.output_dir, f"val_results_{timestamp}.json")
    with open(val_results_path, 'w') as f:
        json.dump(val_results, f, indent=2)
    
    # Auto-annotate new images
    print("\nAuto-annotating new images...")
    auto_annotate(
        trained_model, 
        args.auto_annotate_dir,
        args.auto_annotate_mask_dir,
        args.output_csv,
        device,
        args.threshold,
        args.max_images
    )

if __name__ == "__main__":
    main() 