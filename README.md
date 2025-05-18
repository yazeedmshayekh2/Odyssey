# Car Damage Detection and Analysis System

This project uses computer vision and machine learning techniques to detect, classify, and analyze car damage from images. It provides tools for both automated and manual annotation, few-shot learning for damage classification, and comprehensive reporting capabilities.

## Features

- Car damage detection and segmentation using Mask R-CNN
- Few-shot learning for damage type classification using ResNet with a 4-channel input (RGB + mask)
- Multi-label classification for detecting multiple damage types in one image
- Interactive annotation tools for manual labeling
- Damage severity assessment and visual reporting
- Automated annotation of test sets based on trained models
- Web-based report generation and visualization
- Side-by-side comparison of original and processed images

## Damage Types

The system classifies damage into 7 categories:

1. **Dent**: Deformation of the vehicle's external panels
2. **Scratch**: Surface abrasions affecting the paint layer
3. **Crack**: Structural fractures in panels, glass, or plastic components
4. **Glass Shatter**: Broken or fragmented glass components
5. **Lamp Broken**: Damage to headlights, taillights, or signal lamps
6. **Tire Flat**: Deflated or damaged tires
7. **No Damage**: Areas without detectable damage

## Dataset Structure

The project uses the CarDD dataset, which is organized in both COCO and SOD formats:

```
CarDD_release/
├── CarDD_COCO/
│   ├── annotations/
│   ├── train2017/
│   ├── val2017/
│   └── test2017/
└── CarDD_SOD/
    ├── CarDD-TR/
    │   ├── CarDD-TR-Image/ # Training images
    │   ├── CarDD-TR-Mask/  # Training masks
    ├── CarDD-VAL/          # Validation data
    └── CarDD-TE/           # Test data
        ├── CarDD-TE-Image/ # Test images
        └── CarDD-TE-Mask/  # Test masks
```

## Prerequisites

- Python 3.8+
- PyTorch >= 1.8.0
- Detectron2 (for segmentation)
- OpenCV >= 4.5.0
- scikit-learn >= 0.24.0
- NumPy, Pandas, Matplotlib

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/car-damage-detection.git
cd car-damage-detection

# Create and activate a virtual environment
python -m venv car_damage_env
source car_damage_env/bin/activate  # On Windows: car_damage_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Manual Annotation

To manually annotate car damage images:

```bash
python annotate_cardd_sod.py --image_dir CarDD_release/CarDD_SOD/CarDD-TR/CarDD-TR-Image --mask_dir CarDD_release/CarDD_SOD/CarDD-TR/CarDD-TR-Mask --output_csv cardd_sod/manual_annotations.csv
```

The annotation tool provides the following controls:
- A/D: Navigate between images
- W/S: Cycle through damage types
- Space: Add annotation for current damage type
- R: Remove last annotation
- Q: Quit and save annotations

### Training a Few-Shot Model

Train a damage classifier using the manually annotated examples:

```bash
python train_fewshot_model.py --train_csv cardd_sod/manual_annotations.csv --image_dir CarDD_release/CarDD_SOD/CarDD-TR/CarDD-TR-Image --mask_dir CarDD_release/CarDD_SOD/CarDD-TR/CarDD-TR-Mask --epochs 15
```

The model uses a modified ResNet18 architecture with a 4-channel input (RGB + mask) and is optimized for few-shot learning scenarios.

### Annotating Test Images with Model Assistance

Use the trained model to assist in annotating test images:

```bash
python annotate_test_set.py --model_path model_output/car_damage_model_*.pth --image_dir CarDD_release/CarDD_SOD/CarDD-TE/CarDD-TE-Image --mask_dir CarDD_release/CarDD_SOD/CarDD-TE/CarDD-TE-Mask --output_csv cardd_sod/test_annotations.csv
```

This tool shows model predictions for each image and allows manual correction.

### Automated Test Set Annotation

To automatically annotate test images using the trained model:

```bash
python train_fewshot_model.py --train_csv cardd_sod/manual_annotations.csv --auto_annotate_dir CarDD_release/CarDD_SOD/CarDD-TE/CarDD-TE-Image --auto_annotate_mask_dir CarDD_release/CarDD_SOD/CarDD-TE/CarDD-TE-Mask --output_csv cardd_sod/auto_annotations.csv --model_path model_output/car_damage_model_*.pth
```

### Generating Damage Reports

Create a visual report of car damage analysis:

```bash
# Prepare deployment files
python prepare_deployment.py

# The deployment folder can be hosted using GitHub Pages, Netlify, or other static hosting services
```

## Model Architecture

### Few-Shot Learning Approach

The system uses a transfer learning approach with the following characteristics:
- Base model: ResNet18 pretrained on ImageNet
- Input modification: 4-channel input (RGB + segmentation mask)
- Output layer: Multi-label classification with sigmoid activation
- Loss function: Binary Cross-Entropy for multi-label classification
- Optimization: Adam optimizer with reduced learning rate (0.0005)
- Regularization: Weight decay to prevent overfitting on small datasets

### Damage Assessment Methodology

Damage severity is calculated by considering:
1. The damage type (some types like glass shatter are inherently more severe)
2. The affected area (calculated from the mask)
3. The number of different damage types present
4. The model's confidence in the damage classification

## Project Structure

```
.
├── train_fewshot_model.py      # Training script for few-shot learning
├── annotate_cardd_sod.py       # Manual annotation tool
├── annotate_test_set.py        # Test set annotation with model assistance
├── prepare_deployment.py       # Script to prepare deployment files
├── requirements.txt            # Project dependencies
├── cardd_sod/                  # Directory for annotations
│   └── manual_annotations.csv  # Manually created annotations
├── model_output/               # Saved models and results
└── deployment/                 # Deployment-ready files
```

## Web Report Interface

The web-based report interface provides:
- Interactive visualization of damage analysis
- Side-by-side comparison of original and processed images
- Damage type and severity information
- Charts showing damage type distribution
- Manual editing capability for damage types
- Responsive design for desktop and mobile viewing

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- CarDD_SOD dataset creators for providing the dataset
- PyTorch team for the deep learning framework
- OpenCV community for computer vision utilities 