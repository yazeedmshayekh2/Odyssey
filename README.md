# Car Damage Detection and Segmentation

This project uses Detectron2 with a ResNet-50 backbone to detect and segment car damage in images. The model is trained to identify 6 types of damage:

1. Dent
2. Scratch
3. Crack
4. Glass shatter
5. Lamp broken
6. Tire flat

## Features

- Car damage detection and segmentation using Mask R-CNN
- Damage severity and repair cost estimation
- Automatic annotation generation for SOD dataset format
- Detailed damage analysis with confidence scores and bounding boxes
- Support for both COCO and SOD dataset formats

## Dataset

The project uses the CarDD dataset, which is organized in both COCO and SOD formats. The dataset structure is:

```
CarDD_release/
├── CarDD_COCO/
│   ├── annotations/
│   ├── train2017/
│   ├── val2017/
│   └── test2017/
└── CarDD_SOD/
    ├── CarDD-TR/
    │   ├── CarDD-TR-Image/
    │   ├── CarDD-TR-Mask/
    │   └── CarDD-TR-Edge/
    ├── CarDD-VAL/
    └── CarDD-TE/
```

## Prerequisites

- Python 3.7+
- PyTorch >= 1.8.0
- Detectron2
- OpenCV >= 4.5.0
- NumPy >= 1.19.0
- Pandas >= 1.3.0
- Matplotlib >= 3.3.0

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/car-damage-detection.git
cd car-damage-detection

# Create and activate a virtual environment (optional but recommended)
python -m venv car_damage_env
source car_damage_env/bin/activate  # On Windows: car_damage_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Detectron2
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/index.html
```

## Usage

### Training the Model

To train the model on the CarDD dataset:

```bash
python train_car_damage_model.py
```

### Evaluating the Model

To evaluate the trained model and visualize results:

```bash
python evaluate_car_damage_model.py
```

### Generating Annotations

To generate annotations for the SOD dataset:

```bash
python create_sod_annotations.py
```

This script will:
- Process images from the CarDD_SOD dataset
- Generate detailed annotations including:
  - Damage class
  - Confidence scores
  - Bounding box coordinates
  - Damage percentage
  - Weighted damage scores
- Save annotations in CSV format
- Generate summary statistics in JSON format

The annotations are saved in the `cardd_sod` directory:
- `cardd_sod_annotations.csv`: Detailed annotations for each detected damage
- `annotation_summary.json`: Summary statistics including:
  - Total number of processed images
  - Total number of annotations
  - Distribution of damage classes
  - Average confidence scores
  - Average damage percentages

## Model Configuration

The model uses the following configuration:
- Architecture: Mask R-CNN
- Backbone: ResNet-50 with Feature Pyramid Network (FPN)
- Learning rate: 0.00025
- Batch size: 2
- Training iterations: 5000
- Score threshold for detection: 0.5

## Damage Analysis

### Damage Percentage Calculation

The damage percentage is calculated using a weighted approach that considers repair cost rather than just safety impact:

1. Raw area percentage - The ratio of damaged pixels to the total car area
2. Damage severity - Different types of damage have different repair cost weightings:
   - Scratch: 0.3 (inexpensive paint touch-up)
   - Dent: 0.8 (moderate panel work required)
   - Crack: 1.5 (more significant repair needed)
   - Glass shatter: 2.0 (glass replacement cost)
   - Lamp broken: 1.5 (light assembly replacement)
   - Tire flat: 0.6 (relatively inexpensive to replace tire)
3. Component importance - Based on typical repair/replacement costs:
   - Engine: 8.0 (most expensive to repair)
   - Windshield: 2.5 (expensive glass component)
   - Hood: 2.0 (large panel, expensive to replace)
   - Headlight: 1.8 (modern headlights are expensive)
   - Door: 1.5 (contains electronics, moderately expensive)
   - Taillight: 1.2 (usually less expensive than headlights)
   - Bumper: 1.2 (usually plastic, moderately priced)
   - Fender: 1.0 (relatively simple panel replacement)
   - Wheel: 1.0 (moderate cost to replace)

The formula is:
```
weighted_damage = raw_percentage * damage_severity * component_importance
```

Additional adjustments:
- Special handling for tire flat damage (capped at 25% regardless of size)
- Individual damage contributions are capped at 50%
- Total damage is capped at 100%
- Small damages (< 5% visible area) are scaled appropriately
- Very small damages (< 0.5% visible area) are further reduced

All parameters for damage calculation can be adjusted in the `damage_config.py` file.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- CarDD dataset creators
- Detectron2 team at Facebook AI Research
- PyTorch community 