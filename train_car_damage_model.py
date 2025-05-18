#!/usr/bin/env python
# Car damage detection and segmentation using Detectron2 with ResNet-50 backbone

import os
import json
import numpy as np
from datetime import datetime

# Check if Detectron2 is installed, if not install it
try:
    import detectron2
except ImportError:
    print("Detectron2 is not installed. Installing...")
    # Install dependencies
    import subprocess
    import os
    
    # Check if requirements.txt exists
    if os.path.exists("requirements.txt"):
        print("Installing dependencies from requirements.txt...")
        subprocess.check_call(["pip", "install", "-r", "requirements.txt"])
    else:
        print("requirements.txt not found, installing dependencies individually...")
        subprocess.check_call(["pip", "install", "torch", "torchvision", "opencv-python", "cython", "matplotlib", "pycocotools", "numpy", "tqdm", "pillow", "pyyaml"])
    
    subprocess.check_call(["pip", "install", "detectron2", "-f", "https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/index.html"])

# Import Detectron2 libraries
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.model_zoo import model_zoo
from detectron2.evaluation import COCOEvaluator

# Define paths
DATASET_ROOT = "CarDD_release/CarDD_COCO"
TRAIN_JSON = f"{DATASET_ROOT}/annotations/instances_train2017.json"
VAL_JSON = f"{DATASET_ROOT}/annotations/instances_val2017.json"
TEST_JSON = f"{DATASET_ROOT}/annotations/instances_test2017.json"
TRAIN_DIR = f"{DATASET_ROOT}/train2017"
VAL_DIR = f"{DATASET_ROOT}/val2017"
TEST_DIR = f"{DATASET_ROOT}/test2017"
OUTPUT_DIR = "output/car_damage_detection"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Register the datasets
register_coco_instances("car_damage_train", {}, TRAIN_JSON, TRAIN_DIR)
register_coco_instances("car_damage_val", {}, VAL_JSON, VAL_DIR)
register_coco_instances("car_damage_test", {}, TEST_JSON, TEST_DIR)

# Class metadata
classes = ["dent", "scratch", "crack", "glass shatter", "lamp broken", "tire flat"]
MetadataCatalog.get("car_damage_train").thing_classes = classes
MetadataCatalog.get("car_damage_val").thing_classes = classes
MetadataCatalog.get("car_damage_test").thing_classes = classes

class CocoTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

def setup_cfg():
    # Configure the model
    cfg = get_cfg()
    
    # Load ResNet-50 FPN Mask R-CNN config from model zoo
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    
    # Set dataset params
    cfg.DATASETS.TRAIN = ("car_damage_train",)
    cfg.DATASETS.TEST = ("car_damage_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    
    # Load model weights from model zoo
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    
    # Model settings
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6  # 6 damage classes
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    
    # Training parameters
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 5000
    cfg.SOLVER.STEPS = []  # Learning rate decay
    
    # Checkpoint saving settings
    cfg.OUTPUT_DIR = OUTPUT_DIR
    
    return cfg

def main():
    # Setup configuration
    cfg = setup_cfg()
    
    # Print training info
    print(f"Starting training with ResNet-50 backbone for car damage detection")
    print(f"Dataset: {DATASET_ROOT}")
    print(f"Classes: {classes}")
    print(f"Training iterations: {cfg.SOLVER.MAX_ITER}")
    print(f"Base learning rate: {cfg.SOLVER.BASE_LR}")
    print(f"Output directory: {cfg.OUTPUT_DIR}")
    
    # Create trainer
    trainer = CocoTrainer(cfg)
    trainer.resume_or_load(resume=False)
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main() 