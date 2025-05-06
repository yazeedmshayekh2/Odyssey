import os
import json
import logging
import pandas as pd
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode, BitMasks
from detectron2.engine import DefaultTrainer, hooks
from detectron2.config import get_cfg
from detectron2.model_zoo import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_test_loader
import numpy as np
import torch
from detectron2.data import DatasetMapper, build_detection_train_loader
from detectron2.data import transforms as T
from detectron2.data.samplers import RepeatFactorTrainingSampler
from detectron2.data.build import get_detection_dataset_dicts
from detectron2.data.common import DatasetFromList, MapDataset
import copy

def get_dataset_dicts(json_file, img_dir):
    """
    Load dataset from JSON annotation file and image directory.
    Returns data in Detectron2's standard format with segmentation masks.
    """
    with open(json_file) as f:
        dataset_anns = json.load(f)
    
    dataset_dicts = []
    for idx, ann in enumerate(dataset_anns['images']):
        record = {}
        
        # Get image info
        filename = os.path.join(img_dir, ann['file_name'])
        height, width = ann['height'], ann['width']
        
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
        
        # Get annotations
        objs = []
        for annotation in dataset_anns['annotations']:
            if annotation['image_id'] == ann['id']:
                category_id = annotation['category_id'] - 1
                
                # Get bbox coordinates
                bbox = annotation['bbox']  # [x, y, w, h]
                x, y, w, h = [int(coord) for coord in bbox]
                
                # Create segmentation if not provided
                if 'segmentation' not in annotation:
                    # Create a polygon from bbox
                    segmentation = [[x, y, x+w, y, x+w, y+h, x, y+h]]
                else:
                    segmentation = annotation['segmentation']
                
                obj = {
                    "bbox": bbox,
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "segmentation": segmentation,
                    "category_id": category_id,
                    "iscrowd": annotation.get('iscrowd', 0)
                }
                objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    
    return dataset_dicts

def register_datasets():
    """
    Register the train, validation and test datasets
    """
    damage_classes = [
        "scratch",
        "dent",
        "crack",
        "glass_shatter",
        "tire_flat",
        "lamp_broken"
    ]  
    
    base_dir = "cardd"
    for d in ["train", "val", "test"]:
        DatasetCatalog.register(
            f"my_dataset_{d}", 
            lambda d=d: get_dataset_dicts(
                os.path.join(base_dir, f"{d}.json"), 
                os.path.join(base_dir, d)
            )
        )
        MetadataCatalog.get(f"my_dataset_{d}").set(thing_classes=damage_classes)

def setup_cfg():
    """
    Create a configuration for Mask R-CNN with improved settings
    """
    cfg = get_cfg()
    
    # Use Mask R-CNN config
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    
    # Model settings
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6
    
    # Loss weights for better handling of rare classes
    cfg.MODEL.ROI_HEADS.LOSS_WEIGHTS = {
        "loss_box_reg": 1.0,
        "loss_cls": 2.0,
        "loss_mask": 1.5
    }
    
    # Dataset settings
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ("my_dataset_val",)
    
    # Improved training settings
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.001  # Lower initial LR
    cfg.SOLVER.MAX_ITER = 15000  # Train longer
    cfg.SOLVER.STEPS = [8000, 12000]  # Adjust LR decay points
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000
    cfg.SOLVER.WARMUP_ITERS = 2000  # Longer warmup
    cfg.SOLVER.WARMUP_FACTOR = 0.001
    
    # Enhanced input settings
    cfg.INPUT.MIN_SIZE_TRAIN = (480, 512, 544, 576, 608, 640, 672, 704)
    cfg.INPUT.MAX_SIZE_TRAIN = 1000
    cfg.INPUT.MIN_SIZE_TEST = 640
    cfg.INPUT.MAX_SIZE_TEST = 1000
    
    # Class balancing
    cfg.DATALOADER.REPEAT_THRESHOLD = 0.001  # More aggressive resampling
    
    # Output directory
    cfg.OUTPUT_DIR = "./output"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    return cfg

def validate_dataset():
    """
    Validate that the dataset is loaded correctly
    """
    # Define base directory
    base_dir = "cardd"
    
    # Check if json files exist
    required_files = ["train.json", "val.json", "test.json"]
    for file in required_files:
        file_path = os.path.join(base_dir, file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file {file_path} not found")
    
    # Check if image directories exist
    required_dirs = ["train", "val", "test"]
    for dir in required_dirs:
        dir_path = os.path.join(base_dir, dir)
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Required directory {dir_path} not found")
    
    # Try loading one dataset to verify format
    try:
        train_dicts = get_dataset_dicts(
            os.path.join(base_dir, "train.json"), 
            os.path.join(base_dir, "train")
        )
        if len(train_dicts) == 0:
            raise ValueError("No data found in train.json")
        
        # Verify first annotation
        first_record = train_dicts[0]
        required_keys = ["file_name", "image_id", "height", "width", "annotations"]
        for key in required_keys:
            if key not in first_record:
                raise KeyError(f"Required key {key} not found in dataset records")
                
    except Exception as e:
        raise Exception(f"Error validating dataset: {str(e)}")

class BalancedDatasetMapper(DatasetMapper):
    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        # Implement class-aware sampling here
        return super().__call__(dataset_dict)

class CarDamageTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(
            dataset_name, 
            tasks=("bbox", "segm"),  # Evaluate both bbox and segmentation
            distributed=False,
            output_dir=output_folder
        )
    
    def build_hooks(self):
        hooks_list = super().build_hooks()
        hooks_list.append(
            hooks.EvalHook(
                eval_period=500,
                eval_function=lambda: self.test(
                    self.cfg,
                    self.model,
                    evaluators=[self.build_evaluator(
                        self.cfg,
                        self.cfg.DATASETS.TEST[0]
                    )]
                )
            )
        )
        return hooks_list
    
    @classmethod
    def build_train_loader(cls, cfg):
        dataset_dicts = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
        )
        
        # Calculate repeat factors based on category frequency
        repeat_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
            dataset_dicts, cfg.DATALOADER.REPEAT_THRESHOLD
        )
        
        dataset = DatasetFromList(dataset_dicts, copy=False)
        
        # Enhanced augmentation pipeline
        mapper = BalancedDatasetMapper(cfg, is_train=True, augmentations=[
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN, 
                "choice"
            ),
            T.RandomBrightness(0.4, 1.6),  # More aggressive brightness
            T.RandomContrast(0.4, 1.6),    # More aggressive contrast
            T.RandomSaturation(0.4, 1.6),  # More aggressive saturation
            T.RandomLighting(1.5),         # Increased lighting variation
            T.RandomFlip(prob=0.5),
            T.RandomRotation([-45, 45]),   # More rotation
            T.RandomCrop("relative_range", (0.2, 0.2)),  # More aggressive cropping
            T.RandomBrightness(0.4, 1.6),  # Double brightness augmentation
            T.RandomNoise(std=0.1),        # Add noise
        ])
        
        dataset = MapDataset(dataset, mapper)
        sampler = RepeatFactorTrainingSampler(repeat_factors)
        
        return build_detection_train_loader(
            cfg,
            dataset=dataset,
            sampler=sampler,
            mapper=mapper,
            total_batch_size=cfg.SOLVER.IMS_PER_BATCH,
        )
    
    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.SOLVER.BASE_LR,
            total_steps=cfg.SOLVER.MAX_ITER,
            pct_start=0.3,
            anneal_strategy='cos',
            cycle_momentum=True,
            base_momentum=0.85,
            max_momentum=0.95,
            div_factor=25.0
        )

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("detectron2")
    
    # Validate dataset
    logger.info("Validating dataset...")
    validate_dataset()
    
    # Register the datasets
    logger.info("Registering datasets...")
    register_datasets()
    
    # Setup configuration
    logger.info("Setting up configuration...")
    cfg = setup_cfg()
    
    # Log important settings
    logger.info(f"Training with:")
    logger.info(f"- Number of classes: {cfg.MODEL.ROI_HEADS.NUM_CLASSES}")
    logger.info(f"- Batch size: {cfg.SOLVER.IMS_PER_BATCH}")
    logger.info(f"- Learning rate: {cfg.SOLVER.BASE_LR}")
    logger.info(f"- Max iterations: {cfg.SOLVER.MAX_ITER}")
    
    # Create trainer and start training
    logger.info("Starting training...")
    try:
        trainer = CarDamageTrainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()
    except KeyboardInterrupt:
        # Save checkpoint on interrupt
        logger.info("Training interrupted. Saving checkpoint...")
        if trainer.checkpointer:
            trainer.checkpointer.save("model_interrupted")
        logger.info("Checkpoint saved. Exiting...")
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()
