# -*- coding: utf-8 -*-
r"""
YOLOX-S Hand Detector

Dataset root:
<PROJECT_ROOT>/data/hand_detection

Expected structure:
data/hand_detection/
├─ annotations/
│  ├─ instances_train2017.json
│  └─ instances_val2017.json
├─ train2017/
└─ val2017/
"""

from pathlib import Path
from yolox.exp import Exp as MyExp


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Exp(MyExp):
    def __init__(self):
        super().__init__()

        # Model
        self.num_classes = 1
        self.depth = 0.33
        self.width = 0.50

        # Dataset
        self.data_dir = str(
            PROJECT_ROOT / "data" / "hand_detection"
        )

        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"

        self.train_name = "train2017"
        self.val_name = "val2017"

        # Input
        self.input_size = (640, 640)
        self.test_size = (640, 640)
        self.multiscale_range = 0

        # Augmentation
        self.degrees = 5.0
        self.translate = 0.05
        self.mosaic_scale = (0.8, 1.2)

        self.enable_mixup = False
        self.mixup_prob = 0.0

        self.flip_prob = 0.5
        self.hsv_prob = 0.5
        self.mosaic_prob = 0.5

        # Training
        self.max_epoch = 100
        self.warmup_epochs = 5
        self.no_aug_epochs = 10

        self.data_num_workers = 2
        self.eval_interval = 1
        self.save_history_ckpt = False

        # Evaluation / inference
        self.test_conf = 0.20
        self.nmsthre = 0.70

        # Output
        self.exp_name = "yolox_s_hand_tight"