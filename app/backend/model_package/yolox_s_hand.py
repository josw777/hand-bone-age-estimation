# -*- coding: utf-8 -*-
"""YOLOX-S 1-class hand detector inference config."""

import os
from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super().__init__()

        self.num_classes = 1

        # YOLOX-S
        self.depth = 0.33
        self.width = 0.50

        self.input_size = (640, 640)
        self.test_size = (640, 640)

        self.test_conf = 0.20
        self.nmsthre = 0.70

        self.exp_name = os.path.splitext(
            os.path.basename(__file__)
        )[0]
