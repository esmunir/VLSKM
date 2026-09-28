import os
from .model_zoo import dedode_detector_B, dedode_detector_L, dedode_descriptor_B, dedode_descriptor_G
DEBUG_MODE = False
RANK = int(os.environ.get('RANK', default = 0))
GLOBAL_STEP = 0
STEP_SIZE = 1
LOCAL_RANK = -1
