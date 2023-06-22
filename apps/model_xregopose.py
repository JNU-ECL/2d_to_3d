import math
import logging
from collections import OrderedDict
from yacs.config import CfgNode as CN
from easydict import EasyDict
import os
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models as models
from smplx.body_models import SMPL
import numpy as np
from collections import OrderedDict
from geometry import rot6d_to_rotmat, projection_temp, rotation_matrix_to_angle_axis, create_euler
import torch.utils.model_zoo as model_zoo

class xR_egoposemodel(nn.Module):
    pass

class Resnet_101():
    pass