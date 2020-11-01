import torch
import torch.nn as nn
import torch.nn.functional as F

from apex import amp

import argparse
from torch.autograd import Variable
import numpy as np
from PIL import Image

import PIL
import os
from tqdm import tqdm
import cv2
import time
import pandas as pd
import torchvision
import torchvision.transforms as transforms
import sys

from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
from tqdm import tqdm_notebook , tnrange

import torch.utils.model_zoo as model_zoo
import torchvision.models as models
from torchvision import datasets
from tqdm.notebook import *
from torchcontrib.optim import SWA

import time
import matplotlib.pyplot as plt
import copy
import torch.nn.functional as F

transform = transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                        ])
trans1= transforms.ToPILImage()
