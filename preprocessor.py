import os
import cv2
import json
import math
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from facenet_pytorch import MTCNN

#need to check GPU status for that we will use torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device : {}' .format(device))
