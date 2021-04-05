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

fp=open('metadata.json',)
data=json.load(fp)

if not os.path.exists('data'):
    os.makedirs('data')

if not os.path.exists("data/train"):
    os.makedirs("data/train")

if not os.path.exists("data/train/real"):
    os.makedirs("data/train/real")

if not os.path.exists("data/train/fake"):
    os.makedirs("data/train/fake")

if not os.path.exists('data/val'):
    os.makedirs('data/val')

if not os.path.exists('data/val/real'):
    os.makedirs('data/val/real')

if not os.path.exists('data/val/fake'):
    os.makedirs('data/val/fake')