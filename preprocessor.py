import os
import cv2
import json
import math
import torch
import shutil
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
from facenet_pytorch import MTCNN

# need to check GPU status for that we will use torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device : {}'.format(device))

fp = open('metadata.json', )
data = json.load(fp)


def create_folders():
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

    # creating validation video file
    if not os.path.exists('validation_videos'):
        os.makedirs('validation_videos')


def train_valid():
    x = (os.listdir("train_videos"))

    print(len(x))

    amt1 = 0.2 * len(x)
    amt1 = math.ceil(amt1)

    i = 1
    reqd = "train_videos"
    dest_file = ("validation_videos")

    while i != amt1:
        i = i + 1
        source = random.choice(os.listdir("train_videos"))
        src = reqd + "/" + source
        shutil.copy(src, dest_file)
        os.remove(src)


def mtcnnfun():
    mtcnn = MTCNN(margin=20, keep_all=True, post_process=False, device=device)
    return mtcnn


def create_datasets():
    d = "train_videos"  # folder where train videos are located
    model = mtcnnfun()
    for path in os.listdir(d):

        # load video
        v_cap = cv2.VideoCapture("train_videos" + '/' + path)
        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # for every video select every 30th frame

        frames = []
        for i in tqdm(range(v_len)):

            # Load frame
            success = v_cap.grab()
            if i % 30 == 0:
                success, frame = v_cap.retrieve()
            else:
                continue
            if not success:
                continue

            # add to batch
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))

        file_name = "real" if data[path]['label'] == 'REAL' else "fake"

        f = "data/train/" + file_name + "/"
        frames_paths = [f + path[0:-4] + str(i) + '.jpg' for i in range(1, len(frames) + 1)]
        faces = model(frames, save_path=frames_paths)


def validation_datasets():
    d = r"validation_videos"
    model = mtcnnfun()
    for path in os.listdir(d):

        # load video
        v_cap = cv2.VideoCapture("validation_videos" + '/' + path)
        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # for every video, select the 30th frame

        frames = []
        for i in tqdm(range(v_len)):

            # Load frame
            success = v_cap.grab()
            if i % 30 == 0:
                success, frame = v_cap.retrieve()
            else:
                continue
            if not success:
                continue

            # add to batch
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))

        file_name = "real" if data[path]['label'] == 'REAL' else "fake"

        f = r"data/val/" + file_name + "/"
        frames_paths = [f + path[0:-4] + str(i) + '.jpg' for i in range(1, len(frames) + 1)]
        faces = model(frames, save_path=frames_paths)


if __name__ == '__main__':
    create_folders()
    train_valid()
    create_datasets()
    validation_datasets()