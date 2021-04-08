import torch
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from facenet_pytorch import MTCNN
import cv2
from tqdm import tqdm
import os
import copy
import math

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))


# function to load the ResNet-50 model with the trained weights in checkpoint file

def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model_conv = torchvision.models.resnet50(pretrained=True)

    for param in model_conv.parameters():
        param.requires_grad = False

    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 2)

    # loading the weights
    model_conv.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model_conv.eval()
    return model_conv

def applyTransforms(inp):
    outp = transforms.functional.resize(inp, [224,224])
    outp = transforms.functional.to_tensor(outp)
    outp = transforms.functional.normalize(outp, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return outp


def myVideo(file_name, model):
    mtcnn = MTCNN(margin=20, keep_all=True, post_process=False, device=device)
    # Load video
    v_cap = cv2.VideoCapture(file_name)
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Loop through video, taking some no of frames to form a batch   (here, every 30th frame)
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

        # Add to batch
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))

    # detect faces in frames &  saving frames to file
    f = "test/test_frames" + "/"
    frames_paths = [f + 'image' + str(i) + '.jpg' for i in range(len(frames))]
    faces = mtcnn(frames, save_path=frames_paths)

    path = os.listdir("test/test_frames")
    vals_f = []
    vals_r = []
    for x in path:
        img = Image.open("test/test_frames" + "/" + x)
        imageTensor = applyTransforms(img)
        minibatch = torch.stack([imageTensor])
        # model_conv(minibatch)
        softMax = nn.Softmax(dim=1)
        preds = softMax(model(minibatch))
        vals_f.append(preds[0, 0].item())
        vals_r.append(preds[0, 1].item())

    av = sum(vals_f) / len(path)
    print("average probability of fakeness:", av)
    print('Percentage of fakeness: {:.4f}'.format(av * 100))

    av = sum(vals_r) / len(path)
    print("average probability of realness:", av)
    print('Percentage of realness: {:.4f}'.format(av * 100))


def testing(name, model):
    # import os
    f = "test/test_frames"
    reqd = os.listdir(f)

    if len(reqd) != 0:
        for i in reqd:
            os.remove(f + "/" + i)

    path = "/user/test/test_videos" + "/" + name
    myVideo(path, model)

    import json
    fp = open(r"\user\test\metadata.json", )
    data = json.load(fp)
    path = name
    # print('The true label is:',data[path]['label'])
    fp.close()

model_conv=load_checkpoint("checkpoint.pth")

testing('funny_deepfake.mp4',model_conv)

