import sys
import argparse

import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms

from retinanet import inference as inferencemodel
from torch.utils.data import DataLoader

import cv2
import imutils
from imutils import paths

print('CUDA available: {}'.format(torch.cuda.is_available()))

parser = argparse.ArgumentParser(description='Inference a RetinaNet network.')
parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
parser.add_argument('--classes', help='number of classes', type=int, default=80)
parser.add_argument('--model', help='Model to load', required=True)
parser.add_argument('--outname', help='Name of output script', required=True)
parser = parser.parse_args()

if parser.depth == 18:
    retinanet = inferencemodel.resnet18(num_classes=parser.classes)
elif parser.depth == 34:
    retinanet = inferencemodel.resnet34(num_classes=parser.classes)
elif parser.depth == 50:
    retinanet = inferencemodel.resnet50(num_classes=parser.classes)
elif parser.depth == 101:
    retinanet = inferencemodel.resnet101(num_classes=parser.classes)
elif parser.depth == 152:
    retinanet = inferencemodel.resnet152(num_classes=parser.classes)
else:
    raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')


statedict = torch.load(parser.model, map_location=torch.device('cpu'))
retinanet.load_state_dict(statedict)
retinanet.eval()
print("Starting conversion")
traced = torch.jit.script(retinanet)
print(traced)
traced.save(parser.outname)
