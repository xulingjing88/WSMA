import torch.nn as nn
import torch
from torch.autograd import Variable
from PIL import Image
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from models.model import MODEL
import argparse
from utils.evaluation import cal_kl, cal_sim, cal_nss

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='data_path')
parser.add_argument('--phase', type=str, default='test')
parser.add_argument("--divide", type=str, default="Unseen") #"Seen" or "Unseen" or "HICO-IIF"
parser.add_argument("--model_path", type=str, default="save_models_path") # the model weight path
parser.add_argument("--crop_size", type=int, default=224)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument('--threshold', type=float, default='0.2')
# parser.add_argument("--init_weights", type=bool, default=False)
parser.add_argument('--num_workers', type=int, default=1)
args = parser.parse_args()



def normalize_map(atten_map):
    min_val = np.min(atten_map)
    max_val = np.max(atten_map)
    atten_norm = (atten_map - min_val) / (max_val - min_val + 1e-10)

    return atten_norm


if args.divide == "Seen":
    args.num_classes = 36
    aff_list = ['beat', "boxing", "brush_with", "carry", "catch", "cut", "cut_with", "drag", 'drink_with',
                "eat", "hit", "hold", "jump", "kick", "lie_on", "lift", "look_out", "open", "pack", "peel",
                "pick_up", "pour", "push", "ride", "sip", "sit_on", "stick", "stir", "swing", "take_photo",
                "talk_on", "text_on", "throw", "type_on", "wash", "write"]
elif args.divide=="Unseen":
    aff_list = ["carry", "catch", "cut", "cut_with", 'drink_with',
                     "eat", "hit", "hold", "jump", "kick", "lie_on", "open", "peel",
                     "pick_up", "pour", "push", "ride", "sip", "sit_on", "stick",
                     "swing", "take_photo", "throw", "type_on", "wash"]
else: # HICO-IIF
    aff_list = ['cut_with', 'drink_with', 'hold', 'open', 'pour', 'sip', 'stick', 'stir', 'swing', 'type_on']
    
args.test_root = os.path.join(args.data_root, args.divide, "testset", "egocentric")
args.mask_root = os.path.join(args.data_root, args.divide, "testset", "GT")

model, par = MODEL(args, num_classes=len(aff_list), pretrained=False)
model.load_state_dict(torch.load(args.model_path))
model.eval()
model.cuda()

import datatest

testset = datatest.TrainData(egocentric_root=args.test_root, crop_size=args.crop_size, divide=args.divide, mask_root=args.mask_root)
MyDataLoader = torch.utils.data.DataLoader(dataset=testset,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=args.num_workers,
                                               pin_memory=True)

dict_1 = {}
for step, (image, label, mask_path, name) in enumerate(MyDataLoader):
    label = label.cuda(non_blocking=True)
    image = image.cuda(
        non_blocking=True)
    cam, cam1 = model.get(image, label, name)
    cam = cam[0].cpu().detach().numpy()
    cam1 = cam1[0].cpu().detach().numpy()
    cam = normalize_map(cam)
    cam1 = normalize_map(cam1)
    cam1[cam <args.threshold]=0
    final_cam = normalize_map(cam+cam1)

    names = mask_path[0].split("/")
    key = names[-3] + "_" + names[-2] + "_" + names[-1]
    dict_1[key] = final_cam
torch.save(dict_1, args.divide + "_preds.t7")

masks = torch.load("Unseen_gt.t7")# GT_path
preds = torch.load(args.divide + "_preds.t7")

KLs = []
SIMs = []
NSSs = []

for key in masks.keys():
    gt = masks[key]
    mask = gt / 255.0
    pred = preds[key]
    pred = cv2.resize(pred, (224, 224))
    mask = cv2.resize(mask, (224, 224))
    pred = (pred - pred.min()) / (pred.max() - pred.min())
    kld = cal_kl(pred, mask)

    sim = cal_sim(pred, mask)
    nss = cal_nss(pred, mask)

    KLs.append(kld)
    SIMs.append(sim)

    NSSs.append(nss)
print("kld = " + str(sum(KLs) / len(KLs)))
print("sim = " + str(sum(SIMs) / len(SIMs)))
print("nss = " + str(sum(NSSs) / len(NSSs)))
