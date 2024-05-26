import os
from torch.utils import data
import collections
import torch
import numpy as np
import h5py
from joblib import Parallel, delayed
import random
from utils import util
import cv2
from utils.transform import Normalize, cv_random_crop_flip, load_image
from torchvision import transforms
from PIL import Image

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


class TrainData(data.Dataset):
    def __init__(self, egocentric_root, crop_size=224, divide="Unseen", mask_root=None):
        self.egocentric_root = egocentric_root
        self.image_list = []
        self.crop_size = crop_size
        self.mask_root = mask_root
        if divide == "Seen":
            self.aff_list = ['beat', "boxing", "brush_with", "carry", "catch",
                             "cut", "cut_with", "drag", 'drink_with', "eat",
                             "hit", "hold", "jump", "kick", "lie_on", "lift",
                             "look_out", "open", "pack", "peel", "pick_up",
                             "pour", "push", "ride", "sip", "sit_on", "stick",
                             "stir", "swing", "take_photo", "talk_on", "text_on",
                             "throw", "type_on", "wash", "write"]
        elif divide=="Unseen":
            self.aff_list = ["carry", "catch", "cut", "cut_with", 'drink_with',
                             "eat", "hit", "hold", "jump", "kick", "lie_on", "open", "peel",
                             "pick_up", "pour", "push", "ride", "sip", "sit_on", "stick",
                             "swing", "take_photo", "throw", "type_on", "wash"]
        else: # HICO-IIF
            self.aff_list = ['cut_with', 'drink_with', 'hold', 'open', 'pour', 'sip', 'stick', 'stir', 'swing', 'type_on']

        self.transform = transforms.Compose([
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        files = os.listdir(self.egocentric_root)
        for file in files:
            file_path = os.path.join(self.egocentric_root, file)
            obj_files = os.listdir(file_path)
            for obj_file in obj_files:
                obj_file_path = os.path.join(file_path, obj_file)
                images = os.listdir(obj_file_path)
                for img in images:
                    if 'json' not in img:
                        img_path = os.path.join(obj_file_path, img)
                        cur = img_path.split("/")
                        if os.path.exists(os.path.join(self.mask_root, file, obj_file, img[:-3] + "png")):
                            self.image_list.append(img_path)

    def __getitem__(self, item):
        egocentric_image_path = self.image_list[item]
        names = egocentric_image_path.split("/")
        aff_name, object = names[-3], names[-2]
        label = self.aff_list.index(aff_name)
        egocentric_image = self.load_static_image(egocentric_image_path)  # At this time, the graph of individual items has been converted into tensor

        mask_path = os.path.join(self.mask_root, names[-3], names[-2], names[-1][:-3] + "png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (224, 224))

        return egocentric_image, label, mask_path, aff_name

    def load_static_image(self, path):

        img = util.load_img(path)
        img = self.transform(img)
        return img

    def __len__(self):

        return len(self.image_list)
