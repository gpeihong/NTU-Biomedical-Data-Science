import cv2
import numpy as np
import glob
import os
import random

import torch
import torch.utils.data
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader

def get_files(path, file_type):   #Get all files in the path
    files = []
    for ext in [file_type]:
        files.extend(glob.glob(
            os.path.join(path, '*.{}'.format(ext))))
    return files


class MNIST(torch.utils.data.Dataset):

    def __init__(self, patch_size=100):
        self.patch_size = patch_size
        self.source_path = ["../mnist/train0/", "../mnist/train7/"]
        self.img_path = []
        self.img_path.append(get_files(self.source_path[0], "jpg"))
        self.img_path.append(get_files(self.source_path[1], "jpg"))

    def __getitem__(self, item):
        result = {}
        total_img_path, result['gt'] = self.get_patch()
        img = []
        for path in total_img_path:
            temp = cv2.imread(path)
            temp = temp/temp.max()
            img.append(temp)
        result['img'] = img
        
        return result


    def get_patch(self):
        result = []
        rand = random.random()   #Generate the ground truth for each package
        num = int(self.patch_size * rand)
        sample = random.sample(self.img_path[0],num)  
        sample2 = random.sample(self.img_path[1],self.patch_size-num)
        for i in sample:
            result.append(i)
        for i in sample2:
            result.append(i)
        
        """for i in range(num):
            rd1 = random.randint(0, len(self.img_path[0]) - 1)
            img_path = self.img_path[0][rd1]
            self.result.append(img_path)

        for i in range(self.patch_size-num):
            rd1 = random.randint(0, len(self.img_path[1]) - 1)
            img_path = self.img_path[1][rd1]
            self.result.append(img_path)"""
            
        
        return result, rand   #Return the bag and the corresponding ground truth


    def __len__(self):
        return self.patch_size


def collater(data):
    imgs = [i['img'] for i in data]
    gt = [i['gt'] for i in data]
    imgs = torch.tensor(imgs, dtype=torch.float)
    gt = torch.tensor(gt, dtype=torch.float)
    imgs = imgs.view(-1,imgs.size(4),imgs.size(2),imgs.size(3))
    return imgs, gt

