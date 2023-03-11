### Creates ASL dataset for use in PyTorch CNN
### Assumes data is in /Kaggle/asl_alphabet_train/ and /Kaggle/asl_alphabet_test/

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import datasets

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: ", device)

import os
import pandas as pd
from torchvision.io import read_image

BATCH_SIZE = 128

class ASLDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

def get_ASL_data(augmentation=0):
  # Data augmentation transformations. Not for Testing!
  if augmentation:
    transform_train = transforms.Compose([
      transforms.RandomCrop(200, padding=10, padding_mode='edge'), # Take 200x200 crops from 220x220 padded images
      transforms.RandomHorizontalFlip(),    # 50% of time flip image along y-axis
      transforms.ToTensor(),
    ])
  else:
    transform_train = transforms.Compose([
      transforms.ToPILImage(), # fixes "TypeError: pic should be PIL Image or ndarray. Got <class 'torch.Tensor'>"
      transforms.ToTensor(),
    ])

  transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
  ])

  trainset = ASLDataset(
    annotations_file = "Kaggle/asl_alphabet_train/training_labels.csv",
    img_dir="Kaggle/asl_alphabet_train/",
    transform=transform_train
  )
  # trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

  testset = ASLDataset(
    annotations_file = "Kaggle/asl_alphabet_test/test_labels.csv",
    img_dir="Kaggle/asl_alphabet_test/",
    transform=transform_test
  )
  # testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
  testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)
  classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L'
  , 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
  , 'delete', 'nothing', 'space']
  return {'train': trainloader, 'test': testloader, 'classes': classes}

data = get_ASL_data()
