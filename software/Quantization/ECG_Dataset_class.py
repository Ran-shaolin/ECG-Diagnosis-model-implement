import sys
sys.path
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from tqdm import tqdm_notebook
import json
import torch
import torchvision
from torchvision import datasets, transforms
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

class ECG_Dataset(Dataset):
    def __init__(self,txt, transform=None):
        super(ECG_Dataset,self).__init__()
        fp = open(txt, 'r')
        datas = []
        labels = []
        for line in fp: 
            line = line.strip('\n')
            line = line.rstrip('\n')
            words = line.split()
            datas.append(words[0])

            labels.append([int(l) for l in words[1:len(words)]])
        self.datas = datas
        self.labels = labels
        self.transform = transform
 
    def __getitem__(self, index):
        data_path = self.datas[index]
        label = self.labels[index]
        data = np.load(data_path)
        if self.transform is not None:
            data = self.transform(data) 
        label = torch.FloatTensor(label)
        return data,label
    def __len__(self):
        return len(self.datas)




