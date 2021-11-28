import sys
sys.path
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from tqdm import tqdm_notebook
import json
import xlrd
import torch
import torchvision

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from torchvision.transforms import transforms
from torchvision.utils import make_grid

from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


class ECG_Net(nn.Module):
    def __init__(self, init_weights= True, num_classes=26, cfg = None):
        super(ECG_Net, self).__init__()
        if cfg is None:
            cfg = [32, 32,32,   64,64,64,    128, 128, 128,  256, 256, 256,  512, 512, 512]

        inchannel = 12
        self.pre_conv = nn.Sequential(
            nn.Conv1d(inchannel, cfg[0], kernel_size=21, stride= 2, padding=10, bias=True),
            nn.BatchNorm1d(cfg[0]),
            nn.ReLU(),
            nn.Conv1d(cfg[0], cfg[1], kernel_size=21, stride= 2, padding=10, bias=True),
            nn.BatchNorm1d(cfg[1]),
            nn.ReLU(),
            nn.Conv1d(cfg[1], cfg[2], kernel_size=21, stride= 2, padding=10, bias=True),
            nn.BatchNorm1d(cfg[2]),
            nn.ReLU(),           
            nn.MaxPool1d(2, stride=2)
            
        )    
        self.stage1 = nn.Sequential(
            nn.Conv1d(cfg[2], cfg[3], kernel_size=5, stride= 1, padding=2, bias=True),
            nn.BatchNorm1d(cfg[3]),
            nn.ReLU(),

            nn.Conv1d(cfg[3], cfg[4], kernel_size=5, stride= 1, padding=2, bias=True),
            nn.BatchNorm1d(cfg[4]),
            nn.ReLU(),

            nn.Conv1d(cfg[4], cfg[5], kernel_size=5, stride= 1, padding=2, bias=True),
            nn.BatchNorm1d(cfg[5]),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2)

                )    
        self.stage2 = nn.Sequential(
            nn.Conv1d(cfg[5], cfg[6], kernel_size=5, stride= 1, padding=2, bias=True),
            nn.BatchNorm1d(cfg[6]),
            nn.ReLU(),

            nn.Conv1d(cfg[6], cfg[7], kernel_size=5, stride= 1, padding=2, bias=True),
            nn.BatchNorm1d(cfg[7]),
            nn.ReLU(),

            nn.Conv1d(cfg[7], cfg[8], kernel_size=5, stride= 1, padding=2, bias=True),
            nn.BatchNorm1d(cfg[8]),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2)

             )    
        
        self.stage3 = nn.Sequential(
            nn.Conv1d(cfg[8], cfg[9], kernel_size=5, stride= 1, padding=2, bias=True),
            nn.BatchNorm1d(cfg[9]),
            nn.ReLU(),

            nn.Conv1d(cfg[9], cfg[10], kernel_size=5, stride= 1, padding=2, bias=True),
            nn.BatchNorm1d(cfg[10]),
            nn.ReLU(),

            nn.Conv1d(cfg[10], cfg[11], kernel_size=5, stride= 1, padding=2, bias=True),
            nn.BatchNorm1d(cfg[11]),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2)

             ) 
        self.stage4 = nn.Sequential(
            nn.Conv1d(cfg[11], cfg[12], kernel_size=5, stride= 1, padding=2, bias=True),
            nn.BatchNorm1d(cfg[12]),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(cfg[12], cfg[13], kernel_size=5, stride= 1, padding=2, bias=True),
            nn.BatchNorm1d(cfg[13]),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(cfg[13], cfg[14], kernel_size=5, stride= 1, padding=2, bias=True),
            nn.BatchNorm1d(cfg[14]),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2)

             )  
   
        self.avg_pool = nn.AvgPool1d(2, stride=2)
        
        self.dense1 =nn.Sequential(
            nn.Linear(cfg[14]*2, 512, bias = True),
            nn.ReLU(),
            nn.Dropout(p=0.6)
            )
        self.dense2 =nn.Sequential(
            nn.Linear(512, 512, bias = True),
            nn.ReLU(),
            nn.Dropout(p=0.6)
            )
     
        self.classifer = nn.Sequential(
            nn.Linear(512, num_classes, bias = True),
            nn.Sigmoid()
            )

        if init_weights:
            self._initialize_weights()
        
    def forward(self, x):
        x = torch.squeeze(x, dim=1)
        x = torch.transpose(x, 1, 2)       
        out = self.pre_conv(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.dense1(out)
        out = self.dense2(out)
        out = self.classifer(out)
        return out
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.bias, 0)
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)        
                nn.init.xavier_normal_(m.weight)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = ECG_Net().to(device)

def train_model(model,train_loader,val_loader,batch_size, num_epochs,loss_fun,optimizer):
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []     

    for epoch in tqdm_notebook(range(1, num_epochs + 1)):

        if epoch in [10,20,30,40,50]:
            for param_group in optimizer.param_groups:
                param_group['lr']*=0.1
        
        train_running_precision = 0.0
        train_running_recall = 0.0
        val_running_precision = 0.0
        val_running_recall = 0.0
        train_running_acc = 0
        val_running_acc = 0
        
        train_batch_num = 0
        val_batch_num = 0
        ###################
        # train the model #
        ###################
        model.train() # model for training
        for batch_idx, (data, label) in tqdm_notebook(enumerate(train_loader)):
            data, label = data.float().to(device), label.to(device)
            data, target = Variable(data), Variable(label)            
            
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fun(output, label)  
            
            loss.backward()
            
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
            optimizer.step()            

            train_losses.append(loss.item())
            train_batch_num +=1            

        ######################    
        # validate the model #
        ######################
        pre_result = []
        label_tmp = []
        model.eval() # prep model for evaluation
        with torch.no_grad():
            for batchsize,(data, label) in tqdm_notebook(enumerate(val_loader)):
                data, label = data.float().to(device), label.to(device)
                data, target = Variable(data), Variable(label)
                output = model(data)
                loss = loss_fun(output, label).item()  # sum up batch loss
                valid_losses.append(loss)
                
                pre_result.append(output)
                label_tmp.append(label)                

        pred = torch.cat(pre_result,dim=0)
        label = torch.cat(label_tmp,dim=0)
        
        best_thres=np.array([0.8, 0.8, 0.64, 0.76 , 0.75 ,0.61, 0.71, 0.78, 0.47, 0.8, 0.49, 0.85, 0.57,
                     0.32 , 0.68, 0.46, 0.22, 0.83, 0.87 ,0.11, 0.52, 0.58, 0.85, 0.43 , 0.75, 0.33 ])
        best_thres = torch.from_numpy(best_thres)
        
        acc_result = torch.ge(pred.cpu(), best_thres)
        acc_result = acc_result+0

        acc = accuracy_score(label.cpu(), acc_result.cpu(),normalize = True)

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        
        epoch_len = len(str(num_epochs))        

        print('Epoch:{}  train_loss:{:.5f}   valid_loss:{:.5f}  valid_acc:{:.5f}'.format(epoch,train_loss,valid_loss,acc))
        
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
 
        torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict()},'./ECG/Model_parameter/ECG_model' + str(epoch) + '.pth.tar')