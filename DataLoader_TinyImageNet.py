from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import os
import torch
from torchnet.meter import AUCMeter

            
class tiny_dataset(Dataset): 
    def __init__(self, dataset, num_meta, r, noise_mode, root_dir, transform, mode, noise_file=''): 
        
        self.r = r # noise ratio
        self.transform = transform
        self.mode = mode  
        self.transition = {} # class transition for asymmetric noise
        for i in range(199):
            self.transition[i]=i+1
        self.transition[199]=0
        self.num_meta = num_meta
        self.log_outputs = []
        self.log_cost = []
        self.donewarm=False

        data_transforms = {
            'train': transforms.Compose([
                transforms.ToTensor(),
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
            ]),
        }
        if self.mode=='test':               
            temp_dataset = datasets.ImageFolder(os.path.join(root_dir, 'val'),data_transforms['val'])
            temp_loader = torch.utils.data.DataLoader(temp_dataset, batch_size=10000)
            temp_iter = iter(temp_loader)
            self.test_data,self.test_label = next(temp_iter)
            self.test_data,self.test_label = self.test_data.numpy(),self.test_label.numpy()
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  
            self.test_data = (self.test_data * 255).astype(np.uint8)
                          
        else:    
            temp_dataset = datasets.ImageFolder(os.path.join(root_dir, 'train'),data_transforms['train'])
            lenth = len(temp_dataset)
            temp_loader = torch.utils.data.DataLoader(temp_dataset, batch_size=lenth)
            temp_iter = iter(temp_loader)
            train_data, train_label = next(temp_iter)
            train_data, train_label = train_data.numpy(), train_label.numpy()
            train_data = train_data.transpose((0, 2, 3, 1))
            train_data = (train_data * 255).astype(np.uint8)

            img_num_list = [int(self.num_meta/200)] * 200
            data_list_val = {}
            for j in range(200):
                data_list_val[j] = [i for i, label in enumerate(train_label) if label == j]
            idx_to_meta = []
            idx_to_train = []
            print(img_num_list)
            for cls_idx, img_id_list in data_list_val.items():
                np.random.shuffle(img_id_list)
                img_num = img_num_list[int(cls_idx)]
                idx_to_meta.extend(img_id_list[:img_num])
                idx_to_train.extend(img_id_list[img_num:])
        


            if self.mode == 'train':
                train_data = train_data[idx_to_train]
                train_label = train_label[idx_to_train]
                self.true_label = train_label
                train_label = train_label.tolist()
                if os.path.exists(noise_file):
                    noise_label = json.load(open(noise_file,"r"))
                else:    #inject noise   
                    noise_label = []
                    idx = list(range(lenth-self.num_meta))
                    random.shuffle(idx)
                    num_noise = int(self.r*(lenth-self.num_meta))            
                    noise_idx = idx[:num_noise]
                    for i in range(lenth-self.num_meta):
                        if i in noise_idx:
                            if noise_mode=='sym':   
                                noiselabel = random.randint(0,199)
                                noise_label.append(noiselabel)
                            elif noise_mode=='asym':   
                                noiselabel = self.transition[train_label[i]]
                                noise_label.append(noiselabel)                    
                        else:    
                            noise_label.append(train_label[i])  
                    print("save noisy labels to %s ..."%noise_file)        
                    json.dump(noise_label,open(noise_file,"w"))       
                self.train_data = train_data
                self.noise_label = noise_label
                self.init_noise_labels = noise_label

                
    def __getitem__(self, index):
        if self.mode=='train':
            if self.donewarm==True:
                img, true_label, target, log_outputs_mean = self.train_data[index], self.true_label[index], self.noise_label[index], self.log_outputs_mean[index]
                img = Image.fromarray(img)
                img1 = self.transform(img) 
                img2 = self.transform(img) 
                return img1, img2, true_label, target, log_outputs_mean,index
            elif self.donewarm==False:
                img, true_label, target= self.train_data[index], self.true_label[index], self.noise_label[index]
                img = Image.fromarray(img)
                img1 = self.transform(img) 
                img2 = self.transform(img) 
                return img1, img2, true_label, target, index     
        elif self.mode=='test':
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target
           
    def __len__(self):
        if self.mode!='test':
            return len(self.train_data)
        else:
            return len(self.test_data)         

class tiny_dataloader():  
    def __init__(self, dataset, r, noise_mode, batch_size, num_workers, root_dir, noise_file=''):
        self.dataset = dataset
        self.r = r
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.noise_file = noise_file

        self.transform_train = transforms.Compose([
                transforms.RandomCrop(64, padding=8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]) 
        self.transform_test = transforms.Compose([
                transforms.ToTensor(),
            ])      
    def run(self,mode):      
        if mode=='train':
            labeled_dataset = tiny_dataset(dataset=self.dataset,num_meta=0, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode="train", noise_file=self.noise_file)              
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)   
            return labeled_trainloader

        elif mode=='test':
            test_dataset = tiny_dataset(dataset=self.dataset,num_meta=0, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_test, mode='test')      
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return test_loader      