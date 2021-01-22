from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import os
import torch
from torchnet.meter import AUCMeter

            
def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

class cifar_dataset(Dataset): 
    def __init__(self, dataset, num_meta, r, noise_mode, root_dir, transform, mode, noise_file=''): 
        
        self.r = r # noise ratio
        self.transform = transform
        self.mode = mode  
        self.transition10 = {0:0,2:0,4:7,7:7,1:1,9:1,3:5,5:3,6:6,8:8} # class transition for asymmetric noise
        self.transition100 = {} # class transition for asymmetric noise
        for i in range(99):
            self.transition100[i]=i+1
        self.transition100[99]=0
        self.num_meta = num_meta
        self.log_outputs = []
        self.log_cost = []
        self.donewarm=False

        if self.mode=='test':
            if dataset=='cifar10':                
                test_dic = unpickle('%s/test_batch'%root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))  
                self.test_label = test_dic['labels']
            elif dataset=='cifar100':
                test_dic = unpickle('%s/test'%root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))  
                self.test_label = test_dic['fine_labels']                            
        else:    
            train_data=[]
            train_label=[]
            if dataset=='cifar10': 
                for n in range(1,6):
                    dpath = '%s/data_batch_%d'%(root_dir,n)
                    data_dic = unpickle(dpath)
                    train_data.append(data_dic['data'])
                    train_label = train_label+data_dic['labels']
                train_data = np.concatenate(train_data)
                train_label = np.array(train_label)

            elif dataset=='cifar100':    
                train_dic = unpickle('%s/train'%root_dir)
                train_data = train_dic['data']
                train_label = train_dic['fine_labels']
                train_label = np.array(train_label)

            train_data = train_data.reshape((50000, 3, 32, 32))
            train_data = train_data.transpose((0, 2, 3, 1))
            
            self.true_label = train_label
            train_label = train_label.tolist()
            if os.path.exists(noise_file):
                noise_label = json.load(open(noise_file,"r"))
            else:    #inject noise   
                noise_label = []
                idx = list(range(50000-self.num_meta))
                random.shuffle(idx)
                num_noise = int(self.r*(50000-self.num_meta))            
                noise_idx = idx[:num_noise]
                for i in range(50000-self.num_meta):
                    if i in noise_idx:
                        if noise_mode=='sym':
                            if dataset=='cifar10':
                                noiselabel = random.randint(0,9)
                            elif dataset=='cifar100':    
                                noiselabel = random.randint(0,99)
                            noise_label.append(noiselabel)
                        elif noise_mode=='asym':   
                            if dataset=='cifar10':
                                noiselabel = self.transition10[train_label[i]]
                            elif dataset=='cifar100':
                                noiselabel = self.transition100[train_label[i]]
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
        elif self.mode=='meta': 
            img, target = self.train_data[index], self.train_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target        
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

class cifar_dataloader():  
    def __init__(self, dataset, r, noise_mode, batch_size, num_workers, root_dir, noise_file=''):
        self.dataset = dataset
        self.r = r
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.noise_file = noise_file
        if self.dataset=='cifar10':
            self.transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                ]) 
            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                ])    
        elif self.dataset=='cifar100':    
            self.transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ]) 
            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ])   
    def run(self,mode):      
        if mode=='train':
            labeled_dataset = cifar_dataset(dataset=self.dataset,num_meta=0, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode="train", noise_file=self.noise_file)              
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)   
            return labeled_trainloader

        elif mode=='test':
            test_dataset = cifar_dataset(dataset=self.dataset,num_meta=0, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_test, mode='test')      
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return test_loader      