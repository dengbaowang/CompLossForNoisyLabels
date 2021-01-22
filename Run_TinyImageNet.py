from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import argparse
import numpy as np
from PreResNet import *
from sklearn.mixture import GaussianMixture
import DataLoader_TinyImageNet as dataloader
from torch.autograd import Variable
from loss import *
from torchsummary import summary

parser = argparse.ArgumentParser(description='PyTorch Tiny Training')
parser.add_argument('--batch_size', default=100, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--noise_mode',  default='sym')
parser.add_argument('--gamma1', default=0.6, type=float)
parser.add_argument('--alpha', default=4, type=float, help='for mixup')
parser.add_argument('--begin_warm', default=30, type=int)
parser.add_argument('--warm_up', default=60, type=int)
parser.add_argument('--r', default=0.5, type=float, help='noise ratio')
parser.add_argument('--seed', default=123)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--data_path', default='./tiny', type=str, help='path to dataset')
parser.add_argument('--dataset', default='tiny', type=str)
parser.add_argument('--rloss', default='NCEandRCE2', type=str, help='robust loss function')
parser.add_argument('--randidx', default=0, type=int,  help='random index')
args = parser.parse_args()
print(args)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

device = torch.device('cuda')

# Training
def train(epoch,net,optimizer,labeled_trainloader):
    net.train()    
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1
    outputs_all = torch.zeros(len(labeled_trainloader.dataset),args.num_class)
    for batch_idx, (inputs_x, inputs_x2, true_label, noise_labels, log_outputs_mean, index) in enumerate(labeled_trainloader):  
        batch_size = inputs_x.size(0)
        labels_x = torch.argmax(log_outputs_mean, 1)
        ent = entropy(log_outputs_mean,1,False)
        ent_sort, _=torch.sort(ent,descending=True)
        w = (ent<ent_sort[int(batch_size*(1-args.gamma1))-1]).float().cuda()

        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1,1), 1) 
        noise_labels_bi = torch.zeros(batch_size, args.num_class).scatter_(1, noise_labels.view(-1,1), 1)    
        true_label_bi = torch.zeros(batch_size, args.num_class).scatter_(1, true_label.view(-1,1), 1)    
        inputs_x, inputs_x2, targets_x, noise_labels, true_label,noise_labels_bi = inputs_x.cuda(), inputs_x2.cuda(), labels_x.cuda(), noise_labels.cuda(), true_label.cuda(),noise_labels_bi.cuda()
        #######################################################################################argument


        all_inputs = torch.cat([inputs_x, inputs_x2], dim=0)
        all_targets = torch.cat([targets_x, targets_x], dim=0)
        all_w = torch.cat([w,w],dim=0)
        all_noise_labels_bi = torch.cat([noise_labels_bi, noise_labels_bi], dim=0)
        #######################################################################################easy sample
        all_inputs_easy = all_inputs[all_w==1]
        all_targets_easy = all_targets[all_w==1]
        # mixmatch
        idx = torch.randperm(all_inputs_easy.size(0))
        input_a_easy, input_b_easy = all_inputs_easy, all_inputs_easy[idx]
        target_a_easy, target_b_easy = all_targets_easy, all_targets_easy[idx]
        l = np.random.beta(args.alpha, args.alpha)        
        l = max(l, 1-l)
        mixed_input_easy = l * input_a_easy + (1 - l) * input_b_easy        
        mixed_target_easy = l * target_a_easy + (1 - l) * target_b_easy

        logits_easy = net(mixed_input_easy)
        Lx_easy = -torch.mean(torch.sum(F.log_softmax(logits_easy, dim=1) * mixed_target_easy, dim=1))
        #######################################################################################hard sample     
        logits_all = net(all_inputs)
        if args.rloss == 'MAE':
            Lx_robust = torch.mean(torch.sum(torch.abs(F.softmax(logits_all, dim=1) - all_noise_labels_bi), 1))##########MAE
        if args.rloss == 'NCEandRCE2':
            Lx_robust = NCEandRCE(logits_all, all_noise_labels_bi)
        if args.rloss == 'no':
            loss=Lx_easy
        else:
            loss = Lx_easy + Lx_robust
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        for j, k in enumerate(index):
            sm_outputs = torch.softmax(logits_all[:batch_size], dim=1)
            outputs_all[k,:] = sm_outputs[j,:].detach()

    outputs_all = outputs_all.reshape(len(labeled_trainloader.dataset), -1, args.num_class)
    labeled_trainloader.dataset.log_outputs = torch.cat([labeled_trainloader.dataset.log_outputs, outputs_all], 1)            

def warmup(epoch,net,optimizer,dataloader):
    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    outputs_all = torch.zeros(len(dataloader.dataset),args.num_class)
    for batch_idx, (inputs, _,true_label, labels, index) in enumerate(dataloader):     
        batch_size = inputs.size(0)
        labels_bi = torch.zeros(batch_size, args.num_class).scatter_(1, labels.view(-1,1), 1)  
        true_label_bi = torch.zeros(batch_size, args.num_class).scatter_(1, true_label.view(-1,1), 1)
        inputs, labels, labels_bi = inputs.cuda(), labels.cuda(), labels_bi.cuda()

        optimizer.zero_grad()
        outputs = net(inputs)

        outputs_clamp = torch.clamp(F.softmax(outputs), min=1e-4, max=1.0)
        outputs_y = torch.sum(outputs_clamp*labels_bi,1)
        weight = Variable((1/outputs_y)**(1/2), requires_grad=False).to(device)
        loss = torch.mean(torch.sum(torch.abs(F.softmax(outputs, dim=1) - labels_bi), 1)*weight) # MAE with weighted gradient

        if args.noise_mode=='asym':  # penalize confident prediction for asymmetric noise
            penalty = conf_penalty(outputs)
            L = loss + penalty      
        elif args.noise_mode=='sym':   
            L = loss
        L.backward()  
        optimizer.step() 
        for j, k in enumerate(index):
            sm_outputs = torch.softmax(outputs, dim=1)
            outputs_all[k,:] = sm_outputs[j,:].detach()
    outputs_all = outputs_all.reshape(len(dataloader.dataset), -1, args.num_class)
    if epoch >= begin_warm:
        if epoch == begin_warm:
            dataloader.dataset.log_outputs = outputs_all
        else:
            dataloader.dataset.log_outputs = torch.cat([dataloader.dataset.log_outputs, outputs_all], 1)

def test(epoch,net):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)            
                       
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()                 
    acc = 100.*correct/total
    print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" %(epoch,acc))  
    test_log.write('Epoch:%d   Accuracy:%.2f\n'%(epoch,acc))
    test_log.flush()  

def create_model():
    model = ResNettiny(num_classes=args.num_class)
    model = model.cuda()
    summary(model,(3,64,64))
    return model

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

def entropy(p, dim = -1, keepdim = None):
   return -torch.where(p > 0, p * p.log(), p.new([0.0])).sum(dim = dim, keepdim = keepdim)


warm_up = args.warm_up
test_log=open('./checkpoint/%s_%.1f_%s'%(args.dataset,args.r,args.noise_mode)+'_acc_pth.txt','w') 
loader = dataloader.tiny_dataloader(args.dataset,r=args.r,noise_mode=args.noise_mode,batch_size=args.batch_size,num_workers=5,\
    root_dir=args.data_path,noise_file='%s/%.1f_%s_%d.json'%(args.data_path,args.r,args.noise_mode,args.randidx))

net = create_model()
cudnn.benchmark = True

optimizer = optim.SGD(net.params(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
NFLandMAE = NFLandMAE(num_classes=args.num_class,gamma=0.5,alpha=10.0,beta=1.0)
NCEandRCE = NCEandRCE(num_classes=args.num_class,alpha=10.0,beta=0.1)
CEloss = nn.CrossEntropyLoss()
NCE = NormalizedCrossEntropy(scale=1.0, num_classes=args.num_class)
MAE = MeanAbsoluteError(scale=1.0, num_classes=args.num_class)
if args.noise_mode=='asym':
    conf_penalty = NegEntropy()

test_loader = loader.run('test')
trainloader = loader.run('train')
begin_warm = args.begin_warm
for epoch in range(201):   
    lr=args.lr
    if epoch >= 100:
        lr /= 10 

    print('lr: ', lr)      
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr        
    
    if epoch<warm_up:       
        print('Warmup Net')
        warmup(epoch,net,optimizer, trainloader)    
    else:              
        trainloader.dataset.donewarm=True
        trainloader.dataset.log_outputs_mean = torch.sum(trainloader.dataset.log_outputs,1)/trainloader.dataset.log_outputs.size(1)

        print('Train Net')
        train(epoch,net,optimizer, trainloader) # train net  
    test(epoch,net)  