import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import random
import json
import os
import tqdm
import time
import datetime

from data_loader import dataloader,mult_branch_loader
from utils import AverageMeter, save_checkpoint, accuracy, adjust_learning_rate,get_network

def train(train_loader, model, criterion, optimizer, epoch, writer):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()
    
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        input,target = (input[0].cuda().to(torch.float32),input[1].cuda().to(torch.float32),
                        input[2].cuda().to(torch.float32)),target.cuda().to(torch.long)
        #input = input.unsqueeze(2)
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1= accuracy(output, target, topk=(1,))
        losses.update(loss.item(), input[0].size(0))
        top1.update(prec1[0].item(), input[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 20 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  .format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
    writer.add_scalar('loss/train', losses.val, global_step=epoch+1)
    writer.add_scalar('top1/train', top1.avg, global_step=epoch+1)
def validate(val_loader, model, criterion, epoch, writer):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input,target = (input[0].cuda().to(torch.float32),input[1].cuda().to(torch.float32),
                        input[2].cuda().to(torch.float32)),target.cuda().to(torch.long)
            # input=torch.squeeze(input)
            
            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1= accuracy(output.data, target, topk=(1,))
            losses.update(loss.item(), input[0].size(0))
            top1.update(prec1[0].item(), input[0].size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 20 == 0:
                print('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    .format(
                    i, len(val_loader), batch_time=batch_time,
                      loss=losses, 
                      top1=top1))

        print(' * Prec@1 {top1.avg:.3f} '.format(top1=top1))
    writer.add_scalar('loss/valid', losses.val, global_step=epoch+1)
    writer.add_scalar('top1/valid', top1.avg, global_step=epoch+1)
    return top1.avg

def main():
    #确定随机种子
    fix_seed = 2024
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    #数据根路径(1,2,3)
    dataset_id=2
    #数据集选择 ('STFT_72_72','MFCC_36_72')
    dataset_choose='MFCC_36_72'
    #模型选择 (mult_vgg,mult_resnet,mult_seresnet,mult_cbamresnet,mult_attention)
    model_choose='mult_attention'
    #数据集参数(8,16,32,64,128,256)
    Batch_size = 32
    #学习率 (0.1,0.01,0.001,0.0001,0.00001,0.000001)
    learning_rate = 0.0001
    #正则化参数(0.01,0.001,0.0001,0.00001,0.000001)
    Weight_decay = 0.0001
    # 训练次数 
    epochs = 100 
    #类别数
    Num_classes=3

    #验证集占训练集的比例  
    Validation_percentage=0.1
    #数据路径  
    base_path='/home/zzy/Python projects/Dataset/'+'{}'.format(dataset_id)
    # 模型权重保存的路径  
    checkpoint_path = '/home/zzy/Python projects/Earthquake waveform/my_project/ZzyNet-master/checkpoint/'+'{}'.format(dataset_id)
    # 设置TensorBoard的日志目录
    log_dir = '/home/zzy/Python projects/Earthquake waveform/my_project/ZzyNet-master/logs/'+'{}_{}_{}_{}_{}_{}_{}'.format(dataset_id,dataset_choose,model_choose,Batch_size,learning_rate,Weight_decay,epochs)
    #datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir)
    
    #数据集加载
    train_loader,val_loader=mult_branch_loader(dataset_id, dataset_choose, base_path,Batch_size)
    #定义模型
    model = get_network(dataset_choose,net=model_choose,Num_classes=Num_classes)
    model.cuda()
    # for i, (input, target) in enumerate(train_loader):
    #     input,target = (input[0].cuda().to(torch.float32),input[1].cuda().to(torch.float32),
    #                     input[2].cuda().to(torch.float32)),target.cuda().to(torch.long)
    #     print(target.shape)
    #     print(input[0].shape)
    #     print(input[1].shape)
    #     print(input[2].shape)
    #     output = model(input)
    #     print(f"输出大小{output.shape}")
    #     print(model)
        
    #     if i >= 0:
    #         break
    # 定义交叉熵损失函数
    criterion = nn.CrossEntropyLoss().cuda()
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=Weight_decay)
    
    best_prec1=0
    total_time=0
    for epoch in range(epochs):
        adjust_learning_rate(optimizer, epoch, learning_rate)#每经过 30 轮就将其减小为原来的 0.1 倍
        end = time.time()
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, writer)

        # evaluate on validation set
        prec1= validate(val_loader, model, criterion, epoch, writer)
        
        # remember the best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        print('best_prec1:'+str(best_prec1))
        save_checkpoint({
            'epoch': epoch+1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict()
        }, is_best,dataset_choose,model_choose,Batch_size,learning_rate,Weight_decay,checkpoint_path)
        epoch_time = time.time() -end
        total_time= total_time + epoch_time
        print('Epoch %s time cost %f' %(epoch, epoch_time))
    print('total time cost %f' %(total_time))
if __name__ == '__main__':
    main()