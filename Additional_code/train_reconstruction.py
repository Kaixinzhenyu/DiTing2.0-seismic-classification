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
import torch.nn.functional as F
from data_loader import dataloader,two_class_dataloader,noise_dataloader
from utils import AverageMeter, save_checkpoint, accuracy, adjust_learning_rate,get_network,MarginLoss

def train(train_loader, model, criterion, optimizer, epoch, writer,reconstructed=False):

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
        input = input.cuda().to(torch.float32)
        
        target = target.cuda().to(torch.long)
        input=torch.squeeze(input)
        #input = input.unsqueeze(2)
        # compute output
        output = model(input)

        loss = criterion(output, target)
        if reconstructed:
            reconstruction_loss = F.mse_loss(model.reconstruct(target), input.view(-1, 8640))#3*40*72
            loss += 0.005* reconstruction_loss
        # measure accuracy and record loss
        prec1= accuracy(output, target, topk=(1,))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0].item(), input.size(0))

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
            input = input.cuda().to(torch.float32)
            target = target.cuda().to(torch.long)
            input=torch.squeeze(input)
            
            # compute output
            output = model(input)
            loss = criterion(output, target)
            
            # measure accuracy and record loss
            prec1= accuracy(output.data, target, topk=(1,))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0].item(), input.size(0))

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

    #数据根路径(3,expand)
    dataset_id='3'
    if dataset_id=='expand':
        #stft,mfcc
        dataset_choose='mfcc'
    else:
        #数据集选择 ('1d','GADF','STFT_26_118','STFT_72_72','STFT_31_1001','MFCC19_24','MFCC_36_72','MFCC_40_72','MFCC_40_601','MFCC_40_1001')
        dataset_choose='MFCC_40_72'
    #(None,True)
    Noise='None'
    #模型选择 (cnn1d,resnet1d,vgg1d,alexnet1d,cnn2d,googlenet,googlenet1d,alexnet,resnet18,vit,cct,vit_small,swin,mult_cnn,vgg11,
    #         cnn_capsnet,res_capsnet,seres_capsnet,res18_capsnet
    #         cct_capsnet,gra_capsnet,cnn_capsnet_reconstruct,res_capsnet_reconstruct)
    model_choose='res_capsnet_reconstruct' 
    order='10'
    routing = 'AgreementRouting' # AgreementRouting,Multi_Head
    #数据集参数(16,32,64,128) 
    Batch_size = 32
    #学习率 (0.01,0.001,0.0001,0.00001)
    learning_rate = 0.0001
    #正则化参数(0.001,0.0001,0.00001)
    Weight_decay = 0.0001
    # 训练次数 
    epochs = 100      
    #类别数(2,3)
    Num_classes=3

    #验证集占训练集的比例  
    Validation_percentage=0.1
    #数据路径  
    base_path='/root/autodl-tmp/Python projects/Dataset/'+'{}'.format(dataset_id)

    # 模型权重保存的路径  
    checkpoint_path = '/root/autodl-tmp/Python projects/my_project/checkpoint/'+'{}'.format(dataset_id)
    # 设置TensorBoard的日志目录
    log_dir = '/root/autodl-tmp/Python projects/my_project/logs/'+'{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(dataset_id,dataset_choose,model_choose,Batch_size,learning_rate,Weight_decay,epochs,Num_classes,order)
    
    writer = SummaryWriter(log_dir)
    if Num_classes ==3:
        if dataset_id =='expand' and dataset_choose== 'mfcc' and Noise=='None':
            train_loader,val_loader =dataloader(natural_train_path_0=base_path+'/mfcc/natural_train_MFCC.pt',
                                            natural_train_path_1=None,
                                            ep_train_path=base_path+'/mfcc/ep_train_MFCC.pt',
                                            ss_train_path=base_path+'/mfcc/ss_train_MFCC.pt',
                                            batch_size=Batch_size,validation_percentage=Validation_percentage)
        if dataset_id =='expand' and dataset_choose== 'stft' and Noise=='None':
            train_loader,val_loader =dataloader(natural_train_path_0=base_path+'/stft/natural_train_STFT.pt',
                                            natural_train_path_1=None,
                                            ep_train_path=base_path+'/stft/ep_train_STFT.pt',
                                            ss_train_path=base_path+'/stft/ss_train_STFT.pt',
                                            batch_size=Batch_size,validation_percentage=Validation_percentage)
        if dataset_id =='expand' and dataset_choose== 'mfcc' and Noise=='True':
            train_loader,val_loader =noise_dataloader(natural_train_path_0=base_path+'/mfcc/natural_train_MFCC.pt',
                                            natural_train_path_1=None,
                                            ep_train_path=base_path+'/mfcc/ep_train_MFCC.pt',
                                            ss_train_path=base_path+'/mfcc/ss_train_MFCC.pt',
                                            noise_train_path=base_path+'/mfcc/noise_train_MFCC.pt',
                                            batch_size=Batch_size,validation_percentage=Validation_percentage)
        if dataset_id =='expand' and dataset_choose== 'stft' and Noise=='True':
            train_loader,val_loader =noise_dataloader(natural_train_path_0=base_path+'/stft/natural_train_STFT.pt',
                                            natural_train_path_1=None,
                                            ep_train_path=base_path+'/stft/ep_train_STFT.pt',
                                            ss_train_path=base_path+'/stft/ss_train_STFT.pt',
                                            noise_train_path=base_path+'/stft/noise_train_STFT.pt',
                                            batch_size=Batch_size,validation_percentage=Validation_percentage)  
        #数据集加载
        if dataset_choose=='1d':
            train_loader,val_loader =dataloader(natural_train_path_0=base_path+'/tensor_dataset/natural_train_dataset.pt',
                                            natural_train_path_1=None,
                                            ep_train_path=base_path+'/tensor_dataset/ep_train_dataset.pt',
                                            ss_train_path=base_path+'/tensor_dataset/ss_train_dataset.pt',
                                            batch_size=Batch_size,validation_percentage=Validation_percentage)
        elif dataset_choose=='GADF':
            train_loader,val_loader =dataloader(natural_train_path_0=base_path+'/processed_data/GADF/natural_train_GADF_0.pt',
                                            natural_train_path_1=base_path+'/processed_data/GADF/natural_train_GADF_1.pt',
                                            ep_train_path=base_path+'/processed_data/GADF/ep_train_GADF.pt',
                                            ss_train_path=base_path+'/processed_data/GADF/ss_train_GADF.pt',
                                            batch_size=Batch_size,validation_percentage=Validation_percentage)
        elif dataset_choose == 'STFT_26_118' or dataset_choose == 'STFT_72_72' or dataset_choose == 'STFT_31_1001':
            train_loader,val_loader =dataloader(natural_train_path_0=base_path+'/processed_data/'+dataset_choose+'/natural_train_STFT.pt',
                                            natural_train_path_1=None,
                                            ep_train_path=base_path+'/processed_data/'+dataset_choose+'/ep_train_STFT.pt',
                                            ss_train_path=base_path+'/processed_data/'+dataset_choose+'/ss_train_STFT.pt',
                                            batch_size=Batch_size,validation_percentage=Validation_percentage)
        elif dataset_choose=='MFCC_39_24' or dataset_choose == 'MFCC_36_72' or dataset_choose == 'MFCC_40_1001'or dataset_choose == 'MFCC_40_72' or dataset_choose == 'MFCC_40_601':
            train_loader,val_loader =dataloader(natural_train_path_0=base_path+'/processed_data/'+dataset_choose+'/natural_train_MFCC.pt',
                                            natural_train_path_1=None,
                                            ep_train_path=base_path+'/processed_data/'+dataset_choose+'/ep_train_MFCC.pt',
                                            ss_train_path=base_path+'/processed_data/'+dataset_choose+'/ss_train_MFCC.pt',
                                            batch_size=Batch_size,validation_percentage=Validation_percentage)
    elif Num_classes ==2:
        if dataset_id =='expand' and dataset_choose== 'mfcc':
            train_loader,val_loader =two_class_dataloader(natural_train_path_0=base_path+'/mfcc/natural_train_MFCC.pt',
                                            natural_train_path_1=None,
                                            ep_train_path=base_path+'/mfcc/ep_train_MFCC.pt',
                                            ss_train_path=base_path+'/mfcc/ss_train_MFCC.pt',
                                            batch_size=Batch_size,validation_percentage=Validation_percentage)
        if dataset_id =='expand' and dataset_choose== 'stft':
            train_loader,val_loader =two_class_dataloader(natural_train_path_0=base_path+'/stft/natural_train_STFT.pt',
                                            natural_train_path_1=None,
                                            ep_train_path=base_path+'/stft/ep_train_STFT.pt',
                                            ss_train_path=base_path+'/stft/ss_train_STFT.pt',
                                            batch_size=Batch_size,validation_percentage=Validation_percentage)
        if dataset_choose=='1d':
            train_loader,val_loader =two_class_dataloader(natural_train_path_0=base_path+'/tensor_dataset/natural_train_dataset.pt',
                                            natural_train_path_1=None,
                                            ep_train_path=base_path+'/tensor_dataset/ep_train_dataset.pt',
                                            ss_train_path=base_path+'/tensor_dataset/ss_train_dataset.pt',
                                            batch_size=Batch_size,validation_percentage=Validation_percentage)
        elif dataset_choose=='GADF':
            train_loader,val_loader =two_class_dataloader(natural_train_path_0=base_path+'/processed_data/GADF/natural_train_GADF_0.pt',
                                            natural_train_path_1=base_path+'/processed_data/GADF/natural_train_GADF_1.pt',
                                            ep_train_path=base_path+'/processed_data/GADF/ep_train_GADF.pt',
                                            ss_train_path=base_path+'/processed_data/GADF/ss_train_GADF.pt',
                                            batch_size=Batch_size,validation_percentage=Validation_percentage)
        elif dataset_choose == 'STFT_26_118' or dataset_choose == 'STFT_72_72' or dataset_choose == 'STFT_31_1001':
            train_loader,val_loader =two_class_dataloader(natural_train_path_0=base_path+'/processed_data/'+dataset_choose+'/natural_train_STFT.pt',
                                            natural_train_path_1=None,
                                            ep_train_path=base_path+'/processed_data/'+dataset_choose+'/ep_train_STFT.pt',
                                            ss_train_path=base_path+'/processed_data/'+dataset_choose+'/ss_train_STFT.pt',
                                            batch_size=Batch_size,validation_percentage=Validation_percentage)
        elif dataset_choose=='MFCC_39_24' or dataset_choose == 'MFCC_36_72' or dataset_choose == 'MFCC_40_72' or dataset_choose == 'MFCC_40_601':
            train_loader,val_loader =two_class_dataloader(natural_train_path_0=base_path+'/processed_data/'+dataset_choose+'/natural_train_MFCC.pt',
                                            natural_train_path_1=None,
                                            ep_train_path=base_path+'/processed_data/'+dataset_choose+'/ep_train_MFCC.pt',
                                            ss_train_path=base_path+'/processed_data/'+dataset_choose+'/ss_train_MFCC.pt',
                                            batch_size=Batch_size,validation_percentage=Validation_percentage)
    # 定义交叉熵损失函数
    if model_choose in ['cnn_capsnet', 'res_capsnet', 'seres_capsnet', 'cct_capsnet', 'res18_capsnet', 'gra_capsnet']:
        criterion = MarginLoss(0.9, 0.1, 0.5).cuda()
        reconstruct = True
    else:
        criterion = nn.CrossEntropyLoss().cuda()
        reconstruct = False
    #定义模型
    model = get_network(dataset_choose,net=model_choose,Num_classes=Num_classes,Reconstruct=reconstruct)
    model.cuda()
    # for i, (input, target) in enumerate(train_loader):
    #     input,target = input.cuda().to(torch.float32),target.cuda().to(torch.long)
    #     print(input.shape)
    #     output = model(input)
    #     print(f"输出大小{output.shape}")
    #     # print(model)
    #     if i >= 0:
    #          break
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=Weight_decay)
    
    best_prec1=0
    total_time=0
    for epoch in range(epochs):
        adjust_learning_rate(optimizer, epoch, learning_rate)#每经过 30 轮就将其减小为原来的 0.1 倍
        end = time.time()
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, writer,reconstruct)

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
        }, is_best,dataset_choose,model_choose,Batch_size,learning_rate,Weight_decay,checkpoint_path,Num_classes,order)
        epoch_time = time.time() -end
        total_time= total_time + epoch_time
        print('Epoch %s time cost %f' %(epoch, epoch_time))

    print('total time cost %f' %(total_time))

if __name__ == '__main__':
    main()