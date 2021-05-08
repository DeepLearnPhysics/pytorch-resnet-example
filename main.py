from __future__ import print_function
import os,sys
import shutil
import time
import traceback
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from larcvdataset import LArCVDataset
import resnet_example

best_prec1 = 0.0

torch.cuda.device( 1 )

def padandcrop(npimg2d):
    imgpad  = np.zeros( (264,264), dtype=np.float32 )
    imgpad[4:256+4,4:256+4] = npimg2d[:,:]
    randx = np.random.randint(0,8)
    randy = np.random.randint(0,8)
    return imgpad[randx:randx+256,randy:randy+256]

def padandcropandflip(npimg2d):
    imgpad  = np.zeros( (264,264), dtype=np.float32 )
    imgpad[4:256+4,4:256+4] = npimg2d[:,:]
    if np.random.rand()>0.5:
        imgpad = np.flip( imgpad, 0 )
    if np.random.rand()>0.5:
        imgpad = np.flip( imgpad, 1 )
    randx = np.random.randint(0,8)
    randy = np.random.randint(0,8)
    return imgpad[randx:randx+256,randy:randy+256]


def main():

    global best_prec1

    # create model: loading resnet18 as defined in torchvision module
    #model = resnet_example.resnet18(pretrained=False, num_classes=5, input_channels=1)
    model = resnet_example.resnet14(pretrained=False, num_classes=5, input_channels=1)
    model.cuda()

    print("Loaded model: ",model)


    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    # training parameters
    lr = 1.0e-2
    momentum = 0.9
    weight_decay = 1.0e-4
    batchsize = 64
    batchsize_valid = 64
    start_epoch = 0
    epochs      = 50000
    nbatches_per_epoch = int(epochs/batchsize)
    nbatches_per_valid = int(epochs/batchsize_valid)

    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=weight_decay)

    cudnn.benchmark = True

    # dataset
    iotrain = LArCVDataset("train_dataloader.cfg", "ThreadProcessor", loadallinmem=True)
    iovalid = LArCVDataset("valid_dataloader.cfg", "ThreadProcessorTest",loadallinmem=False)
    
    iotrain.start(batchsize)
    iovalid.start(batchsize_valid)

    # Resume training option
    if False:
        checkpoint = torch.load( "checkpoint.pth.p01.tar" )
        best_prec1 = checkpoint["best_prec1"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    if False:
        data = iotrain[0]
        img = data["image"]
        lbl = data["label"]
        img_np = np.zeros( (img.shape[0], 1, 256, 256), dtype=np.float32 )
        for j in range(img.shape[0]):
            imgtemp = img[j].reshape( (256,256) )
            print(imgtemp.shape)
            img_np[j,0,:,:] = padandcrop(imgtemp)
            lbl_np[j] = np.argmax(lbl[j])
        print("Train label")
        print(lbl_np)

        datatest = iovalid[0]
        imgtest = data["image"]
        print("Test image shape")
        print(imgtest.shape)

        iotrain.stop()
        iovalid.stop()
        
        return

    for epoch in range(start_epoch, epochs):

        adjust_learning_rate(optimizer, epoch, lr)
        #print("Epoch [%d]: "%(epoch),)
        #for param_group in optimizer.param_groups:
        #    print("lr=%.3e"%(param_group['lr']),)
        #print()

        # train for one epoch
        try:
            train_ave_loss, train_ave_acc = train(iotrain, model, criterion, optimizer, nbatches_per_epoch, epoch, 50)
        except Exception as e:
            print("Error in training routine!")
            print(e.message)
            print(e.__class__.__name__)
            traceback.print_exc(e)
            break
        print("Epoch [%d] train aveloss=%.3f aveacc=%.3f"%(epoch,train_ave_loss,train_ave_acc))

        # evaluate on validation set
        try:
            prec1 = validate(iovalid, model, criterion, nbatches_per_valid, 1)
        except Exception as e:
            print("Error in validation routine!")
            print(e.message)
            print(e.__class__.__name__)
            traceback.print_exc(e)
            break

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, -1)
        if epoch==5*50:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, False, epoch)
            

    iotrain.stop()
    iovalid.stop()



def train(train_loader, model, criterion, optimizer, nbatches, epoch, print_freq):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    format_time = AverageMeter()
    train_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    for i in range(0,nbatches):
        #print("epoch ",epoch," batch ",i," of ",nbatches)
        optimizer.zero_grad()
        
        batchstart = time.time()
    
        end = time.time()        
        data = train_loader[i]
        # measure data loading time
        data_time.update(time.time() - end)

        end = time.time()
        img = data["image"]
        lbl = data["label"]
        img_np = np.zeros( (img.shape[0], 1, 256, 256), dtype=np.float32 )
        lbl_np = np.zeros( (lbl.shape[0] ), dtype=np.int )
        # batch loop
        for j in range(img.shape[0]):
            imgtmp = img[j].reshape( (256,256) )
            img_np[j,0,:,:] = padandcropandflip(imgtmp) # data augmentation
            lbl_np[j] = np.argmax(lbl[j])
            #print(lbl[j]," ",lbl_np[j])
        input_var  = torch.from_numpy(img_np).cuda()
        target_var = torch.from_numpy(lbl_np).cuda()
        #print("target: ",target_var,target_var.shape)

        # measure data formatting time
        format_time.update(time.time() - end)        

        # compute output
        end = time.time()
        # forward
        output = model(input_var)
        # loss calculation
        loss = criterion(output, target_var)
        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        with torch.no_grad():
            prec1 = accuracy(output.detach(), target_var, topk=(1,))        
            losses.update(loss.detach().cpu().item(), input_var.size(0))
            top1.update(prec1[0], input_var.size(0))

        train_time.update(time.time()-end)

        # measure elapsed time
        batch_time.update(time.time() - batchstart)


        if i % print_freq == 0:
            status = (epoch,i,nbatches,
                      batch_time.val,batch_time.avg,
                      data_time.val,data_time.avg,
                      format_time.val,format_time.avg,
                      train_time.val,train_time.avg,
                      losses.val,losses.avg,
                      top1.val,top1.avg)
            print("Epoch: [%d][%d/%d]\tTime %.3f (%.3f)\tData %.3f (%.3f)\tFormat %.3f (%.3f)\tTrain %.3f (%.3f)\tLoss %.3f (%.3f)\tPrec@1 %.3f (%.3f)"%status)
            #print('Epoch: [{0}][{1}/{2}]\t'
            #      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            #      'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
            #      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
            #          epoch, i, len(train_loader), batch_time=batch_time,
            #          data_time=data_time, losses=losses, top1=top1 ))
    return losses.avg,top1.avg


def validate(val_loader, model, criterion, nbatches, print_freq):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i in range(0,nbatches):
        data = val_loader[i]
        img = data["imagetest"]
        lbl = data["labeltest"]
        img_np = np.zeros( (img.shape[0], 1, 256, 256), dtype=np.float32 )
        lbl_np = np.zeros( (lbl.shape[0] ), dtype=np.int )
        for j in range(img.shape[0]):
            img_np[j,0,:,:] = img[j].reshape( (256,256) )
            lbl_np[j] = np.argmax(lbl[j])
        inimg_var  = torch.from_numpy(img_np).cuda()
        target = torch.from_numpy(lbl_np).cuda()
        
        # compute output
        with torch.no_grad():
            output = model(inimg_var)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1 = accuracy(output.detach(), target, topk=(1,))
            losses.update(loss.detach().cpu().item(), inimg_var.size(0))
            top1.update(prec1[0], inimg_var.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            status = (i,nbatches,batch_time.val,batch_time.avg,losses.val,losses.avg,top1.val,top1.avg)
            #print("Test: [%d/%d]\tTime %.3f (%.3f)\tLoss %.3f (%.3f)\tPrec@1 %.3f (%.3f)"%status)
            #print('Test: [{0}/{1}]\t'
            #      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
            #        i, len(val_loader), batch_time=batch_time, loss=losses,
            #        top1=top1))

    #print(' * Prec@1 {top1.avg:.3f}'
    #      .format(top1=top1))
    print("Test:Result* Prec@1 %.3f\tLoss %.3f"%(top1.avg,losses.avg))

    return float(top1.avg)


def save_checkpoint(state, is_best, p, filename='checkpoint.pth.tar'):
    if p>0:
        filename = "checkpoint.%dth.tar"%(p)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    #lr = lr * (0.5 ** (epoch // 300))
    lr = lr
    #lr = lr*0.992
    #print "adjust learning rate to ",lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def dump_lr_schedule( startlr, numepochs ):
    for epoch in range(0,numepochs):
        lr = startlr*(0.5**(epoch//300))
        if epoch%10==0:
            print("Epoch [%d] lr=%.3e"%(epoch,lr))
    print("Epoch [%d] lr=%.3e"%(epoch,lr))
    return

if __name__ == '__main__':
    #dump_lr_schedule(1.0e-2, 4000)
    main()
