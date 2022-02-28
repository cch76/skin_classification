import argparse
import os
import random
import shutil
import time
import glob
import json

import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
from torchvision.datasets import ImageFolder

from efficientnet_pytorch import EfficientNet
from vit_pytorch import ViT
from vit_pytorch.nest import NesT
from vit_pytorch.cait import CaiT
from mlp_mixer_pytorch import MLPMixer
from models.src.convmlp import convmlp_l, convmlp_s
from models.iresnet import iresnet50, iresnet100
import timm
import torch.distributed as dist

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from facenet_pytorch import MTCNN, InceptionResnetV1

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

parser = argparse.ArgumentParser(description='PyTorch face Training')
# Datasets
parser.add_argument('-d', '--dataset', default='face', type=str)
parser.add_argument('--label', default='bau', type=str)
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
# Optimization options
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=128, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[20, 40],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--optim', default='adam', type=str)
# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50')
parser.add_argument('--patch_size', type=int, default=16, help='patch size in ViT')
parser.add_argument('--pretrain', dest='pretrain', action='store_true',
                    help='pretrained model')
parser.add_argument('--loss', default='l1', type=str)
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed', default=42)
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('--save_dir', default='results/', type=str)
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

os.makedirs(args.save_dir, exist_ok=True)

# save argsparser
with open(os.path.join(args.save_dir,'args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)

def main():
    # start from epoch 0 or last checkpoint epoch
    start_epoch = args.start_epoch
    torch.multiprocessing.set_start_method('spawn')
    best_loss = 10000
    mean = (0.5955, 0.4800, 0.4182)
    std = (0.2350, 0.2061, 0.1927)

    # Data
    print('==> Preparing dataset %s' % args.dataset)
    
    # trasform setting
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((0,15)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    meta = pd.read_csv('./data/label/meta.csv')
    
    label = []
    if args.label == 'bau':
        num_classes=35
        for i, value in enumerate(meta['test answer']):
            label.append([int(x) for x in meta['test answer'][i].split(',')[0:35]])
    elif args.label == 'score':
        num_classes=5
        label = meta.iloc[:,17:22].values
    gt = meta['awre'].values

    if args.dataset == 'face':
        data_dir='./data/face/'

    trainset = FACE(data_dir, transform_train, torch.FloatTensor(label), gt)
    train_set, val_set = torch.utils.data.random_split(trainset, [17000, 3000])

    train_set = ApplyTransform(train_set, transform_train)
    val_set = ApplyTransform(val_set, transform_test)

    trainloader = data.DataLoader(train_set, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
    testloader = data.DataLoader(val_set, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    # Model
    print("==> creating model '{}'".format(args.arch))
    if args.arch == 'resnet152':
        model = models.resnet152(pretrained=args.pretrain)
        model.fc = nn.Linear(2048, num_classes)
        model = torch.nn.DataParallel(model).cuda()
    elif args.arch == 'resnet50':
        model = models.resnet50(pretrained=args.pretrain)
        model.fc = nn.Linear(2048, num_classes)
        model = torch.nn.DataParallel(model).cuda()
    elif args.arch == 'resnet18':
        model = models.resnet18(pretrained=args.pretrain)
        model = model.cuda()
    elif args.arch == 'efficient7':
        if args.pretrain == True:
            model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=num_classes)
        else :
            model = EfficientNet.from_name('efficientnet-b7', num_classes=num_classes)
        model = torch.nn.DataParallel(model).cuda()
    elif args.arch == 'efficient4':
        if args.pretrain == True:
            model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=num_classes)
        else :
            model = EfficientNet.from_name('efficientnet-b4', num_classes=num_classes)
    elif args.arch == 'vit':
        model = timm.create_model('vit_large_patch16_224', pretrained=args.pretrain, num_classes=num_classes)

    elif args.arch == 'nest':
        model = NesT(
            image_size = 224,
            patch_size = args.patch_size,
            dim = 96,
            heads = 3,
            num_hierarchies = 3,        # number of hierarchies
            block_repeats = (8, 4, 1),  # the number of transformer blocks at each heirarchy, starting from the bottom
            num_classes = num_classes
        )
    elif args.arch == 'cait':
        model = CaiT(
            image_size = 224,
            patch_size = args.patch_size,
            num_classes = num_classes,
            dim = 1024,
            depth = 12,             # depth of transformer for patch to patch attention only
            cls_depth = 2,          # depth of cross attention of CLS tokens to patch
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1,
            layer_dropout = 0.05    # randomly dropout 5% of the layers
        )
    elif args.arch == 'mixer':
        model = MLPMixer(
            image_size = 224,
            channels = 3,
            patch_size = 16,
            dim = 1024,
            depth = 24,
            num_classes = num_classes
        )
    elif args.arch == 'mixerH':
        model = MLPMixer(
            image_size = 224,
            channels = 3,
            patch_size = 14,
            dim = 1280,
            depth = 32,
            num_classes = num_classes
        )
    elif args.arch == 'mixerS':
        model = MLPMixer(
            image_size = 224,
            channels = 3,
            patch_size = 16,
            dim = 512,
            depth = 8,
            num_classes = num_classes
        )
    elif args.arch == 'convmlp':
        model = convmlp_l(pretrained=args.pretrain, progress=True, num_classes=num_classes)
    elif args.arch == 'iresnet50':
        model = iresnet50(num_features=num_classes)
    model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    # loss setting
    if args.loss == 'l1' :
        criterion = nn.L1Loss()
    elif args.loss == 'l2' :
        criterion = nn.MSELoss()

    # optimizer setting
    if args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Resume
    title = 'face type' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.save_dir = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.save_dir, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.save_dir, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train std.', 'Valid std.'])

    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_std, bau = test(testloader, model, criterion, start_epoch, use_cuda, num_classes=num_classes)
        print(' Test Loss:  %.8f, Test std:  %.4f' % (test_loss, test_std))
        np.savetxt(os.path.join(args.save_dir, 'bau.csv'),bau,delimiter=',')
        return

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        # decay learning rate
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_std, forest = train(trainloader, model, criterion, optimizer, epoch, use_cuda)
        test_loss, test_std, bau = test(testloader, model, criterion, epoch, use_cuda, forest, num_classes=num_classes)


        # append logger file
        logger.append([state['lr'], train_loss, test_loss, train_std, test_std])  
        
        # save model
        is_best = test_loss < best_loss
        best_loss = min(test_loss, best_loss)

        if is_best:
            np.savetxt(os.path.join(args.save_dir, 'bau.csv'), bau, delimiter=',')
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=args.save_dir)

        

    logger.close()

def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    std = AverageMeter()
    end = time.time()

    out_list = []
    gt_list = ()
    bar = Bar('Processing', max=len(trainloader))
    print(args)

    for batch_idx, (inputs,targets, gt) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            
        # compute output
        outputs = model(inputs)
        out_list.append(outputs)
        gt_list = gt_list + gt

        # measure loss
        loss = criterion(outputs, targets)
        losses.update(loss.item(), inputs.size(0))
        std.update(np.mean(np.abs((outputs-targets).cpu().detach().numpy())), inputs.size(0))
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | STD : {std:.4f} |'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    std=std.avg,
                    )
        bar.next()
    bar.finish()

    output = torch.vstack(out_list).cpu().detach().numpy()

    # train rf classifier
    forest = RandomForestClassifier(n_estimators=200, max_depth= None, random_state=42, oob_score=True)
    forest.fit(output, gt_list)
    predicted=forest.predict(output)
    acc = accuracy_score(gt_list, predicted)
    print('train acc : ',acc)
    
    return (losses.avg, std.avg, forest)

def test(testloader, model, criterion, epoch, use_cuda, forest, num_classes):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    std = AverageMeter()

    out_list = []
    gt_list = ()

    # switch to evaluate mode
    model.eval()

    bau = np.zeros(num_classes,)

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    with torch.no_grad():
        for batch_idx, (inputs, targets, gt) in enumerate(testloader):
            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            out_list.append(outputs)
            gt_list = gt_list + gt

            # measure accuracy and record loss
            bau = bau + np.mean(np.abs((outputs-targets).cpu().numpy()), axis=0)
            losses.update(loss.item(), inputs.size(0))
            std.update(np.mean(np.abs((outputs-targets).cpu().numpy())), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | STD : {std:.4f} |'.format(
                        batch=batch_idx + 1,
                        size=len(testloader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        std=std.avg,
                        )
            bar.next()
        bar.finish()

    output = torch.vstack(out_list).cpu().detach().numpy()

    predicted=forest.predict(output)
    acc = accuracy_score(gt_list, predicted)
    print('test acc : ',acc)

    return (losses.avg, std.avg, bau/len(testloader))

def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

class ApplyTransform(data.Dataset):
    """
    Apply transformations to a Dataset

    Arguments:
        dataset (Dataset): A Dataset that returns (sample, target)
        transform (callable, optional): A function/transform to be applied on the sample
        target_transform (callable, optional): A function/transform to be applied on the target

    """
    def __init__(self, dataset, transform=None, target_transform=None):
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, idx):
        sample, target, gt = self.dataset[idx]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target, gt

    def __len__(self):
        return len(self.dataset)


class FACE(data.Dataset):
    def __init__(self, data_dir, transform, labels, gt):
        self.data_dir = data_dir
        self.transform = transform
        self.image_list = sorted(glob.glob(self.data_dir + '/*.png'))
        self.labels = labels
        self.gt = gt

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img = Image.open(self.image_list[idx])
        img = img.convert("RGB")
        label = self.labels[idx]
        gt = self.gt[idx]

        return img, label, gt

if __name__ == '__main__':
    main()