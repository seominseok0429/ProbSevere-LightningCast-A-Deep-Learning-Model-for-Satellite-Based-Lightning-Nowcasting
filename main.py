import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import torchvision.datasets as datasets

from dataloader import GLMLoader
from models.unet import UNet
from utils import progress_bar


def calculate_csi(predictions, targets, threshold):
    """
    Calculate the Critical Success Index (CSI) for a given threshold.
    """
    preds_binary = (predictions > threshold).float()
    tp = (preds_binary * targets).sum().item()  # True Positives
    fp = (preds_binary * (1 - targets)).sum().item()  # False Positives
    fn = ((1 - preds_binary) * targets).sum().item()  # False Negatives

    csi = tp / (tp + fp + fn + 1e-8)  # Adding a small value to avoid division by zero
    return csi

def print_csi_scores(outputs, targets):
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    for threshold in thresholds:
        csi_score = calculate_csi(outputs, targets, threshold)
        print(f"Threshold: {threshold:.2f} - CSI: {csi_score:.4f}")


device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


traindataset = GLMLoader() 
train_loader = torch.utils.data.DataLoader(traindataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

print('==> Building model..')
net = UNet(n_channels=4, n_classes=1)
net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.BCELoss()

optimizer = optim.Adam(net.parameters(), lr=0.001)

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,30])

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.float().to(device), targets.float().to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        print_csi_scores(outputs, targets)
        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f |  (%d/%d)'
                     % (train_loss/(batch_idx+1),  correct, total))
    scheduler.step()
    save_path = './checkpoint/' + str(epoch) + '.pth'
    torch.save(net.state_dict(), save_path)

for epoch in range(start_epoch, 90):
    train(epoch)
