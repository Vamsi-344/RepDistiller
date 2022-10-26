import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

import torchvision.models as models
import torchvision.datasets as datasets

import copy
from tqdm import tqdm
from models import model_dict

#print(os.system('nvidia-smi'))
device = 'cpu' if torch.cuda.is_available() else 'cpu'

def compute_calibration_metrics(num_bins=100, net=None, loader=None, device=device):
    """
    Computes the calibration metrics ECE and OE along with the acc and conf values
    :param num_bins: Taken from email correspondence and 100 is used
    :param net: trained network
    :param loader: dataloader for the dataset
    :param device: cuda or cpu
    :return: ECE, OE, acc, conf
    """
    acc_counts = [0 for _ in range(num_bins+1)]
    conf_counts = [0 for _ in range(num_bins+1)]
    overall_conf = []
    n = float(len(loader.dataset))
    counts = [0 for i in range(num_bins+1)]
    net.eval()
    with torch.no_grad():
        for idx, (images, labels) in enumerate(tqdm(loader)):
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confs, preds = probabilities.max(1)
            for (conf, pred, label) in zip(confs, preds, labels):
                bin_index = int(((conf * 100) // (100/num_bins)).cpu())
                try:
                    if pred == label:
                        acc_counts[bin_index] += 1.0
                    conf_counts[bin_index] += conf.cpu()
                    counts[bin_index] += 1.0
                    overall_conf.append(conf.cpu())
                except:
                    print(bin_index, conf)
                    raise AssertionError('Bin index out of range!')


    avg_acc = [0 if count == 0 else acc_count / count for acc_count, count in zip(acc_counts, counts)]
    avg_conf = [0 if count == 0 else conf_count / count for conf_count, count in zip(conf_counts, counts)]
    ECE, OE = 0, 0
    for i in range(num_bins):
        ECE += (counts[i] / n) * abs(avg_acc[i] - avg_conf[i])
        OE += (counts[i] / n) * (avg_conf[i] * (max(avg_conf[i] - avg_acc[i], 0)))

    return ECE, OE, avg_acc, avg_conf, sum(acc_counts) / n, counts

# Data
print('==> Preparing data..')

train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

valset = datasets.CIFAR100('./data', train=False, transform=test_transform)
valloader = torch.utils.data.DataLoader(valset, batch_size=1000, num_workers=2, pin_memory=True)

print(len(valset))

# Model
print('==> Building model..')
net = model_dict['ShuffleV1'](num_classes=100)
#net = models.resnet34()
net.load_state_dict(torch.load('./save_savg_true/student_model/S:ShuffleV1_T:wrn_40_2_cifar100_cmcrd_r:1_a:0.0_b:0.8_1/ShuffleV1_best.pth', map_location='cpu')['model'])
#net = torch.load('ckpt-148.t7', map_location=device)['net']
print('model loaded')

#net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# %pip install prettytable
from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

criterion = nn.CrossEntropyLoss()

for param in net.parameters():
    param.requires_grad = False

count_parameters(net)

### Output dirs for the experiments
ece, oe, acc, conf, A, counts = compute_calibration_metrics(num_bins=100, net=net, loader=valloader)
print('Accuracy: ', A)
print('ECE: ', ece)
print('OE: ', oe)
with open('output.txt', 'w') as wf:
  wf.write('Accuracy: ')
  wf.write(str(A))
  wf.write('\n')
  wf.write('ECE: ')
  wf.write(str(ece))
  wf.write('\n')
  wf.write('OE: ')
  wf.write(str(oe))
  wf.write('\n')
