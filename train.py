import argparse
import sys
import os
import pickle
import numpy as np 
import pickle, gzip
import shutil
import logging
from collections import OrderedDict
# from tabulate import tabulate

import torch
import torch.nn.functional as F 
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.parameter import Parameter 
import time

import random

from loader import MyDataLoader
from model import SuperResolution, Transpose_Model

best_loss = 100000


def seed_torch(seed):
  random.seed(seed)
  np.random.seed(seed)
  os.environ['PYTHONHASHSEED']=str(seed)
  os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed) # multiple gpu
  torch.manual_seed(seed)

  torch.use_deterministic_algorithms(True)
  torch.backends.cudnn.deterministric=True
  torch.backends.cudnn.enabled=False
  torch.backends.cudnn.benchmark = False



def getPSNRLoss():
  mseloss_fn = nn.MSELoss(reduction='none')

  def PSNRLoss(output, target):
    loss = mseloss_fn(output, target)
    loss = torch.mean(loss, dim=(1,2))
    loss = 10 * torch.log10(loss)
    mean = torch.mean(loss)
    return mean

  return PSNRLoss

loss_function = getPSNRLoss()

def train(args, model, train_loader, optimizer, epoch, device):
  model.train()
  tot_loss = 0.
  count = 0.
  for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = loss_function(output, target)
    loss.backward()
    optimizer.step()
    tot_loss += loss.item() * data.size()[0]
    count += data.size()[0]

  tot_loss /= count
  print('Train Epoch: {} Loss: {:.6f}'.format(epoch, tot_loss),flush=True)
  return tot_loss

def test(args, model, test_loader, epoch, device):
  global best_loss
  model.eval()
  test_loss = 0
  count = 0
  best = 100
  tot_time = 0.0

  with torch.no_grad():
    for batch_idx, (data, target) in enumerate(test_loader):
      data, target = data.to(device), target.to(device)
      start_time = time.time()
      output = model(data)
      diff = time.time()-start_time
      tot_time += diff
      loss = loss_function(output, target)

      test_loss += loss.item() * data.size()[0]
      count += data.size()[0]

    test_loss /= count
    per_time = tot_time/count
    print('Avg inference time: {:.5f}'.format(per_time), flush=True)
    print('Test Epoch: {} Loss: {:.6f}'.format(epoch, test_loss),flush=True)
  return test_loss


def isnotin(it):
  buffer = ['G','F2V','L','NS','EW','rot_NS',
            'unique','separated_src_idx',
            'NS_grad_op','EW_grad_op','rot_NS_45_grad_op',
            'rot_NS_90_grad_op','rot_NS_135_grad_op','rot_NS_180_grad_op',
            'unique_mat','all_grad_ops']
  for b in buffer:
    if b in it.split('.'):
      return False
  return True


def main():
  parser = argparse.ArgumentParser(description='Spherical Super Resolution')
  parser.add_argument('--model_idx',type=int,default=0,metavar='N',
    help= 'model index')
  parser.add_argument('--batch-size', type = int, default = 64, metavar = 'N',
    help = 'input batch size for training (default: 64)')
  parser.add_argument('--test-batch-size', type = int, default = 64, metavar = 'N',
    help = 'input batch size for testing (default: 64)')
  parser.add_argument('--epochs', type = int, default = 100, metavar = 'N',
    help = 'number of epochs to train (default: 100)')
  parser.add_argument('--lr', type = float, default = 5e-3,metavar = 'LR',
    help = 'learning rate (default: 0.005')
  parser.add_argument('--no-cuda', action = 'store_true', default = False,
    help = 'disables CUDA training')
  parser.add_argument('--seed', type=int, default=0, metavar='S',
        help='random seed (default: 0)')
  parser.add_argument('--mesh_folder', type=str, default="mesh_files",
        help='path to mesh folder (default: mesh_files)')
  parser.add_argument('--max_level', type=int, default=9, help='max mesh level')
  parser.add_argument('--min_level', type=int, default=7, help='min mesh level')
  parser.add_argument('--log-interval', type=int, default=100, metavar='N',
        help='how many batches to wait before logging training status')
  parser.add_argument('--feat', type=int, default=16, help='filter dimensions')
  parser.add_argument('--decay', action="store_true", help="switch to decay learning rate")
  parser.add_argument('--optim', type=str, default="adam", choices=["adam", "sgd"])
  parser.add_argument('--in_ch', type=str, default="rgb", choices=["rgb", "rgbd"], help="input channels")

  parser.add_argument('--data_folder', type=str, default="data",
        help='path to data folder (default: data)')
  parser.add_argument('--load', type = int, default = 0,
    help='Load model or not')
  parser.add_argument('--video_name', type = str, default = 'NA',
    help='video name')
  parser.add_argument('--up_method', type = str, default = 'SSR2',
    help='upscale method ("Trans", "SSR1", "SSR2")')



  args = parser.parse_args()

  use_cuda = not args.no_cuda and torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")

  seed_torch(args.seed)

  data_path = args.video_name+'_data'

  trainset = MyDataLoader(data_path, sp_level = args.max_level, min_level = args.min_level, in_ch=len(args.in_ch))
  testset = MyDataLoader(data_path, sp_level = args.max_level, min_level = args.min_level, in_ch = len(args.in_ch))

  train_loader = DataLoader(trainset, batch_size = args.batch_size, shuffle = True)
  test_loader = DataLoader(testset, batch_size = args.test_batch_size, shuffle = True)

  if args.up_method == "Trans":
    assert (args.max_level-args.min_level == 2)
    model = Transpose_Model(mesh_folder=args.mesh_folder, in_ch=len(args.in_ch), out_ch=len(args.in_ch), \
                          max_level=args.max_level, min_level=args.min_level, fdim=args.feat)
  else:
    model = SuperResolution(mesh_folder=args.mesh_folder, in_ch=len(args.in_ch), out_ch=len(args.in_ch), \
                          max_level=args.max_level, min_level=args.min_level, fdim=args.feat,method=args.up_method)

  # for multiple GPU use
  model = nn.DataParallel(model)
  if args.optim == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
  else:
    optimizer = optim.Adam(model.parameters(), lr = args.lr)

  if args.decay:
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 3, gamma = 0.9)


  start_epoch = 0
  best_loss = 100

  cpk_name = args.video_name+'_checkpoint'
  if args.load == 1:
    print('------------Load Model-----------------')
    assert os.path.isdir(cpk_name)
    checkpoint = torch.load('./'+cpk_name+'/Model'+str(args.model_idx-1))
    state = checkpoint['state_dict']
    
    def load_my_state_dic(self, state_dict, exclude = 'none'):
      own_state = self.state_dict()
      for name, param in state_dict.items():
        if name not in own_state:
          continue
        if exclude in name:
          continue
        if isinstance(param, Parameter):
          param = param.data
        own_state[name].copy_(param)

    load_my_state_dic(model, state)

  print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
  print('----------------video: ', args.video_name)
  print('----------------model_idx: ', args.model_idx)

  for epoch in range(start_epoch+1, start_epoch+args.epochs+1):
    start_time = time.time()
    train_loss = train(args, model, train_loader, optimizer, epoch, device)
    if args.decay:
      scheduler.step()
    end_time = time.time()
    period = end_time-start_time
    print('Train Time for each epoch: {}'.format(period))

    start_time = time.time()
    test_loss = test(args, model, test_loader, epoch, device)
    end_time = time.time()
    period = end_time-start_time
    print('Test Time for each epoch: {}'.format(period))

    state_dict_no_buffer = [it for it in model.state_dict().items() if isnotin(it[0])]
    state_dict_no_buffer = OrderedDict(state_dict_no_buffer)

    if test_loss < best_loss:
      print('---------------------Save Model------------------')
      state = {
      'state_dict': state_dict_no_buffer,
      # 'epoch': epoch,
      'best_loss': test_loss,
      # 'optimizer': optimizer.state_dict(),
      }
      if not os.path.isdir(cpk_name):
        os.mkdir(cpk_name)

      torch.save(state, './'+cpk_name+'/Model'+str(args.model_idx)+'_seed'+str(args.seed))
      best_loss = test_loss

if __name__ == '__main__':
  main()
