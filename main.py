import os
import argparse
import random

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms as T
from torchvision.utils import make_grid, save_image

import train
from model import build_discriminator, build_generator


parser = argparse.ArgumentParser()

# General training
parser.add_argument('--dataset', default='cifar10', help='mnist | cifar10')
parser.add_argument('--dataroot', type=str, default='./data', help='data path')
parser.add_argument('--batch_size', type=int, default=16, help='num of batch size')
parser.add_argument('--image_size', type=int, default=64, help='image size')
parser.add_argument('--workers', type=int, default=4, help='num of loading workers')
parser.add_argument('--model', type=str, default='dcgan', help='dcgan | presgan')
parser.add_argument('--num_epochs', type=int, default=100, help='num of epochs')
parser.add_argument('--nz', type=int, default=100, help='noise size')
parser.add_argument('--nc', type=int, default=3, help='image dimension, color:3 | greyscale:1')
parser.add_argument('--ngf', type=int, default=64, help='generator filter size')
parser.add_argument('--ndf', type=int, default=64, help='discriminator filter size')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
parser.add_argument('--seed', type=int, default=2020, help='manual seed')
parser.add_argument('--ngpu', type=int, default=1, help='num of gpus')
parser.add_argument('--checkpoint_path', type=str, default='./checkpoint', help='checkpoint dir')

# PresGAN
parser.add_argument('--logsigma_init', type=float, default=-1.0, help='init value for log_sigma_sian')
parser.add_argument('--sigma_lr', type=float, default=0.0002, help='learning rate for sigma')
parser.add_argument('--lambda_', type=float, default=0.01, help='entropy coeff')
parser.add_argument('--stepsize_num', type=float, default=1.0, help='init value for hmc stepsize')
parser.add_argument('--burn_in', type=int, default=2, help='hmc burn in')
parser.add_argument('--num_samples_posterior', type=int, default=2, help='num of samples from posterior')
parser.add_argument('--leapfrog_steps', type=int, default=5, help='num of leap frog steps for hmc')
parser.add_argument('--flag_adapt', type=int, default=1, help='0 or 1')
parser.add_argument('--hmc_learning_rate', type=float, default=0.02, help='hmc learning rate')
parser.add_argument('--hmc_opt_accept', type=float, default=0.67, help='hmc optimal acceptance rate')

args = parser.parse_args()


# create data dir
if not os.path.exists(args.dataroot):
    os.mkdir(args.dataroot)
# create checkpoint dir
if not os.path.exists(args.checkpoint_path):
    os.mkdir(args.checkpoint_path)
if not os.path.exists(os.path.join(args.checkpoint_path, args.dataset)):
    os.mkdir(os.path.join(args.checkpoint_path, args.dataset))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
args.device = device

transforms = [T.Resize(args.image_size), T.ToTensor()]
if args.nc==3:
    transforms.append(T.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)))
else:
    transforms.append(T.Normalize((0.5),(0.5)))

print('Generating data...')
torch.manual_seed(args.seed)
if args.dataset=='cifar10':
    data = torchvision.datasets.CIFAR10(args.dataroot, train=True, transform=T.Compose(transforms), download=True)
    data_test = torchvision.datasets.CIFAR10(args.dataroot, train=False, transform=T.Compose(transforms), download=True)
elif args.dataset=='mnist':
    data = torchvision.datasets.MNIST(args.dataroot, train=True, transform=T.Compose(transforms), download=True)
    data_test = torchvision.datasets.MNIST(args.dataroot, train=False, transform=T.Compose(transforms), download=True)
else:
    print('Dataset is not implemented!!!')

data_loader = DataLoader(data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
data_loader_test = DataLoader(data_test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

print('Building model...')
netG = build_generator(args)
netD = build_discriminator(args)

if args.model=='dcgan':
    print('DCGAN')
    train.dcgan(data_loader, netG, netD, args)
elif args.model=='presgan':
    log_sigma = torch.tensor([args.logsigma_init]*(args.image_size*args.image_size), device=device, requires_grad=True)
    train.presgan(data_loader, netG, netD, log_sigma, args)
else:
    print('Module is not implemented!!!')


