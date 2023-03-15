
import torch
import numpy as np
import random
import os
os.chdir('/disk/xray0/tl3/project-cGAN/repos/rcGAN/')
import sys
sys.path.append('/disk/xray0/tl3/project-cGAN/repos/rcGAN/')
sys.path.append('/disk/xray0/tl3/project-cGAN/repos/')

# from rcGAN import train as train_model
from rcGAN import train as train_model



cuda = True if torch.cuda.is_available() else False
torch.backends.cudnn.benchmark = True

# Mimic argument loader
args = train_model.create_arg_parser().parse_args()
args.data_parallel = False
args.is_mri = True
args.resume = False
args.train_gif = False
args.device = 0
args.plot_dir = ''
args.num_plots = None


# restrict visible cuda devices
if args.data_parallel or (args.device >= 0):
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

train_model.train(args)


