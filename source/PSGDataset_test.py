import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import sys
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

from models.models import Siren
from loaders import PSGDataset
from models.LGD import LGD, detach_var

import argparse

parser = argparse.ArgumentParser(description='Test',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('data', metavar='DATA', help='path to file')
args = parser.parse_args()

psg = PSGDataset(args.data)

print(psg[0])
print(psg[0]['pc_gt'])
print(psg[0]['pc_pred'])




