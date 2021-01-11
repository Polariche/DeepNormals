import numpy as np
import cv2

from models import DeepSDF, PositionalEncoding

import argparse

parser = argparse.ArgumentParser(description='Structure from Motion Learner training on KITTI and CityScapes Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--dataset', dest='dataset', metavar='PATH',
                    help='dataset')
parser.add_argument('--pe', dest='pe', metavar='PE', default=True,
                    help='positional encoding')

args = parser.parse_args()



