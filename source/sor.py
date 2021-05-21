from torch.utils.tensorboard import SummaryWriter
import pcl
import numpy as np
import torch

import argparse


parser = argparse.ArgumentParser(description='Test',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument('--tb-save-path', dest='tb_save_path', metavar='PATH', default='../checkpoints/', 
                        help='tensorboard checkpoints path')

parser.add_argument('--k', dest='k', metavar='K', type=int, default=50, 
                        help='number of neighborhoods for KNN')

parser.add_argument('--stdev-thresh', dest='stdev_thres', metavar='STDEV_THRES', type=float, default=1.0, 
                        help='Standard deviation threshold for SOR')


args = parser.parse_args()

writer = SummaryWriter(args.tb_save_path)

# Statistical Outlier Removal (SOR) from PCL
cloud = pcl.PointCloud()
x_cpu = np.load(args.tb_save_path+'/point.npy')
cloud.from_array(x_cpu)

outrem = cloud.make_statistical_outlier_filter()
outrem.set_mean_k(args.k)
outrem.set_std_dev_mul_thresh(args.stdev_thres)
filtered_cloud = outrem.filter()
filtered_x = np.asarray(filtered_cloud)

writer.add_mesh("filtered LGD", torch.tensor(filtered_x).unsqueeze(0), global_step=args.k)

writer.close()