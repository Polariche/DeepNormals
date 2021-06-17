import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import sys
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

from models.models import Siren
from models.LGD import LGD, detach_var

import argparse

class PSGDataset(Dataset):
    # Dataset for (2D img, GT pointcloud data, validation)

    def __init__(self, data_path):
        self.datadir = data_path

    def __len__(self):
        return 300000

    def __getitem__(self, bno):
        FETCH_BATCH_SIZE=32
        BATCH_SIZE=32
        HEIGHT=192
        WIDTH=256
        POINTCLOUDSIZE=16384
        OUTPUTPOINTS=1024
        REEBSIZE=1024

        path = os.path.join(self.datadir,'%d/%d.gz'%(bno//1000,bno))
        if not os.path.exists(path):
            self.stopped=True
            print ("error! data file not exists: %s"%path)
            print ("please KILL THIS PROGRAM otherwise it will bear undefined behaviors")
            assert False,"data file not exists: %s"%path
        gz = open(path,'r', encoding='utf-16').read()
        binfile=zlib.decompress(gz)
        p=0
        color=np.fromstring(binfile[p:p+FETCH_BATCH_SIZE*HEIGHT*WIDTH*3],dtype='uint8').reshape((FETCH_BATCH_SIZE,HEIGHT,WIDTH,3))
        p+=FETCH_BATCH_SIZE*HEIGHT*WIDTH*3
        depth=np.fromstring(binfile[p:p+FETCH_BATCH_SIZE*HEIGHT*WIDTH*2],dtype='uint16').reshape((FETCH_BATCH_SIZE,HEIGHT,WIDTH))
        p+=FETCH_BATCH_SIZE*HEIGHT*WIDTH*2
        rotmat=np.fromstring(binfile[p:p+FETCH_BATCH_SIZE*3*3*4],dtype='float32').reshape((FETCH_BATCH_SIZE,3,3))
        p+=FETCH_BATCH_SIZE*3*3*4
        ptcloud=np.fromstring(binfile[p:p+FETCH_BATCH_SIZE*POINTCLOUDSIZE*3],dtype='uint8').reshape((FETCH_BATCH_SIZE,POINTCLOUDSIZE,3))
        ptcloud=ptcloud.astype('float32')/255
        beta=math.pi/180*20
        viewmat=np.array([[
            np.cos(beta),0,-np.sin(beta)],[
            0,1,0],[
            np.sin(beta),0,np.cos(beta)]],dtype='float32')
        rotmat=rotmat.dot(np.linalg.inv(viewmat))
        for i in xrange(FETCH_BATCH_SIZE):
            ptcloud[i]=((ptcloud[i]-[0.7,0.5,0.5])/0.4).dot(rotmat[i])+[1,0,0]
        p+=FETCH_BATCH_SIZE*POINTCLOUDSIZE*3
        reeb=np.fromstring(binfile[p:p+FETCH_BATCH_SIZE*REEBSIZE*2*4],dtype='uint16').reshape((FETCH_BATCH_SIZE,REEBSIZE,4))
        p+=FETCH_BATCH_SIZE*REEBSIZE*2*4
        keynames=binfile[p:].split('\n')
        reeb=reeb.astype('float32')/65535
        for i in xrange(FETCH_BATCH_SIZE):
            reeb[i,:,:3]=((reeb[i,:,:3]-[0.7,0.5,0.5])/0.4).dot(rotmat[i])+[1,0,0]
        data=np.zeros((FETCH_BATCH_SIZE,HEIGHT,WIDTH,4),dtype='float32')
        data[:,:,:,:3]=color*(1/255.0)
        data[:,:,:,3]=depth==0
        validating=np.array([i[0]=='f' for i in keynames],dtype='float32')

        data = torch.tensor(data).permute(0,3,1,2)
        ptcloud = torch.tensor(ptcloud)
        validating = torch.tensor(validating)

        return {'img': data, 'pc_gt': ptcloud, 'validating': validating}

parser = argparse.ArgumentParser(description='Test',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('data', metavar='DATA', help='path to file')
args = parser.parse_args()

psg = PSGDataset(args.data)

print(psg[0])
print(psg[0]['pc_gt'])
print(psg[0]['pc_pred'])




