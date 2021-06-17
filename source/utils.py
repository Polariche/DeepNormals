import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable, grad


def writePLY_mesh(filename, X, normal, color, eps=0.1):

    h,w = X.shape[:2]


    norm = lambda x: np.sqrt(np.sum(np.power(x, 2), axis=2))

    e1 = norm(X[:h-1] - X[1:]) < eps                # |
    e2 = norm(X[:, :w-1] - X[:, 1:]) < eps          # -
    e3 = norm(X[:h-1, :w-1] - X[1:, 1:]) < eps      # \
    
    f1 = e1[:,:w-1] & e2[1:] & e3
    f2 = e3 & e1[:, 1:] & e2[:h-1]

    fcount = np.sum(f1) + np.sum(f2)

    ply_file = open(filename,'w')
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n"%(h*w))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("property float nx\n")
    ply_file.write("property float ny\n")
    ply_file.write("property float nz\n")
    ply_file.write("property uchar red\n")
    ply_file.write("property uchar green\n")
    ply_file.write("property uchar blue\n")
    
    ply_file.write("element face %d\n"%(fcount))
    ply_file.write("property list uchar int vertex_index\n")
    
    
    ply_file.write("end_header\n")

    for i in range(h*w):
        u,v = i//w, i%w

        ply_file.write("%f %f %f %f %f %f %d %d %d\n" % (X[u,v,0], X[u,v,1], X[u,v,2], 
                                                        normal[u,v,0], normal[u,v,1], normal[u,v,2], 
                                                        color[u,v,0], color[u,v,1], color[u,v,2]))
    for i in range((h-1)*(w-1)):
        u,v = i//(w-1), i%(w-1)
                   
        p0 = u*w+v
        p1 = u*w+v+1
        p2 = (u+1)*w+v
        p3 = (u+1)*w+v+1
        
        if f1[u,v]:
            ply_file.write("3 %d %d %d\n" % (p0, p2, p3))
        if f2[u,v]:
            ply_file.write("3 %d %d %d\n" % (p0, p3, p1))


def writePLY(filename, X):
    ply_file = open(filename,'w')
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n"%(X.shape[0]))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("end_header\n")

    for i in range(X.shape[0]):
        ply_file.write("%f %f %f\n" % (X[i,0], X[i,1], X[i,2]))


def model_train(model):
    model.train()
    for param in model.parameters():
        param.requires_grad = True

def model_test(model):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False