import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable, grad


class Sobel(nn.Module):
    def __init__(self, k, dtype=torch.float):
        super(Sobel, self).__init__()
        self.k = k
        
        self.conv = nn.Conv2d(k, 2*k, 3, padding=1, padding_mode='replicate', groups=k, bias=False)
        
        for param in self.parameters():
            param.data = torch.tensor([[[-1, -2, -1],
                                 [0,0,0],
                                 [1, 2, 1]],
                                  
                                 [[-1, 0, 1],
                                 [-2,0,2],
                                 [-1, 0, 1]]], dtype=dtype).unsqueeze(1).repeat(k,1,1,1)
            param.requires_grad = False
        
    def forward(self, x):
        return self.conv(x)
    
    def normal(self, x):
        assert self.k == 3 or self.k == 1

        if self.k == 3:
        
            x = self(x)
            
            x = torch.cat([t.unsqueeze(0) for t in [x[:,2]*x[:,5] - x[:,4]*x[:,3], x[:,4]*x[:,1]-x[:,0]*x[:,5], x[:,0]*x[:,3] - x[:,2]*x[:,1]]], dim=1)
            x = x / torch.norm(x, dim=1)
            x[torch.isnan(x)] = 0
            
            return x

        elif self.k == 1:
            w,h = x.shape[2:]
            
            x = torch.cat([torch.zeros_like(x),
                        torch.ones_like(x) / w,
                        torch.ones_like(x) / h,
                        torch_zeros_like(x),
                        self(x)], dim=1)

            x = torch.cat([t.unsqueeze(0) for t in [x[:,2]*x[:,5] - x[:,4]*x[:,3], x[:,4]*x[:,1]-x[:,0]*x[:,5], x[:,0]*x[:,3] - x[:,2]*x[:,1]]], dim=1)

            x = x / torch.norm(x, dim=1)
            x[torch.isnan(x)] = 0
            
            return x


def writePLY_mesh(filename, X, color, eps=0.1):
    w,h = X.shape[2:]

    normal = Sobel(3).normal(X)

    norm = lambda x: torch.norm(x, dim=1)

    e1 = norm(X[:,:,:,:h-1] - X[:,:,:,1:]) < eps                # |
    e2 = norm(X[:,:,:w-1] - X[:,:,1:]) < eps                    # -
    e3 = norm(X[:,:, :w-1, :h-1] - X[:, :, 1:, 1:]) < eps      # \
    
    f1 = e1[:,:,:w-1] & e2[:,:,:,:,1:] & e3     # |\
    f2 = e3 & e1[:,:, 1:] & e2[:,:,:,:h-1]              # \|

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
        u,v = i//h, i%h

        ply_file.write("%f %f %f %f %f %f %d %d %d\n" % (X[:,0,u,v], X[:,1,u,v], X[:,2,u,v], 
                                                        normal[:,0,u,v], normal[:,1,u,v], normal[:,2,u,v], 
                                                        color[:,0,u,v], color[:,1,u,v], color[:,2,u,v]))
    
    for i in range((h-1)*(w-1)):
        u,v = i//(h-1), i%(h-1)
                   
        p0 = u*h+v
        p1 = u*h+v+1
        p2 = (u+1)*h+v
        p3 = (u+1)*h+v+1
        
        if f1[u,v]:
            ply_file.write("3 %d %d %d\n" % (p0, p1, p3))
        if f2[u,v]:
            ply_file.write("3 %d %d %d\n" % (p0, p3, p2))


def normal_from_y(y, x):
    grad = torch.autograd.grad(torch.sum(y), [x], create_graph=True)[0]
    grad = grad / torch.norm(grad, dim=1, keepdim=True)

    return grad


def model_train(model):
    model.train()
    for param in model.parameters():
        param.requires_grad = True

def model_test(model):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
