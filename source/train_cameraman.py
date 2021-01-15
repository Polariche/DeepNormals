import numpy as np
import cv2
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from utils import Sobel
import utils

import argparse

from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import skimage
import matplotlib.pyplot as plt

"""
explore_siren.ipynb's basecode
"""

parser = argparse.ArgumentParser(description='Test',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('data', metavar='DATA', help='path to file')

parser.add_argument('--tb-save-path', dest='tb_save_path', metavar='PATH', default='../checkpoints/', 
                        help='tensorboard checkpoints path')

parser.add_argument('--weight-save-path', dest='weight_save_path', metavar='PATH', default='../weights/', 
                        help='weight checkpoints path')

parser.add_argument('--pretrained-weight', dest='weight', metavar='PATH', default=None, 
                        help='pretrained weight')

parser.add_argument('--activation', dest='activation', metavar='activation', default='relu', 
                        help='activation of network; \'relu\' or \'sin\'')

parser.add_argument('--batchsize', dest='batchsize', type=int, metavar='BATCHSIZE', default=1,
                        help='batch size')
parser.add_argument('--epoch', dest='epoch', type=int,metavar='EPOCH', default=100, 
                        help='epochs')

parser.add_argument('--epsilon', dest='epsilon', type=float, metavar='EPSILON', default=0.1, 
                        help='epsilon')
parser.add_argument('--omega', dest='omega', type=float, metavar='OMEGA', default=30, 
                        help='hyperparameter for periodic layer')
parser.add_argument('--lambda', dest='lamb', type=float, metavar='LAMBDA', default=0.005, 
                        help='hyperparameter for z : tangent loss ratio')


parser.add_argument('--outfile', dest='outfile', metavar='OUTFILE', 
                        help='output file')

class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
    def forward_with_intermediate(self, input): 
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate
    
    
class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords        

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)
                
                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()
                    
                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else: 
                x = layer(x)
                
                if retain_grad:
                    x.retain_grad()
                    
            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations
        
def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

def get_cameraman_tensor(sidelength):
    img = Image.fromarray(skimage.data.camera())        
    transform = Compose([
        Resize(sidelength),
        ToTensor(),
        Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ])
    img = transform(img)
    return img

class ImageFitting(Dataset):
    def __init__(self, sidelength):
        super().__init__()
        img = get_cameraman_tensor(sidelength)
        self.pixels = img.permute(1, 2, 0).view(-1, 1)
        self.coords = get_mgrid(sidelength, 2)

    def __len__(self):
        return 1

    def __getitem__(self, idx):    
        if idx > 0: raise IndexError
            
        return self.coords, self.pixels

def main():
    args = parser.parse_args()

    writer = SummaryWriter(args.tb_save_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create models
    model = Siren(in_features=2, out_features=1, hidden_features=256, hidden_layers=3, outermost_linear=True).to(device) 

    optimizer = optim.Adam(model.parameters(), lr = 1e-3)

    if args.weight != None:
        try:
            model.load_state_dict(torch.load(args.pretrained_weight))
        except:
            print("Couldn't load pretrained weight: " + args.pretrained_weight)

    cameraman = ImageFitting(256)
    dataloader = DataLoader(cameraman, batch_size=1, pin_memory=True, num_workers=0)

    model_input, ground_truth = next(iter(dataloader))
    model_input, ground_truth = model_input.cuda(), ground_truth.cuda()

    
    for step in range(args.epoch):
        model_output, coords = model(model_input)    
        loss = ((model_output - ground_truth)**2).mean()
        
        if not step % 10:
            img_grad = gradient(model_output, coords)
            img_laplacian = laplace(model_output, coords)

            writer.add_images("result : z", model_output.cpu().view(256,256,1).repeat(1,1,3).detach().numpy(), step, dataformats='HWC')
            writer.add_images("result : grad", img_grad.norm(dim=-1).cpu().view(256,256,1).repeat(1,1,3).detach().numpy(), step, dataformats='HWC')
            writer.add_images("result : laplacian", img_laplacian.cpu().view(256,256,1).repeat(1,1,3).detach().numpy(), step, dataformats='HWC')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    
    writer.close()


if __name__ == "__main__":
    main()
