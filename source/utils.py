import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable, grad
from glob import glob
import os
import imageio
import skimage
import cv2

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



# implementation from SRN
def parse_intrinsics(filepath, trgt_sidelength=None, invert_y=False):
    # Get camera intrinsics
    with open(filepath, 'r') as file:
        f, cx, cy, _ = map(float, file.readline().split())
        grid_barycenter = torch.Tensor(list(map(float, file.readline().split())))
        scale = float(file.readline())
        height, width = map(float, file.readline().split())

        try:
            world2cam_poses = int(file.readline())
        except ValueError:
            world2cam_poses = None

    if world2cam_poses is None:
        world2cam_poses = False

    world2cam_poses = bool(world2cam_poses)

    if trgt_sidelength is not None:
        cx = cx/width * trgt_sidelength
        cy = cy/height * trgt_sidelength
        f = trgt_sidelength / height * f

    fx = f
    if invert_y:
        fy = -f
    else:
        fy = f

    # Build the intrinsic matrices
    full_intrinsic = np.array([[fx, 0., cx, 0.],
                               [0., fy, cy, 0],
                               [0., 0, 1, 0],
                               [0, 0, 0, 1]])

    return full_intrinsic, grid_barycenter, scale, world2cam_poses

def model_train(model):
    model.train()
    for param in model.parameters():
        param.requires_grad = True

def model_test(model):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False



def load_rgb(path, sidelength=None):
    img = imageio.imread(path)[:, :, :3]
    img = skimage.img_as_float32(img)

    img = square_crop_img(img)

    if sidelength is not None:
        img = cv2.resize(img, (sidelength, sidelength), interpolation=cv2.INTER_AREA)

    img -= 0.5
    img *= 2.
    img = img.transpose(2, 0, 1)
    return img


def load_depth(path, sidelength=None):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)

    if sidelength is not None:
        img = cv2.resize(img, (sidelength, sidelength), interpolation=cv2.INTER_NEAREST)

    img *= 1e-4

    if len(img.shape) == 3:
        img = img[:, :, :1]
        img = img.transpose(2, 0, 1)
    else:
        img = img[None, :, :]
    return img


def load_pose(filename):
    lines = open(filename).read().splitlines()
    if len(lines) == 1:
        pose = np.zeros((4, 4), dtype=np.float32)
        for i in range(16):
            pose[i // 4, i % 4] = lines[0].split(" ")[i]
        return pose.squeeze()
    else:
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines[:4])]
        return np.asarray(lines).astype(np.float32).squeeze()


def load_params(filename):
    lines = open(filename).read().splitlines()

    params = np.array([float(x) for x in lines[0].split()]).astype(np.float32).squeeze()
    return params

def glob_imgs(path):
    imgs = []
    for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG']:
        imgs.extend(glob(os.path.join(path, ext)))
    return imgs



def square_crop_img(img):
    min_dim = np.amin(img.shape[:2])
    center_coord = np.array(img.shape[:2]) // 2
    img = img[center_coord[0] - min_dim // 2:center_coord[0] + min_dim // 2,
          center_coord[1] - min_dim // 2:center_coord[1] + min_dim // 2]
    return img


def train_val_split(object_dir, train_dir, val_dir):
    dirs = [os.path.join(object_dir, x) for x in ['pose', 'rgb', 'depth']]
    data_lists = [sorted(glob(os.path.join(dir, x)))
                  for dir, x in zip(dirs, ['*.txt', "*.png", "*.png"])]

    cond_mkdir(train_dir)
    cond_mkdir(val_dir)

    [cond_mkdir(os.path.join(train_dir, x)) for x in ['pose', 'rgb', 'depth']]
    [cond_mkdir(os.path.join(val_dir, x)) for x in ['pose', 'rgb', 'depth']]

    shutil.copy(os.path.join(object_dir, 'intrinsics.txt'), os.path.join(val_dir, 'intrinsics.txt'))
    shutil.copy(os.path.join(object_dir, 'intrinsics.txt'), os.path.join(train_dir, 'intrinsics.txt'))

    for data_name, data_ending, data_list in zip(['pose', 'rgb', 'depth'], ['.txt', '.png', '.png'], data_lists):
        val_counter = 0
        train_counter = 0
        for i, item in enumerate(data_list):
            if not i % 3:
                shutil.copy(item, os.path.join(train_dir, data_name, "%06d" % train_counter + data_ending))
                train_counter += 1
            else:
                shutil.copy(item, os.path.join(val_dir, data_name, "%06d" % val_counter + data_ending))
                val_counter += 1


def project(X, P):
    # P : (..., m, 1, 6)
    # X : (..., 1, n, 3)

    pshp = P.shape
    P = decode_P(P).view(*pshp[:-2], 4, 4).transpose(-2, -1)     # (..., m, 4, 4)

    X = torch.cat([X, torch.ones_like(X)[..., :1]], dim=-1)     # (..., 1, n, 4)
    X = torch.matmul(X, P)                                      # (..., m, n, 4)
    X = X[..., :3]                                              # (..., m, n, 3)
    x_hat = X[..., :-1] / X[..., -1:]                           # (..., m, n, 2)

    return x_hat

# referenced https://learnopencv.com/rotation-matrix-to-euler-angles/
def decode_P(P):
    # euler+trans -> 4x4 mat
    shp = P.shape
    P = P.view(-1, 6)

    decoded_P = torch.zeros(P.shape[0], 4, 4, device=P.device)
    

    c0 = torch.cos(P[..., 0]).unsqueeze(-1)
    s0 = torch.sin(P[..., 0]).unsqueeze(-1)
    c1 = torch.cos(P[..., 1]).unsqueeze(-1)
    s1 = torch.sin(P[..., 1]).unsqueeze(-1)
    c2 = torch.cos(P[..., 2]).unsqueeze(-1)
    s2 = torch.sin(P[..., 2]).unsqueeze(-1)

    one = torch.ones_like(P[..., 0]).unsqueeze(-1)
    zero = torch.zeros_like(P[..., 0]).unsqueeze(-1)

    P_x = torch.cat([one, zero, zero, 
                    zero, c0, -s0, 
                    zero, s0, c0], dim=-1).view(-1, 3, 3)

    P_y = torch.cat([c1, zero, s1, 
                    zero, one, zero, 
                    -s1, zero, c1], dim=-1).view(-1, 3, 3)

    P_z = torch.cat([c2, -s2, zero, 
                    s2, c2, zero, 
                    zero, zero, one], dim=-1).view(-1, 3, 3)

    decoded_P[..., :3, :3] = torch.matmul(P_z, torch.matmul(P_y, P_x))
    decoded_P[..., 3, 3] = 1
    decoded_P[..., :3, 3] = P[..., 3:]

    decoded_P.view(*shp[:-2], 4, 4)
    
    return decoded_P



def encode_P(P):
    print(P)
    # 4x4 mat -> euler+trans

    shp = P.shape
    P = P.view(-1, 4,4)

    sy = torch.sqrt(P[..., 0,0]*P[..., 0,0] + P[..., 1,1]*P[..., 1,1])
    singular = sy < 1e-6
    
    encoded_P = torch.zeros(P.shape[0], 6, device=P.device)

    # transform
    encoded_P[..., 3:] = P[..., :3, 3]
    
    if (~singular).sum() > 0:
        encoded_P[~singular, 0] = torch.atan2(P[~singular, 2, 1], P[~singular, 2, 2])
        encoded_P[~singular, 1] = torch.atan2(-P[~singular, 2, 0], sy[~singular])
        encoded_P[~singular, 2] = torch.atan2(P[~singular, 1, 0], P[~singular, 0, 0])

    if singular.sum() > 0:
        encoded_P[singular, 0] = torch.atan2(-P[singular, 1, 2], P[singular, 1, 1])
        encoded_P[singular, 1] = torch.atan2(-P[singular, 2, 0], sy[singular])
        encoded_P[singular, 2] = 0

    encoded_P = encoded_P.view(*shp[:-2], 1, 6)
    
    return encoded_P

    



