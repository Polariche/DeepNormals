import torch
import knn_cuda

class knn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, k, return_ind=False):
        dist, ind = knn_cuda.forward(x,y,k)
        ctx.save_for_backward(x, y, dist, ind)

        if return_ind:
            return dist, ind
        else:
            return dist

    @staticmethod
    def backward(ctx, grad_output):
        x, y, dist, ind = ctx.saved_tensors

        d_x, d_y = knn_cuda.backward(grad_output, x, y, dist, ind)
        return d_x, d_y, None