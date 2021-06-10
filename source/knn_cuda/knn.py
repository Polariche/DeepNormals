import torch
import knn_cuda


class knn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, k):
        dist, ind = knn_cuda.forward(x,y,k)
        ctx.save_for_backward(x, y, dist, ind)

        return dist

    @staticmethod
    def backward(ctx, grad_output):
        x, y, dist, ind = ctx.saved_tensors

        d_x, d_y = knn_cuda.backward(grad_output, x, y, dist, ind)
        return d_x, d_y, None


if __name__ == "__main__":
    x = torch.randn(33,2).cuda().requires_grad_()
    y = torch.randn(33,2).cuda().requires_grad_()
    k = 3


    torch.sqrt(torch.pow(x.unsqueeze(0) - y.unsqueeze(1), 2).sum(dim=2)).sum().backward()

    print(x.grad, y.grad)

    x.grad = None
    y.grad = None

    knn_f = knn.apply
    dist = knn_f(x,y,k)
    dist.sum().backward()

    print(x.grad, y.grad)



