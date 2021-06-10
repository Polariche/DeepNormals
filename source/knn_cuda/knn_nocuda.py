import torch


if __name__ == "__main__":
    x = torch.randn(3,2).cuda().requires_grad_()
    y = torch.randn(3,2).cuda().requires_grad_()
    k = 3


    torch.sqrt(torch.pow(x.unsqueeze(0) - y.unsqueeze(1), 2).sum(dim=2)).sum().backward()

    print(x.grad, y.grad)




