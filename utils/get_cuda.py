import torch
def get_cuda():
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    return cuda,device
