import torch
from model import SiameseUNet

model = SiameseUNet()

x1 = torch.randn(2, 1, 256, 256)
x2 = torch.randn(2, 1, 256, 256)

y = model(x1, x2)

print("Input:", x1.shape)
print("Output:", y.shape)