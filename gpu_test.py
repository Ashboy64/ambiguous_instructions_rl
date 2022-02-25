import os
import torch

DEVICE_IDX = 0

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_IDX)

print("DEVICE_IDX:", DEVICE_IDX)
print("torch.cuda.current_device():", torch.cuda.current_device())
print(f"torch.cuda.device({DEVICE_IDX}):", torch.cuda.device(DEVICE_IDX))
print("torch.cuda.device_count():", torch.cuda.device_count())
print(f"torch.cuda.get_device_name({DEVICE_IDX}):", torch.cuda.get_device_name(DEVICE_IDX))


device = torch.device("cuda")

x = torch.randn((3, 4)).to(device)
y = torch.ones((3, 4)).to(device)
z = x + y
print("x:", x)
print("y:", y)
print("x + y:", z)
