import torch
print(torch.__version__)
print(torch.version.cuda)
# 输出为True，则安装无误
print(torch.cuda.is_available())  