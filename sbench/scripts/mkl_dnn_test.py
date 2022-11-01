import torch
print(f'Pytorch version : {torch.__version__}')
print(*torch.__config__.show().split("\n"), sep="\n")

from torchvision import models
from torch.utils import mkldnn as mkldnn_utils
import time


def forward(net, use_mkldnn=False, iteration=1, batch_size=10):
  net.eval()
  batch = torch.rand(batch_size, 3, 512, 512)
  if use_mkldnn:
    net = mkldnn_utils.to_mkldnn(net)
    batch = batch.to_mkldnn()

  start_time = time.time()
  for i in range(iteration):
      #with torch.no_grad():
      net(batch)
  return time.time() - start_time


net = models.resnet18(False)
iter_cnt = 100
batch_size = 1
no_mkldnn   = forward(net, False, iter_cnt, batch_size)
with_mkldnn = forward(net, True,  iter_cnt, batch_size)

print(f"time-normal: {no_mkldnn:.4f}s")
print(f"time-mkldnn: {with_mkldnn:.4f}s")
print(f"mkldnn is {no_mkldnn/with_mkldnn:.2f}x faster!")

