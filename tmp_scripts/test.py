import torch
import time
import torchvision.models as models
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
#from config import device

model = models.densenet121(pretrained=True)
model.to(torch.device('cpu'))
x = torch.randn((1, 3, 224, 224), requires_grad=True)

start = time.process_time()
with torch.autograd.profiler.profile(use_cuda=True) as profile:
   model(x) 
elapsed = time.process_time() - start

print('profiler', prof.self_cpu_time_total())
print('process_time', elapsed)
print(profile)
