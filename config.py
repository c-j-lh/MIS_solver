import torch
from GPUtil import getGPUs
from sys import stderr
GPUs = getGPUs()
GPUs = sorted([GPU for GPU in GPUs if GPU != 3], key=lambda gpu:gpu.load)

if GPUs:
    lightest = GPUs[0]
    device = torch.device("cuda:{}".format(lightest.id) if torch.cuda.is_available() and lightest.load<0.7 and lightest.memoryUtil<0.7 else "cpu")
    stderr.write("\nGPU {} used\n".format(lightest.id))
#device = "cpu"
else:
    device = "cpu"
use_dense = False
