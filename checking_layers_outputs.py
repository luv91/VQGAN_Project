import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder
from codebook import Codebook

m = nn.Conv2d(256, 256, 1)

input_1 = torch.randn(1,256, 16, 16)

output = m(input_1)

print(output.shape)