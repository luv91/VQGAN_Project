import torch
import torch.nn as nn
import math

class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, max_len=2048, dropout_prob=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_prob)
        
        pe = torch.zeros(max_len, max_len, d_model)
        position_i = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        position_j = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2.) * -(math.log(10000.0) / d_model))

        pe[:, :, 0::2] = torch.sin(position_i * div_term)
        pe[:, :, 1::2] = torch.cos(position_j * div_term)

        pe = pe.permute(2, 0, 1).unsqueeze(0)  # make shape [1, C, H, W]

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :, :x.size(2), :x.size(3)]
        return self.dropout(x)


# Test it
pos_enc = PositionalEncoding2D(d_model=256)

# Input: (batch_size=32, channels=256, height=16, width=16)
input_tensor = torch.randn(32, 256, 16, 16)

# Pass the input through the positional encoding layer
output_tensor = pos_enc(input_tensor)

# Check the output shape (should be same as input)
print(output_tensor.shape)  # Output: torch.Size([32, 256, 16, 16])