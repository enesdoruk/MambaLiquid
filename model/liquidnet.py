import torch
import torch.nn as nn


class LiquidTimeStep(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LiquidTimeStep, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_in = nn.Linear(input_size, hidden_size)
        self.W_h = nn.Linear(hidden_size, hidden_size)
        self.tau = nn.Parameter(torch.ones(hidden_size))
    
    def forward(self, x, h):
        dx = torch.tanh(self.W_in(x) + self.W_h(h))
        h_new = h + (dx - h) / self.tau
        return h_new
    

class LiquidNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LiquidNet, self).__init__()
        self.hidden_size = hidden_size
        self.liquid_step = LiquidTimeStep(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        for t in range(seq_len):
            h = self.liquid_step(x[:, t, :], h)
        output = self.output_layer(h)
        return output