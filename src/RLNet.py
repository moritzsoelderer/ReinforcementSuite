import torch.nn as nn
import torch.nn.functional as F


class RLNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(RLNet, self).__init__()
        self.layers = [nn.Linear(input_dim, hidden_dim[0])]
        self.optimizer = None

        for i in range(1, len(hidden_dim)):
            self.layers.append(nn.Linear(hidden_dim[i-1], hidden_dim[i]))

        self.layers = nn.ModuleList(self.layers)
        self.output_layer = nn.Linear(hidden_dim[-1], output_dim)

    def set_optimiizer(self, optimizer):
        self.optimizer = optimizer

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return F.softmax(self.output_layer(x), dim=-1)

    def backpropagate(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()