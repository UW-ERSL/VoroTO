import dataclasses
from typing import Optional
import torch
import torch.nn as nn

@dataclasses.dataclass
class NNSettings:
  input_dim: int
  num_neurons_per_layer: int
  num_layers: int
  output_dim: int
  use_batch_norm: Optional[bool] = True


class VoronoiNet(nn.Module):
  def __init__(self, nn_settings: NNSettings, seed: int = 27):
    super(VoronoiNet, self).__init__()

    self.nn_settings = nn_settings
    self.layers = nn.ModuleList()
    current_dim = nn_settings.input_dim

    torch.manual_seed(seed)
    for _ in range(nn_settings.num_layers):
      l = nn.Linear(current_dim, nn_settings.num_neurons_per_layer)
      nn.init.xavier_normal_(l.weight)
      nn.init.zeros_(l.bias)
      self.layers.append(l)
      current_dim = nn_settings.num_neurons_per_layer
    self.layers.append(nn.Linear(current_dim, nn_settings.output_dim))

    if self.nn_settings.use_batch_norm:
      self.bn_layers = nn.ModuleList()
      for _ in range(nn_settings.num_layers):
        self.bn_layers.append(nn.BatchNorm1d(nn_settings.num_neurons_per_layer))

  def forward(self, x):
    m = nn.ReLU()
 
    for ctr, layer in enumerate(self.layers[:-1]):
      x = layer(x)
      if self.nn_settings.use_batch_norm:
        x = self.bn_layers[ctr](x)
      x = m(x)

    return self.layers[-1](x)