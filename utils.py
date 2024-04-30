import dataclasses
import numpy as np
import torch
from typing import Union
import torch.nn as nn

_Array = Union[np.ndarray, torch.Tensor]

def to_np(x: torch.Tensor)->np.ndarray:
  if isinstance(x, torch.Tensor):
    return x.detach().cpu().numpy()
  else:
    return x


def to_torch(x: np.ndarray)->torch.Tensor:
  if isinstance(x, np.ndarray):
    return torch.tensor(x).float()
  else:
    return x


@dataclasses.dataclass
class Extent:
  min: float
  max: float

  @property
  def range(self)->float:
    return self.max - self.min

  @property
  def center(self)->float:
    return 0.5*(self.min + self.max)
  

def normalize(x: torch.Tensor, extent: Extent)->torch.Tensor:
  return (x - extent.min)/extent.range

def unnormalize(x: torch.Tensor, extent: Extent)->torch.Tensor:
  return x*extent.range + extent.min

def normalize_z_scale(x: torch.Tensor, mean: torch.Tensor, stddev: torch.Tensor
                      )->torch.Tensor:
  """
                z = (x - mean)/stddev
  Args:
    x: Array of (N, M) that contains the input tensor to be scaled
    mean: Array of (M,) that contains the mean of each feature
    std: Array of (M,) that contains the stddev of each feature
  Returns: Array of (N, M) that has the input scaled 
  """
  return (x - mean[None, :])/stddev[None, :]

def unnormalize_z_scale(z: torch.Tensor, mean: torch.Tensor, stddev: torch.Tensor
                      )->torch.Tensor:
  """
                x = z*stddev + mean
  Args:
    z: Array of (N, M) that contains the Z scaled input tensor
    mean: Array of (M,) that contains the mean of each feature
    std: Array of (M,) that contains the stddev of each feature
  Returns: Array of (N, M) that has the input unscaled 
  """
  return mean[None, :] +  z*stddev[None, :]