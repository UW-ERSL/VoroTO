
import torch
from typing import Union

def impose_same_x_direc_values(values: torch.Tensor, nelx, nely):
  """Results in the same values along the X direction.
  NOTE: Assumes `mesh` is a rectangular grid.
  Args:
    values: Tensor of (num_elems, num_features) that are the values to be
      averaged along the X-axis.
  Returns: Tensor of (num_elems, num_features) that have the same values
    across the X-axis.
  """
  values = values.reshape(nelx, nely, values.shape[1])
  row_means = torch.mean(values, dim=0)#[ None, :, :]
  transformed_arr = row_means.repeat(values.shape[0], 1)
  return transformed_arr



def apply_reflection(x, symMap):
  signs = {}
  if(symMap['YAxis']['isOn']):
    signs['Y'] =  -torch.sign(x[:,0] - symMap['YAxis']['midPt'] + 1e-6)
    xv =( symMap['YAxis']['midPt'] + torch.abs( x[:,0] - symMap['YAxis']['midPt']))
  else:
    signs['Y'] = torch.ones((x.shape[0]))
    xv = x[:, 0]
  if(symMap['XAxis']['isOn']):
    signs['X'] = -torch.sign(x[:,1] - symMap['XAxis']['midPt'] + 1e-6)
    yv = (symMap['XAxis']['midPt'] + torch.abs( x[:,1] - symMap['XAxis']['midPt'])) 
  else:
    signs['X'] = torch.ones((x.shape[0]))
    yv = x[:, 1]
  
  x = torch.transpose(torch.stack((xv,yv)), 0, 1)
  return x, signs


def x_symmetry( nelx: int, nely: int, values: torch.Tensor) -> torch.Tensor:
    """
    Imposes symmetry on the given design variable tensor along the specified axis.

    Args:
        nelx (int): Number of elements along the x-axis.
        nely (int): Number of elements along the y-axis.
        values (torch.Tensor): The design variable tensor.
        axis (str, optional): The axis for symmetry ('x' or 'y'). Defaults to 'x'.

    Returns:
        torch.Tensor: The symmetric version of the design variable tensor.
    """
    # Create a copy of dv to avoid inplace operation
    symmetric_values = values.clone()

    # Reshape symmetric_dv for easier indexing
    symmetric_values = symmetric_values.reshape(nelx, nely, values.shape[1])
    # Imposing symmetry along the x-axis for all x coordinates
    for i in range(int(nely / 2)):
      symmetric_values[:, nely - 1 - i, :] = symmetric_values[:, i, :]

    return symmetric_values.view(-1, values.shape[1]) 

def y_symmetry( nelx: int, nely: int, values: torch.Tensor) -> torch.Tensor:
    """
    Imposes symmetry on the given design variable tensor along the specified axis.

    Args:
        nelx (int): Number of elements along the x-axis.
        nely (int): Number of elements along the y-axis.
        values (torch.Tensor): The design variable tensor.
        axis (str, optional): The axis for symmetry ('x' or 'y'). Defaults to 'x'.

    Returns:
        torch.Tensor: The symmetric version of the design variable tensor.
    """
    # Create a copy of dv to avoid inplace operation
    symmetric_values = values.clone()

    # Reshape symmetric_dv for easier indexing
    symmetric_values = symmetric_values.reshape(nelx, nely, values.shape[1])
  
    # Imposing symmetry along both axes 
    for i in range(int(nelx / 2)):
      symmetric_values[nelx - 1 - i, :, :] = symmetric_values[i, :, :]

    return symmetric_values.view(-1, values.shape[1]) 