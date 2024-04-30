import numpy as np
import torch
from torch.utils.data import Dataset

# np.load('support_img_data_file.npy')
class VoronoiDataset(Dataset):
  def __init__(self,
               voronoi_params: np.ndarray,
               homogen_params: np.ndarray,
               transform=None):
    """
    NOTE: The class assumes that the features have been normalized and cleaned.
      No additional data clean up is considered here.
    Attributes:
      voronoi_params: Array of (num_data, num_input_features) that contain
        the voronoi parameters such as cell coordinates, thickness, aniso etc
      homogen_params: Array of (num_data, num_output_features) that contain
        the homogenized parameters such as C matrix, vol frac etc.
    """
    self.voronoi_params = torch.tensor(voronoi_params).float()
    self.homogen_params = torch.tensor(homogen_params).float()
    self.transform = transform

  # TODO: post init verification that input and output contains same no of data
  def __len__(self)->int:
    return self.voronoi_params.shape[0]


  def num_input_features(self)->int:
    return self.voronoi_params.shape[1]


  def num_output_features(self)->int:
    return self.homogen_params.shape[1]


  def __getitem__(self, idx):
    voro_params = self.voronoi_params[idx, ...]
    homo_params = self.homogen_params[idx, ...]

    if self.transform:
      voro_params = self.transform(voro_params)
      homo_params = self.transform(homo_params)

    return voro_params, homo_params