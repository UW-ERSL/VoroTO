import dataclasses
from typing import Tuple
import torch
import numpy as np
from scipy.special import softmax
import mesher

VoronoiExtent = mesher.BoundingBox

@dataclasses.dataclass
class VoronoiCoordinates:
  """
  Attributes:
    cell_x: Array of size (num_mstr, num_cells_per_mstr) that have the 
    voronoi coordinates in local frame of reference
  """
  cell_x: torch.Tensor
  cell_y: torch.Tensor

  @property
  def num_mstrs(self)->int:
    return self.cell_x.shape[0]

  @property
  def num_cells_per_mstr(self)->int:
    return self.cell_x.shape[1]

  @property
  def num_dimensions(self)->int:
    return 2

  @classmethod
  def from_normalized_stacked_array(self, stkd_array: torch.Tensor,
                                    voro_bbox: VoronoiExtent):
    """
    Args:
      stkd_array: Array of (num_mstr, num_cells_per_mstr, 2)
      voro_bbox: Bounding box that defines the voronoi coordinates in local
        frame of reference.
    """
    cell_x_norm, cell_y_norm = stkd_array[:, :, 0], stkd_array[:, :, 1]
    cell_x = voro_bbox.x.min + cell_x_norm*(voro_bbox.x.range)
    cell_y = voro_bbox.y.min + cell_y_norm*(voro_bbox.y.range)

    return VoronoiCoordinates(cell_x, cell_y)
  

def compute_voronoi_cell_coordns_to_global_frame(cell_x_local: torch.Tensor,
                                                cell_y_local: torch.Tensor,
                                                global_mesh: mesher.Mesher
                                          )->Tuple[torch.Tensor, torch.Tensor]:
  """
  Args:
    cell_x_local: Array of (num_elems, cells_per_elem)
    global_mesh: Contains a mesh with (num_elems,) elements
  Returns: cx_glob is an array of (num_elems, cells_per_elem)
  """
  cx_glob = cell_x_local + global_mesh.elem_centers[:, 0][:, None]
  cy_glob = cell_y_local + global_mesh.elem_centers[:, 1][:, None]
  return cx_glob, cy_glob


def compute_voronoi_cell_coordns_to_local_frame(cell_x_glob: np.ndarray,
                                                cell_y_glob: np.ndarray,
                                                global_mesh: mesher.Mesher
                                          )->Tuple[np.ndarray, np.ndarray]:
  """
  Args:
    cell_x_local: Array of (num_elems, cells_per_elem)
    global_mesh: Contains a mesh with (num_elems,) elements
  Returns: A tuple of (cx_local, cy_local) is an array of (num_elems, cells_per_elem)
  """
  cx_local = cell_x_glob - global_mesh.elem_centers[:, 0][:, None]
  cy_local = cell_y_glob - global_mesh.elem_centers[:, 1][:, None]
  return cx_local, cy_local


def compute_voronoi_density_field(mesh:mesher.Mesher,
                                  voro_cell_x: np.ndarray,
                                  voro_cell_y: np.ndarray,
                                  beta: float,
                                  orientation_rad: float =0.,
                                  deg_aniso: float = 1.,
                                  softness_param: float = -1.e2,
                                  round_density: bool = True):

  nwax = np.newaxis
  delx = voro_cell_x[:, None] - mesh.elem_centers[:,0][None, :]
  dely = voro_cell_y[:, None] - mesh.elem_centers[:,1][None, :]
  c, s = np.cos(orientation_rad), np.sin(orientation_rad)

  zeta = delx*c - dely*s
  eta = delx*s + dely*c
  dist_voronoi_mesh = np.sqrt(deg_aniso*(zeta)**2 + (eta)**2/deg_aniso) #{ce}
  
  min_vals = np.min(dist_voronoi_mesh, axis=1)
  max_vals = np.max(dist_voronoi_mesh, axis=1)

  # Normalize each element by subtracting the minimum and dividing by the range
  normalized_dist = (dist_voronoi_mesh - min_vals[:, nwax]) / (
                    max_vals[:, nwax] - min_vals[:, nwax])

  nearest_point_idx = softmax(softness_param*normalized_dist, axis=0)
  thkns = 10**beta
  density = 1. - np.sum(nearest_point_idx**thkns, axis=0)
  if round_density:
    return np.round(density)
  return density

def compute_voronoi_density_field_aniso(mesh: mesher.Mesher,
                                      voro_cell_x: np.ndarray,
                                      voro_cell_y: np.ndarray,
                                      beta: np.ndarray,
                                      orientation_rad: np.ndarray,
                                      deg_aniso: np.ndarray,
                                      softness_param: float = -1.e2,
                                      round_density: bool = True):

  # c -> cells, p -> points
  nwax = np.newaxis
  delx = voro_cell_x[:, None] - mesh.elem_centers[:,0][None, :] #{cp}
  dely = voro_cell_y[:, None] - mesh.elem_centers[:,1][None, :] #{cp}
  c, s = np.cos(orientation_rad), np.sin(orientation_rad) #{c}

  zeta = np.einsum('cp,c->cp', delx, c) - np.einsum('cp,c->cp',dely, s)
  eta = np.einsum('cp,c->cp', delx, s) + np.einsum('cp,c->cp', dely, c)
  dist_voronoi_mesh = np.sqrt(
                np.einsum('c,cp->cp', deg_aniso, (zeta)**2) +
                np.einsum('cp,c->cp', (eta)**2, 1./deg_aniso)
                ) #{cp}

  min_vals = np.min(dist_voronoi_mesh, axis=1)
  max_vals = np.max(dist_voronoi_mesh, axis=1)

  # Normalize each element by subtracting the minimum and dividing by the range
  normalized_dist = (dist_voronoi_mesh - min_vals[:, nwax]) / (
                    max_vals[:, nwax] - min_vals[:, nwax])
  thkns = 10**beta
  nearest_point_idx = softmax((softness_param*normalized_dist), axis=0)
  density = 1. - np.sum(nearest_point_idx**thkns[:, nwax], axis=0)
  if round_density:
    return np.round(density)
  return density