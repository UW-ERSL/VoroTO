import dataclasses
from typing import Tuple
import numpy as np
import random
import utils

_Ext = utils.Extent


@dataclasses.dataclass
class BoundingBox:
  x: _Ext
  y: _Ext

  @property
  def lx(self)->float:
    return self.x.max - self.x.min

  @property
  def ly(self)->float:
    return self.y.max - self.y.min


class Mesher:
  def __init__(self, nelx:int, nely:int, bounding_box: BoundingBox):
    self.num_dim = 2
    self.nelx, self.nely = nelx, nely
    self.num_elems = nelx*nely
    self.bounding_box = bounding_box

    dx, dy = self.bounding_box.lx/nelx, self.bounding_box.ly/nely
    self.elem_size = np.array([dx, dy])
    self.elem_area = dx*dy
    self.domain_volume = self.elem_area*self.num_elems
    self.num_nodes = (nelx+1)*(nely+1)

    [x_grid, y_grid] = np.meshgrid(
      np.linspace(bounding_box.x.min + dx/2., bounding_box.x.max - dx/2., nelx),
      np.linspace(bounding_box.y.min + dy/2., bounding_box.y.max - dy/2., nely))
    self.elem_centers = np.stack((x_grid, y_grid)).T.reshape(-1, self.num_dim)

    [x_grid, y_grid] = np.meshgrid(
               np.linspace(self.bounding_box.x.min,
                           self.bounding_box.x.max,
                           nelx+1),
               np.linspace(self.bounding_box.y.min,
                           self.bounding_box.y.max,
                           nely+1),)
    self.node_xyz = np.stack((x_grid, y_grid)).T.reshape(-1, self.num_dim)


class BilinearStructMesher(Mesher):
  def __init__(self, nelx:int, nely:int, bounding_box: BoundingBox):
    super().__init__(nelx, nely, bounding_box)
    self.num_nodes = (self.nelx + 1)*(self.nely + 1)
    self.nodes_per_elem = 4
    self.dofs_per_node = 2
    self.dofs_per_elem = self.dofs_per_node*self.nodes_per_elem
    self.num_dofs = self.dofs_per_node*self.num_nodes
    self.elem_node, self.edofMat, self.iK, self.jK = \
        self.compute_connectivity_info()

  def compute_connectivity_info(self)-> Tuple[np.ndarray, np.ndarray,
                                              np.ndarray, np.ndarray]:

    elem_node = np.zeros((self.nodes_per_elem, self.num_elems))
    for elx in range(self.nelx):
      for ely in range(self.nely):
        el = ely+elx*self.nely
        n1=(self.nely+1)*elx+ely
        n2=(self.nely+1)*(elx+1)+ely
        elem_node[:,el] = np.array([n1+1, n2+1, n2, n1])
    elem_node = elem_node.astype(int)

    edofMat = np.zeros((self.num_elems,
        self.nodes_per_elem*self.num_dim), dtype=int)

    for elem in range(self.num_elems):
      enodes = elem_node[:, elem]
      edofs = np.stack((2*enodes, 2*enodes+1), axis=1).reshape(
                                            (1, self.dofs_per_elem))
      edofMat[elem, :] = edofs

    matrx_size = self.num_elems*self.dofs_per_elem**2
    iK = tuple(np.kron(edofMat, np.ones((self.dofs_per_elem, 1),dtype=int)).T
                .reshape(matrx_size, order ='F'))
    jK = tuple(np.kron(edofMat, np.ones((1, self.dofs_per_elem),dtype=int)).T
                .reshape(matrx_size, order ='F'))
    return elem_node, edofMat, iK, jK
   

def compute_point_indices_in_box(xyz: np.ndarray, bbox: BoundingBox):
  """
  Filters the coordinates in `xy` that are within the bounding box.

  Args:
      xy: An array of shape (num_elems, 2) containing the coordinates.
      bbox: Defines the coordinates of the bounding box.

  Returns: A Boolean array of shape (num_elems,) with True values for indices
    of `xyz` whose coordinates are within the bounding box.
  """

  x_in_box = np.logical_and(xyz[:, 0] >= bbox.x.min, xyz[:, 0] <= bbox.x.max)
  y_in_box = np.logical_and(xyz[:, 1] >= bbox.y.min, xyz[:, 1] <= bbox.y.max)
  filtered_indices = np.logical_and(x_in_box, y_in_box)
  return filtered_indices


def get_neighbors(mesh: Mesher)->np.ndarray:
  """
  This function generates an array of size (num_elems, 9) with the index of the
  global neighboring elements (and itself) for a rectangular domain.

  The neighboring elements are in the order below:

              ------------
              | 2 | 5 | 8 |
              ------------
              | 1 | 4 | 7 |
              ------------
              | 0 | 3 | 6 |
              ------------

  NOTE: The boundary edges are wrapped around one another.
  Args:
    mesh: Rectangular domain containing (nelx, nely) elements along X and Y
      respectively.

  Returns:
    neighbors: An array of size (num_elems, 9) with the index of the
      neighboring elements.
  """
  neighbors = np.zeros((mesh.num_elems, 9), dtype=np.int32)

  # Define offsets for neighboring elements
  offsets = [(-1, -1), (-1, 0), (-1, 1),
             (0, -1), (0, 0), (0, 1),
             (1, -1), (1, 0), (1, 1)]



  # Loop through all pixels
  for i in range(mesh.nelx):
    for j in range(mesh.nely):
      idx = i * mesh.nely + j

      # Calculate neighboring indices within the image
      for k, (di, dj) in enumerate(offsets):
        ni = i + di
        nj = j + dj

        # Check if neighbor is within the image
        if 0 <= ni < mesh.nelx and 0 <= nj < mesh.nely:
          neighbors[idx, k] = ni * mesh.nely + nj
        else:
          # Handle edge cases by wrapping around the image
          ni = (ni + mesh.nelx) % mesh.nelx
          nj = (nj + mesh.nely) % mesh.nely
          neighbors[idx, k] = ni *mesh.nely + nj

  return neighbors





def compute_range_from_min_seperation(min_seperation: float,
                                      bounding_box: BoundingBox,
                                      num_pts_x: int,
                                      num_pts_y: int):
  """
  Args:
  Returns:
  """
  dx, dy = bounding_box.lx/num_pts_x, bounding_box.ly/num_pts_y
  range_x, range_y = 0.5*(dx - min_seperation), 0.5*(dy - min_seperation)
  return range_x, range_y


def get_cell_coordinates_from_perturbations(perturb_x: utils._Array,
                                            perturb_y: utils._Array,
                                            ground_x: utils._Array,
                                            ground_y: utils._Array):
  """
  Args:
  Returns:
  """
  cx = ground_x + perturb_x
  cy = ground_y + perturb_y
  return cx, cy


def generate_randomly_perturbed_grid_points(num_pts_x: int,
                                            num_pts_y: int,
                                            min_seperation: float,
                                            bounding_box: BoundingBox,
                                            rng):
  """
  Args:
  Returns:
  """
  num_pts = num_pts_x*num_pts_y
  dx, dy = bounding_box.lx/num_pts_x, bounding_box.ly/num_pts_y

  [x_grid, y_grid] = np.meshgrid(
      np.linspace(bounding_box.x.min + dx/2., bounding_box.x.max - dx/2., num_pts_x),
      np.linspace(bounding_box.y.min + dy/2., bounding_box.y.max - dy/2., num_pts_y))

  range_x, range_y = compute_range_from_min_seperation(min_seperation,
                                                       bounding_box,
                                                       num_pts_x,
                                                       num_pts_y)

  perturb_x = rng.uniform(-range_x, range_x, (num_pts,))
  perturb_y = rng.uniform(-range_y, range_y, (num_pts,))

  cx, cy = get_cell_coordinates_from_perturbations(perturb_x,
                                                   perturb_y,
                                                   x_grid.reshape((-1)),
                                                   y_grid.reshape((-1)))
  return cx, cy


def compute_radial_filter(mesh: Mesher, radius: float)->np.ndarray:
  """Compute a circular filter of radius `radius`.

  Used to smoothen the elemental values
  Args:
  Returns: Array of (num_elems, num_elems) that are the weights that smoothen
    a field.
  """
  nwax = np.newaxis
  distances = mesh.elem_centers[:, nwax, :] - mesh.elem_centers[nwax, :, :]
  dist_mtrx = np.sqrt(np.sum(distances**2, axis=2)) #{num_elem, num_elem}
  filter = np.clip(1. - dist_mtrx/radius, a_max=None, a_min=0.)
  filter = filter/np.sum(filter, axis=1)
  return filter