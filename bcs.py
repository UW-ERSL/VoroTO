import dataclasses
from enum import Enum, auto
import numpy as np
import torch

import mesher
_BilinMesh = mesher.BilinearStructMesher


@dataclasses.dataclass
class BC:
  """
  Attributes:
    force: Array of size (num_dofs, 1) that contain the imposed load on each dof.
    fixed_dofs: Array of size (num_fixed_dofs,) that contain all the dof numbers
      that are fixed.
  """
  force: torch.Tensor
  fixed_dofs: np.ndarray

  @property
  def num_dofs(self):
    return self.force.shape[0]

  @property
  def free_dofs(self):
    return np.setdiff1d(np.arange(self.num_dofs),self.fixed_dofs)
  

class SturctBCs(Enum):
  MID_CANT_BEAM = auto()
  TIP_CANT_BEAM = auto()
  MBB_BEAM = auto()
  BEAM_BONE = auto()
  RIGHT_LOADED = auto()
  TENSILE_BAR = auto()

def get_sample_struct_bc(mesh:_BilinMesh, sample:SturctBCs)->BC:

  force = np.zeros((2*mesh.num_nodes, 1))
  dofs = np.arange(2*mesh.num_nodes)

  if sample == SturctBCs.MID_CANT_BEAM:
    fixed = dofs[0:2*(mesh.nely+1):1]
    force[2*(mesh.nelx+1)*(mesh.nely+1) - 1*(mesh.nely+1), 0] = -1.

  if sample == SturctBCs.TIP_CANT_BEAM:
    fixed = dofs[0:2*(mesh.nely+1):1]
    force[2*(mesh.nelx+1)*(mesh.nely+1) - 2*(mesh.nely+1), 0] = -1.

  if sample == SturctBCs.MBB_BEAM:
    fixed= np.union1d(np.arange(0,2*(mesh.nely+1),2),
                      2*(mesh.nelx+1)*(mesh.nely+1)-2*(mesh.nely+1)+1)
    force[2*(mesh.nely+1)+1 ,0]= -1.
  
  if sample == SturctBCs.BEAM_BONE:
    fixed = dofs[0:(mesh.nely+1):1]
    force[2*(mesh.nelx+1)*(mesh.nely+1) - 1*(mesh.nely+1), 0] = 1.

  if sample == SturctBCs.RIGHT_LOADED:
    fixed = dofs[0:2*(mesh.nely+1):1]
    force_dofs = np.arange(2*(mesh.nelx+1)*(mesh.nely+1)- 2*(mesh.nely+1) +1 ,
                            2*(mesh.nelx+1)*(mesh.nely+1)+1,
                            2)
    force[force_dofs, 0] = -100./(mesh.nely*mesh.elem_size[1])

  if sample == SturctBCs.TENSILE_BAR:
    fixed = dofs[np.union1d(np.arange(0,2*(mesh.nely+1),2), mesh.nely+1)]
  
    midDofX= 2*(mesh.nelx+1)*(mesh.nely+1)- (mesh.nely)
    force_dofs = np.arange(midDofX-int(3*mesh.nely/10),
                           midDofX+int(3*mesh.nely/10), 2)
    force[force_dofs, 0] = 1.
    
  return BC(force= torch.tensor(force), fixed_dofs=fixed)