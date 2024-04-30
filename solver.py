import numpy as np
import torch
from torch_sparse_solve import solve
import mesher
import material
import bcs
import fe_templates

class Solver:

  def __init__(self,
               mesh: mesher.BilinearStructMesher,
               mat: material.MaterialConstants,
               bc: bcs.BC,
               fixture_penalty: float = 1e8):
    self.mesh, self.mat, self.bc = mesh, mat, bc

    self.stiff_templ = fe_templates.compute_struct_fe_templates(self.mesh.elem_size)

    indices = np.stack((np.zeros_like(self.bc.fixed_dofs), self.bc.fixed_dofs, self.bc.fixed_dofs), axis=-1).T

    values = fixture_penalty*torch.ones((self.bc.fixed_dofs.shape[0])).double()
    fixture_shape = (1, self.mesh.num_dofs, self.mesh.num_dofs)
    self.fixed_bc_penalty_matrix = (torch.sparse_coo_tensor(
                                    indices, values, fixture_shape))
    bK = tuple(np.zeros((len(self.mesh.iK))).astype(int)) #batch values
    self.node_idx = [bK, self.mesh.iK, self.mesh.jK]


  def compute_elem_stiffness_matrix(self,
                                    C00: torch.Tensor,
                                    C11: torch.Tensor,
                                    C22: torch.Tensor,
                                    C01: torch.Tensor,
                                    C02: torch.Tensor,
                                    C12: torch.Tensor)->torch.Tensor:
    """
    Args:
      Cij: Array of size (num_elems,) that contain the constitutive property of each
      of the elements for FEA. 
    Returns: Array of size (8, 8, num_elems) which is the structual
      stiffness matrix of each of the bilinear quad elements. Each element has
      8 dofs corresponding to the x and y displacements of the 4 noded quad
      element.
    """ 
    # e - element, i - elem_nodes j - elem_nodes
    elem_stiff =  ( torch.einsum('e, ij -> eij', C00, self.stiff_templ['00']) + 
                    torch.einsum('e, ij -> eij', C11, self.stiff_templ['11']) +
                    torch.einsum('e, ij -> eij', C22, self.stiff_templ['22']) +
                    torch.einsum('e, ij -> eij', C01, self.stiff_templ['01']) +
                    torch.einsum('e, ij -> eij', C02, self.stiff_templ['02']) +
                    torch.einsum('e, ij -> eij', C12, self.stiff_templ['12']) 
                    )
    return elem_stiff


  def assemble_stiffness_matrix(self, elem_stiff_mtrx: torch.Tensor):
    """
    Args:
      elem_stiff_mtrx: Array of size (num_elems, 24, 24) which is the structual
        stiffness matrix of each of the trilinear quad elements. Each element has
        24 dofs corresponding to the u, v, w displacements of the 8 noded quad
        element.
    Returns: Array of size (num_dofs, num_dofs) which is the assembled global
      stiffness matrix.
    """
    return torch.sparse_coo_tensor(self.node_idx,
                                    elem_stiff_mtrx.flatten(),
            (1, self.mesh.num_dofs, self.mesh.num_dofs))


  def solve(self, assm_stiff_mtrx: torch.Tensor):
    """Solve the system of Finite element equations.
    Args:
      glob_stiff_mtrx: Array of size (num_dofs, num_dofs) which is the assembled
        global stiffness matrix.
    Returns: Array of size (num_dofs,) which is the displacement of the nodes.
    """
    force = self.bc.force.unsqueeze(0) #{1, #dofs, 1}
    net_stiff_mtrx = (assm_stiff_mtrx + self.fixed_bc_penalty_matrix).coalesce()
    u = solve(net_stiff_mtrx, force).flatten()
    return u


  def compute_compliance(self, u:torch.Tensor)->torch.Tensor:
    """Objective measure for structural performance.
    Args:
      u: Array of size (num_dofs,) which is the displacement of the nodes
        of the mesh.
    Returns: Structural compliance, which is a measure of performance. Lower
      compliance means stiffer and stronger design.
    """
    return torch.dot(u.view(-1), self.bc.force.view(-1))


  def loss_function(self, C_components: tuple[torch.Tensor])->torch.Tensor:
    """Wrapper function that takes in density field and returns compliance.
    Args:
      C_components: Tuple with each entry of size (num_elems,) that contain the 
      C00, C11, C22, C01, C02, C12 components of the elements.
    Returns: Structural compliance, which is a measure of performance. Lower
      compliance means stiffer and stronger design.
    """
    elem_stiffness_mtrx = self.compute_elem_stiffness_matrix(*C_components)
    glob_stiff_mtrx = self.assemble_stiffness_matrix(elem_stiffness_mtrx)
    u = self.solve(glob_stiff_mtrx)
    return self.compute_compliance(u), u