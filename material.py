import dataclasses
from typing import Optional
import numpy as np
import torch


@dataclasses.dataclass
class MaterialConstants:
  """ Linear material constants.
  Attributes:
    youngs_modulus: The young's modulus of the material [Pa].
    poissons_ratio: The poisson's ratio of the material [-].
  """
  youngs_modulus: Optional[float] = None
  poissons_ratio: Optional[float] = None
  mass_density: Optional[float] = None
  thermal_conductivity: Optional[float] = None
  heat_capacity: Optional[float] = None

def get_isotropic_constitutive_matrix(mat: MaterialConstants):
  E, nu = mat.youngs_modulus, mat.poissons_ratio
  return E/(1-nu**2)*np.array([[1., nu, 0],
                          [nu, 1., 0],
                          [0, 0, (1.-nu)/2.]])

def compute_SIMP_material_modulus(density: torch.Tensor,
                                  material: MaterialConstants,
                                  penal: float = 3.,
                                  young_min: float = 1e-3)->torch.Tensor:
  """
    E = rho_min + E0*( density)^penal
  Args:
    density: Array of size (num_elems,) with values in range [0,1]
    penal: SIMP penalization constant, usually assumes a value of 3
    young_min: Small value added to the modulus to prevent matrix singularity
  Returns: Array of size (num_elems,) which contain the penalized modulus
    at each element
  """
  return young_min + material.youngs_modulus*(density**penal)