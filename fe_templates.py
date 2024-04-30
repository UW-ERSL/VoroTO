import numpy as np
import utils

_to_tch = utils.to_torch

def compute_struct_fe_templates(elem_size: np.ndarray):
  dx, dy = elem_size[0], elem_size[1]

  def Knn_00(dx,dy):
    t2 = 1.0/dx
    t3 = dy*t2*(1.0/3.0)
    t4 = dy*t2*(1.0/6.0)
    Knn_00 = np.reshape(np.array([t3,0,-t3,0,-t4,0,t4,0,0,0,0,0,0,0,0,0
                                  ,-t3,0,t3,0,t4,0,-t4,0,0,0,0,0,0,0,0,0,
                                  dy*t2*(-1.0/6.0),0,t4,0,t3,0,-t3,0,0,0,
                                  0,0,0,0,0,0,t4,0,-t4,0,-t3,0,t3,0,0,0,0,
                                  0,0,0,0,0]),(8,8))
    return Knn_00

  def Knn_11(dx,dy):
    t2 = 1.0/dy
    t3 = dx*t2*(1.0/6.0)
    t4 = dx*t2*(1.0/3.0)
    Knn_11 = np.reshape(np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,t4,0.0,
                        t3,0.0,-t3,0.0,-t4,0.0,0.0,0.0,0.0,0.0,0.0,
                        0.0,0.0,0.0,t3,0.0,t4,0.0,-t4,0.0,-t3,0.0,0.0,
                        0.0,0.0,0.0,0.0,0.0,0.0,0.0,-t3,0.0,-t4,0.0,t4,
                        0.0,t3,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-t4,
                        0.0,-t3,0.0,t3,0.0,t4]),(8,8))
    return Knn_11

  def Knn_22(dx,dy):
    t2 = 1.0/dy
    t3 = dx*t2*(1.0/6.0)
    t4 = dx*t2*(1.0/3.0)
    t5 = 1.0/dx
    t6 = dy*t5*(1.0/3.0)
    t7 = dy*t5*(1.0/6.0)
    Knn_22 = np.reshape(np.array([t4,1.0/4.0,t3,-1.0/4.0,-t3,-1.0/4.0, -t4,
                                  1.0/4.0,1.0/4.0,t6,1.0/4.0,-t6,-1.0/4.0,
                                  -t7,-1.0/4.0,t7,t3,1.0/4.0,t4,-1.0/4.0,-t4,
                                  -1.0/4.0,-t3,1.0/4.0,-1.0/4.0,-t6,-1.0/4.0,
                                  t6,1.0/4.0,t7,1.0/4.0,-t7,-t3,-1.0/4.0,-t4,
                                  1.0/4.0,t4,1.0/4.0,t3,-1.0/4.0,-1.0/4.0,
                                  dy*t5*(-1.0/6.0),-1.0/4.0,t7,1.0/4.0,t6,
                                  1.0/4.0,-t6,-t4,-1.0/4.0,-t3,1.0/4.0,t3,
                                  1.0/4.0,t4,-1.0/4.0,1.0/4.0,t7,1.0/4.0,-t7,
                                  -1.0/4.0,-t6,-1.0/4.0,t6]),(8,8))
    return Knn_22

  def Knn_01(dx,dy):
    Knn_01 = np.reshape(np.array([0.0,1.0/4.0,0.0,1.0/4.0,0.0,-1.0/4.0,0.0,
                        -1.0/4.0,1.0/4.0,0.0,-1.0/4.0,0.0,-1.0/4.0,
                        0.0,1.0/4.0,0.0,0.0,-1.0/4.0,0.0,-1.0/4.0,0.0,
                        1.0/4.0,0.0,1.0/4.0,1.0/4.0,0.0,-1.0/4.0,0.0,
                        -1.0/4.0,0.0,1.0/4.0,0.0,0.0,-1.0/4.0,0.0,-1.0/4.0,
                        0.0,1.0/4.0,0.0,1.0/4.0,-1.0/4.0,0.0,1.0/4.0,0.0,
                        1.0/4.0,0.0,-1.0/4.0,0.0,0.0,1.0/4.0,0.0,1.0/4.0,0.0,
                        -1.0/4.0,0.0,-1.0/4.0,-1.0/4.0,0.0,1.0/4.0,0.0,1.0/4.0,
                        0.0,-1.0/4.0,0.0]),(8,8))
    return Knn_01

  def Knn_02(dx,dy):
    t2 = 1.0/dx
    t3 = dy*t2*(1.0/3.0)
    t4 = dy*t2*(1.0/6.0)
    Knn_02 = np.reshape(np.array([1.0/2.0,t3,0.0,-t3,-1.0/2.0,-t4,0.0,t4,t3,0.0,
                        -t3,0.0,-t4,0.0,t4,0.0,0.0,-t3,-1.0/2.0,t3,0.0,t4,
                        1.0/2.0,-t4,-t3,0.0,t3,0.0,t4,0.0,-t4,0.0,-1.0/2.0,
                        -t4,0.0,t4,1.0/2.0,t3,0.0,-t3,dy*t2*(-1.0/6.0),0.0,t4,
                        0.0,t3,0.0,-t3,0.0,0.0,t4,1.0/2.0,-t4,0.0,-t3,-1.0/2.0,
                        t3,t4,0.0,-t4,0.0,-t3,0.0,t3,0.0]),(8,8))
    return Knn_02

  def Knn_12(dx,dy):
    t2 = 1.0/dy
    t3 = dx*t2*(1.0/6.0)
    t4 = dx*t2*(1.0/3.0)
    Knn_12 = np.reshape(np.array([0.0,t4,0.0,t3,0.0,-t3,0.0,-t4,t4,1.0/2.0,t3,0.0,-t3,
                        -1.0/2.0,-t4,0.0,0.0,t3,0.0,t4,0.0,-t4,0.0,-t3,t3,
                        0.0,t4,-1.0/2.0,-t4,0.0,-t3,1.0/2.0,0.0,-t3,0.0,-t4,
                        0.0,t4,0.0,t3,-t3,-1.0/2.0,-t4,0.0,t4,1.0/2.0,t3,0.0,
                        0.0,-t4,0.0,-t3,0.0,t3,0.0,t4,-t4,0.0,-t3,1.0/2.0,t3,
                        0.0,t4,-1.0/2.0]),(8,8))
    return Knn_12

  stiff_templ = {'00': _to_tch(Knn_00(dx, dy)),
                 '11': _to_tch(Knn_11(dx, dy)),
                 '22': _to_tch(Knn_22(dx, dy)),
                 '01': _to_tch(Knn_01(dx, dy)),
                 '02': _to_tch(Knn_02(dx, dy)),
                 '12': _to_tch(Knn_12(dx, dy)),}
  return stiff_templ