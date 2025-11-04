import os
import numpy as np
import pickle
import torch

from plyfile import PlyData, PlyElement
from utils.format_converter import RGB2SH


def construct_list_of_attributes():
    """Gaussian standard format.
    """
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels except the 3 DC
    for i in range(3):
        l.append('f_dc_{}'.format(i))
    for i in range(0):
        l.append('f_rest_{}'.format(i))
    l.append('opacity')
    for i in range(3):
        l.append('scale_{}'.format(i))
    for i in range(4):
        l.append('rot_{}'.format(i))
    return l


def save_render_ply(render:dict, exp: str, seq: str, t: int, specific_nm: str = None):
    
    """save gaussians in standard format.
    """
    
    os.makedirs(f"./output/{exp}/{seq}/render_params", exist_ok=True)
    if specific_nm is None:
        path = f"./output/{exp}/{seq}/render_params/{t}.ply"
    else:
        path = f"./output/{exp}/{seq}/render_params/{specific_nm}_{t}.ply"
    
    xyz = render['prev_pts']
    normals = np.zeros_like(xyz)
    colors = render['prev_col']
    f_dc = RGB2SH(colors).reshape((xyz.shape[0], -1))
    f_rest = np.zeros((xyz.shape[0], 3, 0)).reshape((xyz.shape[0], -1))
    opacities = render['prev_opa']
    scale = render['prev_scl']
    rotation = render['prev_rot']

    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes()]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)


def save_ctrl_idx(idx:torch.Tensor, exp: str, seq: str, t: int):
    """Save gaussian-control pts connection index to a pickle file.
    """
    path = f"./output/{exp}/{seq}/ctrl_idx.pkl"
    if os.path.exists(path):
        with open(path, 'rb') as f:
            ctrl_idx = pickle.load(f)
    else:
        ctrl_idx = {}
    ctrl_idx[str(t-1)] = idx
    with open(path, 'wb') as f:
        pickle.dump(ctrl_idx, f)



def save_gs_cate(render:dict, exp: str, seq: str, t: int):
    """Save gaussian category information to a npz file.
    """
    
    os.makedirs(f"./output/{exp}/{seq}/gs_cate", exist_ok=True)
    np.savez(f"./output/{exp}/{seq}/gs_cate/{t}", gs_cate=render['gs_cate'])


def save_res_params(output_params:dict, exp: str, seq: str, t: int):
    """Save residual of gaussian attributes to a npz file.
    """

    os.makedirs(f"./output/{exp}/{seq}/res_params", exist_ok=True)
    np.savez(f"./output/{exp}/{seq}/res_params/{t}", **output_params)

