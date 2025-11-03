import torch
import os
import numpy as np
import glob
import pickle
from plyfile import PlyData
from utils.format_converter import SH2RGB
from data_preprocess import Common_Param


def load_ply(common_param: Common_Param, exp: str, seq: str, frame_id: int = None)->tuple:

    """Load PLY file and extract relevant information.

    Returns:
        tuple: (t_index, variables) where variables is a dictionary containing the standard gaussian attributes. t_index for restore training.
    """
    
    device = common_param.device
    torch.cuda.set_device(device)
    variables = {}
    
    dir = f"./output/{exp}/{seq}/render_params/*"
    if frame_id is None:
        files = glob.glob(dir)
        numbers = [int(os.path.splitext(os.path.basename(f))[0]) for f in files]
        path = files[numbers.index(max(numbers))]
        t_index = max(numbers)+1
    else:
        path = f"./output/{exp}/{seq}/render_params/{frame_id}.ply"
        t_index = frame_id+1    
    
    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    variables['prev_pts'] = xyz
    
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
    variables['prev_opa'] = opacities.reshape((xyz.shape[0], 1))
    
    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
    variables['prev_col'] = SH2RGB(features_dc).reshape((xyz.shape[0], 3))
    
    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))

    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, 0))

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])
    variables['prev_scl'] = scales
    
    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
    variables['prev_rot'] = rots        
    
    variables = {k: torch.tensor(v).float().cuda() for k, v in variables.items()}
    return t_index, variables


def load_gs_cate(common_param: Common_Param, exp: str, seq: str, frame_id: int = None)->tuple:
    """Load gs_cate npz file and extract relevant information.
    Returns:
        tuple: (t_index, variables) where variables is a dictionary containing gaussian category. t_index for restore training.
    """
    
    torch.cuda.set_device(common_param.device)
    dir = f"./output/{exp}/{seq}/gs_cate/*"
    if frame_id is None:
        files = glob.glob(dir)
        numbers = [int(os.path.splitext(os.path.basename(f))[0]) for f in files]
        path = files[numbers.index(max(numbers))]
        t_index = max(numbers)+1
    else:
        path = f"./output/{exp}/{seq}/gs_cate/{frame_id}.npz"
        t_index = frame_id+1    
    res_array = np.load(path)
    variables = {k: torch.tensor(v).float().cuda() for k, v in res_array.items()}
    return t_index, variables


def load_ctrl_idx(exp: str, seq: str, t: int)->torch.Tensor|None:
    """
    Load gaussian-control pts connection from pickle file.
    
    Returns:
        torch.Tensor|None: Control indices for the given timestep, or None if not found.
    """
    
    path = f"./output/{exp}/{seq}/ctrl_idx.pkl"
    if os.path.exists(path):
        with open(path, 'rb') as f:
            ctrl_idx = pickle.load(f)
            return ctrl_idx[str(t-1)]
    else:
        return None


def load_res(common_param:Common_Param, exp: str, seq: str, frame_id: int = None, only_ctrl: bool = False)->tuple:

    """Load the residual of gaussian attributes from npz file.

    Returns:
        tuple: (t_index, variables) where variables is a dictionary containing the residual of gaussian attributes. t_index for restore training.
    """
    
    torch.cuda.set_device(common_param.device)
    dir = f"./output/{exp}/{seq}/res_params/*"
    if frame_id is None:
        files = glob.glob(dir)
        numbers = [int(os.path.splitext(os.path.basename(f))[0]) for f in files]
        path = files[numbers.index(max(numbers))]
        t_index = max(numbers)+1
    else:
        path = f"./output/{exp}/{seq}/res_params/{frame_id}.npz"
        t_index = frame_id+1    
    res_array = np.load(path)
    if only_ctrl:
        variables = {k: torch.tensor(v).float().cuda() for k, v in res_array.items() if k in ['ctrl_t1', 'ctrl_r2']}
    else:
        variables = {k: torch.tensor(v).float().cuda() for k, v in res_array.items()}
    return t_index, variables


def get_flow(dataset_dir: str, seq: str, radius: float)->dict:

    """Load preprocessedoptical flow data from a pickle file.

    Returns:
        dict: A dictionary containing H3D control point in flow maps.
    """
    
    # path = f"{dataset_dir}/{seq}/flows/result_dict_model_DIS_radius_{radius}.pkl"
    path = f"{dataset_dir}/{seq}/flows/result_dict_model_DIS_grid-size32_small-range-True-0_avg-motion-True_random-shift-True_big-mask-abl2-unb.pkl"
    # path = f"./flows/result_dict_model_spy_frameNum_300_viewNum_21_clusterNum_0_grid-size32_jump-step2_small-range-True-16_avg-motion-True.pkl"
    with open(path, 'rb') as f:
        flowsss = pickle.load(f)
    return flowsss