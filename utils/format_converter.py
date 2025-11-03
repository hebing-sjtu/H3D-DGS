import torch

C0 = 0.28209479177387814

def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def SH2RGB(sh):
    return sh * C0 + 0.5

def variables2rendervar(variables):
    """from training/saving format to rendering format
    **variables** refers to gaussian attributes at previous timestep.
    **variables** contains some extra info for training process, e.g., gs_cate. ctrl points.
    """
    rendervar = {
        'means3D': variables['prev_pts'],
        'colors_precomp': variables['prev_col'],
        'rotations': torch.nn.functional.normalize(variables['prev_rot']),
        'opacities': torch.sigmoid(variables['prev_opa']),
        'scales': torch.exp(variables['prev_scl']),
        'means2D': torch.zeros_like(variables['prev_pts'], requires_grad=True, device="cuda") + 0
    }
    return rendervar

def params2rendervar(params):
    """Convert parameters from training/saving format to rendering format.
    **params** refers to gaussian attributes at current timestep.
    """
    rendervar = {
        'means3D': params['means3D'],
        'colors_precomp': params['rgb_colors'],
        'rotations': torch.nn.functional.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(params['log_scales']),
        'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device="cuda") + 0
    }
    return rendervar

def params2vars(params):
    """Initialize Gaussian attributes in a variable dictionary from current parameters for the next timestep training.
    """
    variables = {
        'prev_pts': params['means3D'],
        'prev_col': params['rgb_colors'],
        'prev_rot': params['unnorm_rotations'],
        'prev_opa': params['logit_opacities'],
        'prev_scl': params['log_scales']
    }
    return variables

def params2cpu(params):
    """move gaussian residual information to cpu for saving
    """
    res = {k: v.detach().cpu().contiguous().numpy() for k, v in params.items() if
            k in ['means3D', 'rgb_colors', 'unnorm_rotations', 'logit_opacities', 'log_scales', 'ctrl_t1', 'ctrl_r2']}
    return res

def vars2cpu(variables):
    """move gaussian attributes to cpu for saving
    """
    render = {k: v.detach().cpu().contiguous().numpy() for k, v in variables.items() if
            k in ['prev_pts', 'prev_col', 'prev_rot', 'prev_opa', 'prev_scl', 'gs_cate']}
    return render