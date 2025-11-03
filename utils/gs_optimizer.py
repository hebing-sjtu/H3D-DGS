import os
import torch
import numpy as np
from plyfile import PlyData
from utils.data_preprocess import Common_Param, o3d_knn, get_batch
from utils.rotation import build_quaternion_from_euler, quat_mult, rotate_vector
from utils.format_converter import params2rendervar
from utils.CTRL_pts import volume_warping
from utils.loss_function import l1_loss_v1, l1_loss_v2, l1_loss_v3, l1_loss_v4, calc_ssim, calc_psnr
from utils.rotation import build_rotation_from_quaternion
from diff_gaussian_rasterization import GaussianRasterizer as Renderer

def initialize_params(common_param:Common_Param):

    """Initialize parameters for the Gaussian Splatting optimization.
    Params:
        5 gaussian attributes: means3D, rgb_colors, unnorm_rotations, logit_opacities, log_scales
        2 control point attributes: ctrl_t1(1-d translation), ctrl_r2(2-d euler angles)

    Returns:
        Tuple[Dict[str, torch.nn.Parameter], Dict[str, torch.Tensor]]: Initialized parameters and variables.
        variables refers to training state. params refers to optimizable parameters.
    """
    
    torch.cuda.set_device(common_param.device)
    if common_param.is_sport:
        npz_path = os.path.join(common_param.dataset_dir, common_param.seq, "init_pt_cld.npz")
        init_pt_cld = np.load(npz_path)["data"]
        positions = init_pt_cld[:, :3]
        colors = init_pt_cld[:, 3:6]
    else:
        ply_path = os.path.join(common_param.dataset_dir, common_param.seq, "points3D_downsample2.ply")
        plydata = PlyData.read(ply_path)
        vertices = plydata['vertex']
        positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    pts_num = positions.shape[0]

    print("initial point num: ", pts_num)
    sq_dist, _ = o3d_knn(positions, positions, 3, min_contained=False)
    mean3_sq_dist = sq_dist.mean(-1).clip(min=0.0000001)
    params = {
        'means3D': positions,
        'rgb_colors': colors,
        'unnorm_rotations': np.tile([1, 0, 0, 0], (pts_num, 1)),
        'logit_opacities': np.zeros((pts_num, 1)),
        'log_scales': np.tile(np.log(np.sqrt(mean3_sq_dist))[..., None], (1, 3)),
        'ctrl_t1': np.array([[]]),
        'ctrl_r2': np.array([[]]),
    }
    params = {k: torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True)) for k, v in
              params.items()}
    cam_centers = common_param.campos.cpu().numpy()  # Get scene radius
    scene_radius = 1.1 * np.max(np.linalg.norm(cam_centers - np.mean(cam_centers, 0)[None], axis=-1))
    print("scene_radius: ", scene_radius)
    variables = {
        'max_2D_radius': torch.zeros(params['means3D'].shape[0]).cuda().float(),
        'scene_radius': scene_radius,
        'means2D_gradient_accum': torch.zeros(params['means3D'].shape[0]).cuda().float(),
        'denom': torch.zeros(params['means3D'].shape[0]).cuda().float(),
        'gs_cate': torch.zeros(params['means3D'].shape[0],dtype=torch.uint8).cuda(),
        'ctrl_xyz': torch.tensor([[]]),
        'ctrl_t2': torch.tensor([[]]),
        'ctrl_r1': torch.tensor([[]]),
        'ctrl_qw2c': torch.tensor([[]]),
        'ctrl_qc2w': torch.tensor([[]]),
        'ctrl_cate': torch.tensor([[]]),
        'gs_indices': [],
        'ctrl_indices': [],
        'ctrl_weights': [],
    }
    return params, variables


def initialize_optimizer(params, variables):
    """Initialize the optimizer for parameters.
    """
    lrs = {
        'means3D': 0.00016 * variables['scene_radius'],
        'rgb_colors': 0.0025,
        'unnorm_rotations': 0.001,
        'logit_opacities': 0.05,
        'log_scales': 0.001,
        'ctrl_t1': 0.00016 * variables['scene_radius'],
        'ctrl_r2': 0.001,
    }
    param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in params.items()]
    return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)


def residual_init_per_timestep(common_param:Common_Param, variables, only_ctrl=False, only_objs=False):
    """Initialize residual parameters for each timestep.
    """
    torch.cuda.set_device(common_param.device)
    # non-keyframe timesteps
    if only_ctrl:
        params = {
            'ctrl_t1': torch.zeros(variables['ctrl_t2'].shape[0], 1),
            'ctrl_r2': torch.zeros(variables['ctrl_r1'].shape[0], 2),
        }
    # keyframe timesteps && only update dynamic object gaussians
    elif only_objs:
        gspt_cate = variables['gs_cate']
        gs_indices = torch.where(gspt_cate != 0)[0]
        pts_num = gs_indices.shape[0]
        params = {
            'means3D': torch.zeros((pts_num,3)),
            'rgb_colors': torch.zeros((pts_num,3)),
            'unnorm_rotations': torch.tile(torch.tensor([1, 0, 0, 0]), (pts_num, 1)),
            'logit_opacities': torch.zeros((pts_num, 1)),
            'log_scales': torch.zeros((pts_num, 3)),
            'ctrl_t1': torch.zeros(variables['ctrl_t2'].shape[0], 1),
            'ctrl_r2': torch.zeros(variables['ctrl_r1'].shape[0], 2),
        }
    # keyframe timesteps && update all gaussians
    else:
        params = {
            'means3D': torch.zeros_like(variables['prev_pts']),
            'rgb_colors': torch.zeros_like(variables['prev_col']),
            'unnorm_rotations': torch.tile(torch.tensor([1, 0, 0, 0]), (variables['prev_rot'].shape[0], 1)),
            'logit_opacities': torch.zeros_like(variables['prev_opa']),
            'log_scales': torch.zeros_like(variables['prev_scl']),
            'ctrl_t1': torch.zeros(variables['ctrl_t2'].shape[0], 1),
            'ctrl_r2': torch.zeros(variables['ctrl_r1'].shape[0], 2),
        }
    params = {k: torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True)) for k, v in
              params.items()}
    return params


def residual_init_opt(params, only_ctrl=False):
    """Initialize the optimizer for residual parameters.
    """
    if only_ctrl:
        lrs = {
            'ctrl_t1': 0.001,
            'ctrl_r2': 0.01,
        }
    else:
        lrs = {
            'means3D': 0.001,
            'rgb_colors': 0.0025,
            'unnorm_rotations': 0.001,
            'logit_opacities': 0.05,
            'log_scales': 0.001,
            'ctrl_t1': 0.001,
            'ctrl_r2': 0.01,
        }
    param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in params.items()]
    return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)


def initialize_post_timestep(variables,render_params):
    """Initialize the variables using post timestep parameters.
    """
    variables['prev_pts'] = render_params['means3D'].detach()
    variables['prev_rot'] = torch.nn.functional.normalize(render_params['unnorm_rotations']).detach()
    variables['prev_col'] = render_params['rgb_colors'].detach()
    variables['prev_opa'] = render_params['logit_opacities'].detach()
    variables['prev_scl'] = render_params['log_scales'].detach()

#########################################################################################################
## standard densification functions, only used for initial timestep

def accumulate_mean2d_gradient(variables):
    variables['means2D_gradient_accum'] += torch.norm(
        variables['means2D'].grad[:, :2], dim=-1)
    variables['denom'] += 1
    return variables


def update_params_and_optimizer(new_params, params, optimizer):
    for k, v in new_params.items():
        group = [x for x in optimizer.param_groups if x["name"] == k][0]
        stored_state = optimizer.state.get(group['params'][0], None)

        stored_state["exp_avg"] = torch.zeros_like(v)
        stored_state["exp_avg_sq"] = torch.zeros_like(v)
        del optimizer.state[group['params'][0]]

        group["params"][0] = torch.nn.Parameter(v.requires_grad_(True))
        optimizer.state[group['params'][0]] = stored_state
        params[k] = group["params"][0]
    return params


def cat_params_to_optimizer(new_params, params, optimizer):
    for k, v in new_params.items():
        group = [g for g in optimizer.param_groups if g['name'] == k][0]
        stored_state = optimizer.state.get(group['params'][0], None)
        if stored_state is not None:
            stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(v)), dim=0)
            stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(v)), dim=0)
            del optimizer.state[group['params'][0]]
            group["params"][0] = torch.nn.Parameter(torch.cat((group["params"][0], v), dim=0).requires_grad_(True))
            optimizer.state[group['params'][0]] = stored_state
            params[k] = group["params"][0]
        else:
            group["params"][0] = torch.nn.Parameter(torch.cat((group["params"][0], v), dim=0).requires_grad_(True))
            params[k] = group["params"][0]
    return params


def remove_points(to_remove, params, variables, optimizer):
    to_keep = ~to_remove
    keys = [k for k in params.keys() if k not in ['ctrl_t1', 'ctrl_r2']]
    for k in keys:
        group = [g for g in optimizer.param_groups if g['name'] == k][0]
        stored_state = optimizer.state.get(group['params'][0], None)
        if stored_state is not None:
            stored_state["exp_avg"] = stored_state["exp_avg"][to_keep]
            stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][to_keep]
            del optimizer.state[group['params'][0]]
            group["params"][0] = torch.nn.Parameter((group["params"][0][to_keep].requires_grad_(True)))
            optimizer.state[group['params'][0]] = stored_state
            params[k] = group["params"][0]
        else:
            group["params"][0] = torch.nn.Parameter(group["params"][0][to_keep].requires_grad_(True))
            params[k] = group["params"][0]
    variables['means2D_gradient_accum'] = variables['means2D_gradient_accum'][to_keep]
    variables['denom'] = variables['denom'][to_keep]
    variables['max_2D_radius'] = variables['max_2D_radius'][to_keep]
    return params, variables


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def densify(params, variables, optimizer, i):
    if i <= 5000:
        variables = accumulate_mean2d_gradient(variables)
        grad_thresh = 0.0002
        if (i >= 500) and (i % 100 == 0):
            grads = variables['means2D_gradient_accum'] / variables['denom']
            grads[grads.isnan()] = 0.0
            to_clone = torch.logical_and(grads >= grad_thresh, (
                        torch.max(torch.exp(params['log_scales']), dim=1).values <= 0.01 * variables['scene_radius']))
            new_params = {k: v[to_clone] for k, v in params.items() if k not in ['ctrl_t1', 'ctrl_r2']}
            params = cat_params_to_optimizer(new_params, params, optimizer)
            num_pts = params['means3D'].shape[0]

            padded_grad = torch.zeros(num_pts, device="cuda")
            padded_grad[:grads.shape[0]] = grads
            
            to_split = torch.logical_and(padded_grad >= grad_thresh,
                                         torch.max(torch.exp(params['log_scales']), dim=1).values > 0.01 * variables[
                                             'scene_radius'])
            n = 2  # number to split into
            new_params = {k: v[to_split].repeat(n, 1) for k, v in params.items() if k not in ['ctrl_t1', 'ctrl_r2']}
            stds = torch.exp(params['log_scales'])[to_split].repeat(n, 1)
            means = torch.zeros((stds.size(0), 3), device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation_from_quaternion(params['unnorm_rotations'][to_split]).repeat(n, 1, 1)
            new_params['means3D'] += torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
            new_params['log_scales'] = torch.log(torch.exp(new_params['log_scales']) / (0.8 * n))
            params = cat_params_to_optimizer(new_params, params, optimizer)
            num_pts = params['means3D'].shape[0]

            variables['means2D_gradient_accum'] = torch.zeros(num_pts, device="cuda")
            variables['denom'] = torch.zeros(num_pts, device="cuda")
            variables['max_2D_radius'] = torch.zeros(num_pts, device="cuda")
            
            to_remove = torch.cat((to_split, torch.zeros(n * to_split.sum(), dtype=torch.bool, device="cuda")))
            params, variables = remove_points(to_remove, params, variables, optimizer)

            remove_threshold = 0.25 if i == 5000 else 0.005
            to_remove = (torch.sigmoid(params['logit_opacities']) < remove_threshold).squeeze()
            if i >= 3000:
                big_points_ws = torch.exp(params['log_scales']).max(dim=1).values > 0.1 * variables['scene_radius']
                to_remove = torch.logical_or(to_remove, big_points_ws)
            params, variables = remove_points(to_remove, params, variables, optimizer)

            torch.cuda.empty_cache()

        if i > 0 and i % 3000 == 0:
            new_params = {'logit_opacities': inverse_sigmoid(torch.ones_like(params['logit_opacities']) * 0.01)}
            params = update_params_and_optimizer(new_params, params, optimizer)

    return params, variables

#########################################################################################################


def get_loss(params:dict, curr_data:dict, variables:dict, is_initial_timestep:bool, 
             common_param:Common_Param, only_ctrl:bool=False, only_objs:bool=False, 
             transfromrot:bool=False, depth_flag:bool=False):
    """Compute the loss for the current training iteration.
    1. Render the image and depth map using the current parameters.
    2. Calculate the image reconstruction loss (L1 + SSIM) 
    3. Calculate the depth loss (L1) if depth_flag is True.
    4. Calculate optional regularization losses for control points if not initial timestep.
    5. Calculate optional regularization losses for Gaussian residuals at keyframe timesteps.
    6. Weight each loss component and sum them up.
    """
    losses = {}
    torch.cuda.set_device(common_param.device)
    if  is_initial_timestep:
        rendervar = params2rendervar(params)
    else:
        render_params = volume_warping(common_param,variables,params,only_ctrl=only_ctrl,only_objs=only_objs,transfromrot=transfromrot)
        rendervar = params2rendervar(render_params)
    rendervar['means2D'].retain_grad()
    im, _, depth, = Renderer(raster_settings=curr_data['cam'])(**rendervar)
    losses['im'] = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))
    variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification

    if depth_flag:
        min_val = depth.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        max_val = depth.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        nm_depth = (depth - min_val) / (max_val - min_val)
        losses['depth'] = l1_loss_v1(nm_depth, curr_data['depth'])
    
    ### optional regularization losses
    loss_weights = {
        'im': 1.0, 'depth': 0.0,
        'ctrl_reg_t': 0, 'ctrl_reg_r': 0,
        'gs_C': 0, 'gs_op': 0.0, 'gs_scl': 0.0, 'gs_t': 0.0,  'gs_r': 0.0,
        }
    
    if not is_initial_timestep and (loss_weights['ctrl_reg_t']!=0 or loss_weights['ctrl_reg_r']!=0):
        gspt_cate = variables['gs_cate']
        ctrl_self_idss = variables['ctrl_self_indices']
        ctrl_self_vss = variables['ctrl_self_vectors']
        ctrl_self_wgtss = variables['ctrl_self_weights']
        ctrl_t2 = variables['ctrl_t2']
        ctrl_r1 = variables['ctrl_r1']
        ctrl_t1 = params['ctrl_t1']
        ctrl_r2 = params['ctrl_r2']
        ctrl_c2w = variables['ctrl_c2w']
        ctrl_r2c = variables['ctrl_r2c']
        ctrl_qc2r = variables['ctrl_qc2r']
        ctrl_qr2c = variables['ctrl_qr2c']
        ctrl_qw2c = variables['ctrl_qw2c']
        ctrl_qc2w = variables['ctrl_qc2w']
        ctrl_cate = variables['ctrl_cate']        
        ### control point translation xyz
        ### converting from camera coor. system to world coor. system
        ctrl_t = torch.matmul(ctrl_c2w,torch.concatenate((ctrl_t2,ctrl_t1),dim=-1)[:,:,None]).squeeze() # 731,3
        ### control point rotation euler angles -> quaternion
        ### converting from ray coor. system to camera coor. system to world coor. system
        ctrl_r = build_quaternion_from_euler(ctrl_r1.squeeze(),ctrl_r2[:,0],ctrl_r2[:,1])
        q_camcoor = quat_mult(ctrl_qr2c,ctrl_r)
        ctrl_q = quat_mult(ctrl_qc2w,q_camcoor)

        ### manipulate gaussians with neighboring control points
        ctrl_weighted_t = torch.zeros_like(ctrl_t)
        ctrl_weighted_q = torch.zeros_like(ctrl_q)
        for i in range(1,1+common_param.obj_num):
            ctrl_indices = torch.where(ctrl_cate==i)[0]
            gs_indices = torch.where(gspt_cate==i)[0]
            if ctrl_indices.shape[0] < 4 or gs_indices.shape[0] == 0:
                ctrl_weighted_t[ctrl_indices,:] = ctrl_t[ctrl_indices]
                ctrl_weighted_q[ctrl_indices,:] = ctrl_q[ctrl_indices]
                continue
            ctrl_ts = ctrl_t[ctrl_indices] # 275,3
            ctrl_qs = ctrl_q[ctrl_indices] # 275,4
            ctrl_self_ids = ctrl_self_idss[i-1]
            ctrl_self_vs = ctrl_self_vss[i-1]
            ctrl_self_wgts = ctrl_self_wgtss[i-1]
            ctrl_c2ws = ctrl_c2w[ctrl_indices] # 275,3,3
            ctrl_r2cs = ctrl_r2c[ctrl_indices] # 275,3,3

            v_shape = ctrl_self_vs.shape
            ### active Rotation Self-Supervision, rotation affect on translation
            if transfromrot:
                ctrl_ts_fromrot = (rotate_vector(ctrl_self_vs.reshape(-1,3),ctrl_qs[ctrl_self_ids].reshape(-1,4)) - ctrl_self_vs.reshape(-1,3)).reshape(v_shape) # gs_num, knn, 3
                ctrl_ts_fromrot = torch.matmul(ctrl_c2ws[ctrl_self_ids], torch.matmul(ctrl_r2cs[ctrl_self_ids], ctrl_ts_fromrot[:,:,:,None])).squeeze() # gs_num, knn, 3
                ctrl_weighted_t[ctrl_indices,:] = torch.sum((ctrl_ts[ctrl_self_ids] + ctrl_ts_fromrot) * ctrl_self_wgts[:,:,None], dim=1) # 275,3
            ### without Rotation Self-Supervision(In practice, more stable)
            else:
                ctrl_weighted_t[ctrl_indices,:] = torch.sum(ctrl_ts[ctrl_self_ids] * ctrl_self_wgts[:,:,None], dim=1)
            ctrl_weighted_q[ctrl_indices,:] = torch.nn.functional.normalize(torch.sum(ctrl_qs[ctrl_self_ids] * ctrl_self_wgts[:,:,None], dim=1))
        
        
        ### regularization losses(not applied)
        losses['ctrl_reg_t'] = l1_loss_v2(ctrl_weighted_t, ctrl_t)
        losses['ctrl_reg_r'] = l1_loss_v2(ctrl_weighted_q, ctrl_q)


    ### keyframe timestep gaussian residual regularization losses
    if not is_initial_timestep and not only_ctrl:
        res_col = params['rgb_colors']
        res_op = params['logit_opacities']
        res_scl = params['log_scales']
        res_pts = params['means3D']
        res_rot = params['unnorm_rotations']
        no_rot = torch.tile(torch.tensor([[1, 0, 0, 0]]), (res_rot.shape[0], 1)).cuda()
        if loss_weights['gs_C'] != 0:
            losses['gs_C'] = l1_loss_v4(res_col)
        if loss_weights['gs_op'] != 0:
            losses['gs_op'] = l1_loss_v3(res_op)
        if loss_weights['gs_scl'] != 0:
            losses['gs_scl'] = l1_loss_v4(res_scl)
        if loss_weights['gs_t'] != 0:
            losses['gs_t'] = l1_loss_v4(res_pts)
        if loss_weights['gs_r'] != 0:
            losses['gs_r'] = l1_loss_v2(res_rot, no_rot)

    loss = sum([loss_weights[k] * v for k, v in losses.items()])

    return loss

def update_learning_rate(optimizer, xyz_scheduler_args, col_scheduler_args, opa_scheduler_args, iteration):
    ''' Learning rate scheduling per step.
    '''
    for param_group in optimizer.param_groups:
        if param_group["name"] == "means3D":
            lr = xyz_scheduler_args(iteration)
            param_group['lr'] = lr
        if param_group["name"] == "rgb_colors":
            lr = col_scheduler_args(iteration)
            param_group['lr'] = lr
        if param_group["name"] == "logit_opacities":
            lr = opa_scheduler_args(iteration)
            param_group['lr'] = lr    

def report_progress(params, eval_dataset, progress_bar, is_initial_timestep=True, common_param:Common_Param=None, variables=None, only_ctrl=False, only_objs=False, transfromrot=False):
    """Render evaluation images and compute PSNR to report training progress.
    """
    with torch.no_grad():
        if  is_initial_timestep:
            rendervar = params2rendervar(params)
        else:
            render_params = volume_warping(common_param,variables,params,only_ctrl=only_ctrl,only_objs=only_objs,transfromrot=transfromrot)
            rendervar = params2rendervar(render_params)
        todo_eval_dataset = []
        psnrs = []
        for i in range(len(common_param.test_cam_id)):
            todo_eval_dataset, eval_data = get_batch(common_param, todo_eval_dataset, eval_dataset)
            im, _, _, = Renderer(raster_settings=eval_data['cam'])(**rendervar)
            psnr = calc_psnr(im, eval_data['im']).mean()
            psnrs.append(psnr)
        psnr_mean = torch.stack(psnrs).mean()
        progress_bar.set_postfix({'train eval 0 PSNR': f'{psnr_mean:.{7}f}'})


def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """Exponential learning rate decay with delay.
    """
    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


# def update_learning_rate(optimizer, xyz_scheduler_args, iteration):
#     ''' Learning rate scheduling per step '''
#     for param_group in optimizer.param_groups:
#         if param_group["name"] == "means3D":
#             lr = xyz_scheduler_args(iteration)
#             param_group['lr'] = lr
#             return lr