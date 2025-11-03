import torch
import os
from utils.data_preprocess import Common_Param
from utils.rotation import bilinear_interpolate, build_quaternion_from_rotation, quat_mult, build_quaternion_from_euler, \
    rotate_vector, build_c2r_from_vector
from utils.format_converter import variables2rendervar
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from fast_pytorch_kmeans import KMeans
from knn_cuda import KNN

def get_depth(common_param:Common_Param, variables:dict, flowss:list, depths_for_H3D:list|None=None):
    """Organize depth maps with shape [obj][cam].
    During H3D control point generation, each object is manipulated independently, 
    and each camera independently generates the object's H3D control points.
    """
    torch.cuda.set_device(common_param.device)
    obj_num = common_param.obj_num

    tmp_depthss = []
    count = 0
    for cam_id, cam in enumerate(common_param.flow_id):
        if depths_for_H3D is not None:
            depth = depths_for_H3D[cam_id]
        else:
            _,_,depth = Renderer(raster_settings=common_param.cams[cam])(**variables2rendervar(variables))
            depth = depth.detach().squeeze()
        depths = []
        for i in range(obj_num):
            flow = flowss[i][cam_id]
            if flow is not None: 
                flow_uv = torch.tensor(flow[:,:2]).float().cuda()
                depth_sampled = bilinear_interpolate(depth, flow_uv)
                count += depth_sampled.shape[0]
                depths.append(depth_sampled)
            else:
                depths.append(None)
        tmp_depthss.append(depths)
    depthss = []
    for i in range(obj_num):
        depths = []
        for cam_id, cam in enumerate(common_param.flow_id):
            depth = tmp_depthss[cam_id][i]
            depths.append(depth)
        depthss.append(depths)
    return depthss


def ctrlvar_generator(commom_param:Common_Param, variables, flowss, depths_for_H3D=None):
    ''' Generate control point variables from 2D optical flow data.
    '''
    ctrlvar = {}
    torch.cuda.set_device(commom_param.device)
    
    flow_id = commom_param.flow_id
    flow_id_in_train = [commom_param.train_cam_id.index(flow_id[i]) for i in range(len(flow_id))]
    flowss = [[flows[i] for i in flow_id_in_train] for flows in flowss]
    # print(len(flowss),len(flowss[0]))
    camposs = commom_param.campos[flow_id,:] # (cam_num,3)
    h,w = commom_param.h, commom_param.w
    cx,cy = commom_param.cx[flow_id,None],commom_param.cy[flow_id,None]
    inv_ks = commom_param.inv_k[flow_id,:2,:2].reshape(len(flow_id),4) # (cam_num,4)
    c2ws = commom_param.c2w[flow_id,:3,:3].reshape(len(flow_id),9) # (cam_num,9)
    qc2ws = build_quaternion_from_rotation(commom_param.c2w[flow_id,:3,:3])
    qw2cs = build_quaternion_from_rotation(commom_param.w2c[flow_id,:3,:3])
    campos_invks_c2w_qc2w_qw2c = torch.concatenate((camposs,inv_ks,c2ws,qc2ws,qw2cs,cx,cy),dim=-1)
    ctrl_cate = torch.concatenate([torch.tensor(j+1).repeat(sum([flow.shape[0] for flow in flows if flow is not None])).int().cuda() for j,flows in enumerate(flowss)],axis=0)
    ctrl_campos_invks_c2w_qc2w_qw2c = torch.concatenate([torch.concatenate([torch.concatenate((torch.tile(campos_invks_c2w_qc2w_qw2c[None,i],(flow.shape[0],1)).cuda(),torch.tensor(flow).cuda()),dim=-1) for i,flow in enumerate(flows) if flow is not None], axis=0) for flows in flowss],axis=0).float().cuda()
    ctrl_camposs = ctrl_campos_invks_c2w_qc2w_qw2c[:,:3]
    inv_kss = ctrl_campos_invks_c2w_qc2w_qw2c[:,3:7].reshape(ctrl_campos_invks_c2w_qc2w_qw2c.shape[0],2,2)
    c2wss = ctrl_campos_invks_c2w_qc2w_qw2c[:,7:16].reshape(ctrl_campos_invks_c2w_qc2w_qw2c.shape[0],3,3)
    qc2w = ctrl_campos_invks_c2w_qc2w_qw2c[:,16:20]
    qw2c = ctrl_campos_invks_c2w_qc2w_qw2c[:,20:24]
    ctrl_c = ctrl_campos_invks_c2w_qc2w_qw2c[:,24:26]
    ctrl_uv = ctrl_campos_invks_c2w_qc2w_qw2c[:,26:28]
    ctrl_uv = torch.stack((ctrl_uv[:,1],ctrl_uv[:,0]),dim=-1)
    ctrl_t2 = -ctrl_campos_invks_c2w_qc2w_qw2c[:,28:30]
    ctrl_theta = -ctrl_campos_invks_c2w_qc2w_qw2c[:,30:31]
    ctrl_depthss = get_depth(commom_param, variables, flowss, depths_for_H3D) # z in cam_coords
    ctrl_depth = torch.concatenate([torch.concatenate([depth for i,depth in enumerate(depths) if depth is not None], axis=0) for depths in ctrl_depthss],axis=0).cuda()
    ctrl_vector = torch.concatenate((torch.matmul(inv_kss, (ctrl_uv-ctrl_c)[:,:,None]).squeeze(), torch.ones(ctrl_uv.shape[0],1).float().cuda()),dim=-1)

    # Since it represents a displacement, the last element of the homogeneous coordinate is 0; therefore, the c2w matrix can be reduced to its upper-left 3 \times 3 block.
    ctrl_xyz = ctrl_camposs + torch.matmul(c2wss,(ctrl_vector*ctrl_depth.unsqueeze(-1)).unsqueeze(-1)).squeeze()
    c2r = build_c2r_from_vector(ctrl_vector)
    r2c = c2r.transpose(-1, -2)
    qc2r = build_quaternion_from_rotation(c2r)
    qr2c = torch.concatenate((qc2r[:,0:1], -qc2r[:,1:4]), dim=1)
    
    ## the projection relation between the motion represented on the image plane and that in the ray coordinate system(real 3D space):
    ctrl_t = ctrl_depth.unsqueeze(-1) * torch.matmul(c2r[:,:2,:2],torch.matmul(inv_kss,ctrl_t2[:,:,None])).squeeze() # convert from image plane to ray coor. system
    ctrlvar['ctrl_xyz'] = ctrl_xyz.float() # (n,3) in world coor. system
    ctrlvar['ctrl_t2'] = ctrl_t.float() # (n,2) in ray coor. system
    ctrlvar['ctrl_r1'] = ctrl_theta.float() # (n) in ray coor. system
    ctrlvar['ctrl_r2c'] = r2c.float()
    ctrlvar['ctrl_c2w'] = c2wss.float() # (n,3,3)
    ctrlvar['ctrl_qc2r'] = qc2r.float() # (n,4)
    ctrlvar['ctrl_qr2c'] = qr2c.float() # (n,4)    
    ctrlvar['ctrl_qw2c'] = qw2c.float() # (n,4)
    ctrlvar['ctrl_qc2w'] = qc2w.float() # (n,4)
    ctrlvar['ctrl_cate'] = ctrl_cate.int() # (n)
    for k, v in ctrlvar.items():
        variables[k] = v


def ctrl_select(common_param:Common_Param, variables, dataset):
    """ project control points onto masks and use majority,
    compare results to the original cate, select faithful control points
    """
    torch.cuda.set_device(common_param.device)
    
    ctrl_xyz = variables['ctrl_xyz']
    ctrl_t2 = variables['ctrl_t2']
    ctrl_r1 = variables['ctrl_r1']
    ctrl_cate = variables['ctrl_cate']
    ctrl_r2c = variables['ctrl_r2c']
    ctrl_c2w = variables['ctrl_c2w']
    ctrl_qw2c = variables['ctrl_qw2c']
    ctrl_qc2w = variables['ctrl_qc2w']
    ctrl_qr2c = variables['ctrl_qr2c']
    ctrl_qc2r = variables['ctrl_qc2r']

    obj_num = common_param.obj_num
    maskss = [data['masks'].int().cuda() for data in dataset if data['masks'] is not None]
    pts = ctrl_xyz
    cate_matrix = torch.zeros((pts.shape[0], obj_num+1)).cuda()
    mask_indicess = [[torch.nonzero(masks[i]) for i in range(masks.shape[0])] for masks in maskss]
    ctrl = torch.concatenate((pts,torch.ones(pts.shape[0],1).float().cuda()),axis=1)[:,:,None]
    
    for cam_id,cam in enumerate(common_param.mask_id):
        pts2d = torch.matmul(torch.tile(common_param.proj[None, cam],(pts.shape[0],1,1)).cuda(), ctrl).squeeze()
        ptsuv = pts2d / pts2d[:,2].unsqueeze(-1)
        pt_rounded =torch.stack((torch.round(ptsuv[:,1]), torch.round(ptsuv[:,0])),dim=-1) #from xy -> hw
        pt_trans = pt_rounded[:,0] * common_param.w + pt_rounded[:,1]
        for i in range(obj_num+1):
            mask_indice = mask_indicess[cam_id][i]
            if mask_indice.shape[0] > 0:
                mask_trans = mask_indice[:,0] * common_param.w + mask_indice[:,1]
                exists = torch.isin(pt_trans, mask_trans)
                indices = torch.where(exists)[0]
                cate_matrix[indices,i] += 1
    
    ctrl_cate2 = torch.argmax(cate_matrix,dim=1).int()
    idx = torch.where(ctrl_cate2==ctrl_cate)[0]
    variables['ctrl_xyz'] = ctrl_xyz[idx]
    variables['ctrl_t2'] = ctrl_t2[idx]
    variables['ctrl_r1'] = ctrl_r1[idx]
    variables['ctrl_cate'] = ctrl_cate[idx]
    variables['ctrl_r2c'] = ctrl_r2c[idx]
    variables['ctrl_c2w'] = ctrl_c2w[idx]
    variables['ctrl_qw2c'] = ctrl_qw2c[idx]
    variables['ctrl_qc2w'] = ctrl_qc2w[idx]
    variables['ctrl_qr2c'] = ctrl_qr2c[idx]
    variables['ctrl_qc2r'] = ctrl_qc2r[idx]
    return idx


def ctrl_prune(common_param:Common_Param, variables, ctrl_ratio=0.5):
    """ Prune control points using k-means clustering.(Optional)
    """
    torch.cuda.set_device(common_param.device)
    ctrl_xyz = variables['ctrl_xyz']
    ctrl_t2 = variables['ctrl_t2']
    ctrl_r1 = variables['ctrl_r1']
    ctrl_cate = variables['ctrl_cate']
    ctrl_r2c = variables['ctrl_r2c']
    ctrl_c2w = variables['ctrl_c2w']
    ctrl_qw2c = variables['ctrl_qw2c']
    ctrl_qc2w = variables['ctrl_qc2w']
    ctrl_qr2c = variables['ctrl_qr2c']
    ctrl_qc2r = variables['ctrl_qc2r']

    new_ctrl_xyz = []
    new_ctrl_t2 = []
    new_ctrl_r1 = []
    new_ctrl_cate = []
    new_ctrl_r2c = []
    new_ctrl_c2w = []
    new_ctrl_qw2c = []
    new_ctrl_qc2w = []
    new_ctrl_qr2c = []
    new_ctrl_qc2r = []
    idx = []

    device = common_param.device
    success = False
    while not success:
        try:
            for o in range(common_param.obj_num):
                obj_ctrl_idx = torch.where(ctrl_cate==o+1)[0]
                obj_ctrl_xyz = ctrl_xyz[obj_ctrl_idx]
                num_clusters = int(obj_ctrl_xyz.shape[0] * ctrl_ratio)
                if num_clusters < 4:
                    num_clusters = 4
                kmeans = KMeans(n_clusters=num_clusters, mode='euclidean', init_method='kmeans++')
                cluster_ids_x = kmeans.fit_predict(obj_ctrl_xyz)
                cluster_centers = kmeans.centroids

                for i in range(num_clusters):
                    cluster_ids = torch.where(cluster_ids_x==i)[0].int().cuda()
                    cluster_center = cluster_centers[i].float().cuda()
                    cluster_ctrl_xyz = obj_ctrl_xyz[cluster_ids]
                    dist = torch.sum((cluster_ctrl_xyz - cluster_center)**2,dim=1)
                    if dist.numel() == 0:
                        raise ValueError("Empty cluster detected, retrying...")
                    nearest_idx = torch.argmin(dist)
                    new_ctrl_xyz.append(cluster_ctrl_xyz[nearest_idx])
                    new_ctrl_t2.append(ctrl_t2[obj_ctrl_idx][cluster_ids][nearest_idx])
                    new_ctrl_r1.append(ctrl_r1[obj_ctrl_idx][cluster_ids][nearest_idx])
                    new_ctrl_cate.append(ctrl_cate[obj_ctrl_idx][cluster_ids][nearest_idx])
                    new_ctrl_r2c.append(ctrl_r2c[obj_ctrl_idx][cluster_ids][nearest_idx])
                    new_ctrl_c2w.append(ctrl_c2w[obj_ctrl_idx][cluster_ids][nearest_idx])
                    new_ctrl_qw2c.append(ctrl_qw2c[obj_ctrl_idx][cluster_ids][nearest_idx])
                    new_ctrl_qc2w.append(ctrl_qc2w[obj_ctrl_idx][cluster_ids][nearest_idx])
                    new_ctrl_qr2c.append(ctrl_qr2c[obj_ctrl_idx][cluster_ids][nearest_idx])
                    new_ctrl_qc2r.append(ctrl_qc2r[obj_ctrl_idx][cluster_ids][nearest_idx])
                    idx.append(obj_ctrl_idx[cluster_ids[nearest_idx]])
            success = True
        except ValueError as e:
            print(e)
            
    variables['ctrl_xyz'] = torch.stack(new_ctrl_xyz).float().cuda()
    variables['ctrl_t2'] = torch.stack(new_ctrl_t2).float().cuda()
    variables['ctrl_r1'] = torch.stack(new_ctrl_r1).float().cuda()
    variables['ctrl_cate'] = torch.stack(new_ctrl_cate).int().cuda()
    variables['ctrl_r2c'] = torch.stack(new_ctrl_r2c).float().cuda()
    variables['ctrl_c2w'] = torch.stack(new_ctrl_c2w).float().cuda()
    variables['ctrl_qw2c'] = torch.stack(new_ctrl_qw2c).float().cuda()
    variables['ctrl_qc2w'] = torch.stack(new_ctrl_qc2w).float().cuda()
    variables['ctrl_qr2c'] = torch.stack(new_ctrl_qr2c).float().cuda()
    variables['ctrl_qc2r'] = torch.stack(new_ctrl_qc2r).float().cuda()
    return torch.stack(idx).int().cuda()


def get_id_wgt(common_param:Common_Param, variables, gc_knn_num, cc_knn_num):
    """ Compute the kNN indices and weights for motion manipulation.
    consider both the graph from control points to surface points and the graph among control points themselves.
    """
    torch.cuda.set_device(common_param.device)
    gs_pts = variables['prev_pts']
    ctrl_pts = variables['ctrl_xyz']
    ctrl_cate = variables['ctrl_cate']
    gspt_cate = variables['gs_cate']
    variables['ctrl_indices'] = []
    variables['ctrl_vectors'] = []
    variables['ctrl_weights'] = []
    variables['ctrl_self_indices'] = []
    variables['ctrl_self_vectors'] = []
    variables['ctrl_self_weights'] = []
    for i in range(common_param.obj_num):
        gs_indices = torch.where(gspt_cate==i+1)  
        gs_indices = gs_indices[0].cuda()
        
        if gs_indices.shape[0] == 0:
            print('obj', i+1,'has no points')
            variables['ctrl_indices'].append(None)
            variables['ctrl_vectors'].append(None)
            variables['ctrl_weights'].append(None)
            variables['ctrl_self_indices'].append(None)
            variables['ctrl_self_vectors'].append(None)
            variables['ctrl_self_weights'].append(None)
            continue
        ctrl_indices = torch.where(ctrl_cate==i+1)
        ctrl_indices = ctrl_indices[0].cuda()
        if ctrl_indices.shape[0] < 4:
            print('obj', i+1,'has less than 4 control points')
            variables['ctrl_indices'].append(None)
            variables['ctrl_vectors'].append(None)
            variables['ctrl_weights'].append(None)
            variables['ctrl_self_indices'].append(None)
            variables['ctrl_self_vectors'].append(None)
            variables['ctrl_self_weights'].append(None)
            continue
            
        gc_knn = KNN(k=gc_knn_num, transpose_mode=True)
        dist, indices = gc_knn(ctrl_pts[ctrl_indices].detach().unsqueeze(0), gs_pts[gs_indices].detach().unsqueeze(0))
        cc_knn = KNN(k=cc_knn_num+1, transpose_mode=True)
        self_dist, self_indices = cc_knn(ctrl_pts[ctrl_indices].detach().unsqueeze(0), ctrl_pts[ctrl_indices].detach().unsqueeze(0))
        dist = dist.squeeze().float().cuda()
        indices = indices.squeeze().int().cuda()
        ctrl_vectors = gs_pts[gs_indices][:,None,:] - ctrl_pts[ctrl_indices][indices]
        self_dist = self_dist[:,:,1:].squeeze().float().cuda()
        self_indices = self_indices[:,:,1:].squeeze().int().cuda()
        ctrl_self_vectors = ctrl_pts[ctrl_indices][:,None,:] - ctrl_pts[ctrl_indices][self_indices]
        weights = 1.0 / (dist + 1e-8)
        weights = weights / torch.sum(weights, dim=1, keepdim=True)
        self_weights = 1.0 / (self_dist + 1e-8)
        self_weights = self_weights / torch.sum(self_weights, dim=1, keepdim=True)
        variables['ctrl_indices'].append(indices)
        variables['ctrl_vectors'].append(ctrl_vectors)
        variables['ctrl_weights'].append(weights)
        variables['ctrl_self_indices'].append(self_indices)
        variables['ctrl_self_vectors'].append(ctrl_self_vectors)
        variables['ctrl_self_weights'].append(self_weights)


def volume_warping(common_param:Common_Param, variables, params=None, only_ctrl=False, only_objs=False, transfromrot=False):
    """ Manipulate the 3D points based on control points and their transformations.
    """
    if params is None:
        only_ctrl = True

    prev_pts = variables['prev_pts']
    prev_rot = variables['prev_rot']
    prev_col = variables['prev_col']
    prev_opa = variables['prev_opa']
    prev_scl = variables['prev_scl']

    if not only_ctrl:
        res_pts = params['means3D']
        res_rot = params['unnorm_rotations']
        res_col = params['rgb_colors']
        res_op = params['logit_opacities']
        res_scl = params['log_scales']
        
    ctrl_t2 = variables['ctrl_t2']
    ctrl_r1 = variables['ctrl_r1']
    
    if params is None:
        ctrl_t1 = torch.zeros(variables['ctrl_t2'].shape[0], 1).cuda()
        ctrl_r2 = torch.zeros(variables['ctrl_r1'].shape[0], 2).cuda()
    else:
        ctrl_t1 = params['ctrl_t1']
        ctrl_r2 = params['ctrl_r2']
    
    ctrl_r2c = variables['ctrl_r2c']
    ctrl_c2w = variables['ctrl_c2w']
    ctrl_qc2r = variables['ctrl_qc2r']
    ctrl_qr2c = variables['ctrl_qr2c']
    ctrl_qw2c = variables['ctrl_qw2c']
    ctrl_qc2w = variables['ctrl_qc2w']
    ctrl_idss = variables['ctrl_indices']
    ctrl_vss = variables['ctrl_vectors']
    ctrl_wgtss = variables['ctrl_weights']

    ctrl_cate = variables['ctrl_cate']
    gspt_cate = variables['gs_cate']

    render_pts = prev_pts.clone()
    render_rot = prev_rot.clone()
    render_col = prev_col.clone()
    render_op = prev_opa.clone()
    render_scl = prev_scl.clone()

    ctrl_t = torch.matmul(ctrl_c2w, torch.matmul(ctrl_r2c, torch.concatenate((ctrl_t2,ctrl_t1),dim=-1)[:,:,None])).squeeze()
    
    ctrl_r = build_quaternion_from_euler(ctrl_r1.squeeze(),ctrl_r2[:,0],ctrl_r2[:,1])
    q_camcoor = quat_mult(ctrl_qr2c,ctrl_r)
    ctrl_q = quat_mult(ctrl_qc2w,q_camcoor)

    for i in range(common_param.obj_num+1):
        obj_indice_init = 0
        gs_indices = torch.where(gspt_cate==i)[0]
        
        if gs_indices.shape[0] == 0:
            print('obj', i,'has no points')
            continue
        
        obj_gs_num = gs_indices.shape[0]
        if i == 0:
            if only_objs or only_ctrl:
                continue
            else:
                render_pts[gs_indices] = res_pts[gs_indices] + prev_pts[gs_indices]
                render_rot[gs_indices] = quat_mult(res_rot[gs_indices], prev_rot[gs_indices])
        else:
            ctrl_indices = torch.where(ctrl_cate==i)[0] # 275
            if ctrl_indices.shape[0] < 4:
                # print('obj', i,'has less than 4 control points')
                if only_objs or only_ctrl:
                    continue
                else:
                    render_pts[gs_indices] = res_pts[gs_indices] + prev_pts[gs_indices]
                    render_rot[gs_indices] = quat_mult(res_rot[gs_indices], prev_rot[gs_indices])
                    continue
            ctrl_ts = ctrl_t[ctrl_indices] # 731->275
            ctrl_qs = ctrl_q[ctrl_indices]
            ctrl_ids = ctrl_idss[i-1] # gs_num, knn
            ctrl_wgts = ctrl_wgtss[i-1] # gs_num, knn
            ctrl_vs = ctrl_vss[i-1] # gs_num, knn, 3
            ctrl_c2ws = ctrl_c2w[ctrl_indices] # 275,3,3
            ctrl_r2cs = ctrl_r2c[ctrl_indices] # 275,3,3
            v_shape = ctrl_vs.shape
            if transfromrot:
                ctrl_ts_fromrot = (rotate_vector(ctrl_vs.reshape(-1,3),ctrl_qs[ctrl_ids].reshape(-1,4)) - ctrl_vs.reshape(-1,3)).reshape(v_shape) # gs_num, knn, 3
                ctrl_ts_fromrot = torch.matmul(ctrl_c2ws[ctrl_ids], torch.matmul(ctrl_r2cs[ctrl_ids], ctrl_ts_fromrot[:,:,:,None])).squeeze() # gs_num, knn, 3
                weighted_t = torch.sum((ctrl_ts[ctrl_ids] + ctrl_ts_fromrot) * ctrl_wgts[:,:,None], dim=1) # gs_num, 3
            else:
                weighted_t = torch.sum(ctrl_ts[ctrl_ids] * ctrl_wgts[:,:,None], dim=1) # gs_num, 3
                
            weighted_q = torch.nn.functional.normalize(torch.sum(ctrl_qs[ctrl_ids] * ctrl_wgts[:,:,None], dim=1))
            
            if only_ctrl:
                render_pts[gs_indices] = weighted_t + prev_pts[gs_indices]
                render_rot[gs_indices] = quat_mult(weighted_q, prev_rot[gs_indices])
            elif only_objs:
                # obj_indice = torch.nonzero(torch.isin(obj_indices, gs_indices), as_tuple=True)[0]
                render_pts[gs_indices] = res_pts[obj_indice_init:obj_indice_init+obj_gs_num] + weighted_t + prev_pts[gs_indices]
                render_rot[gs_indices] = quat_mult(res_rot[obj_indice_init:obj_indice_init+obj_gs_num], quat_mult(weighted_q, prev_rot[gs_indices]))
                render_col[gs_indices] = prev_col[gs_indices] + res_col[obj_indice_init:obj_indice_init+obj_gs_num]
                render_op[gs_indices] = prev_opa[gs_indices] + res_op[obj_indice_init:obj_indice_init+obj_gs_num]
                render_scl[gs_indices] = prev_scl[gs_indices] + res_scl[obj_indice_init:obj_indice_init+obj_gs_num]
                obj_indice_init = obj_indice_init + obj_gs_num
            else:
                render_pts[gs_indices] = res_pts[gs_indices] + weighted_t + prev_pts[gs_indices]
                render_rot[gs_indices] = quat_mult(res_rot[gs_indices], quat_mult(weighted_q, prev_rot[gs_indices]))
    
    if not only_ctrl and not only_objs:
        render_col = prev_col + res_col
        render_op = prev_opa + res_op
        render_scl = prev_scl + res_scl
        
    render_params = {
        'means3D': render_pts,
        'rgb_colors': render_col,
        'unnorm_rotations': render_rot,
        'logit_opacities': render_op,
        'log_scales': render_scl,
    }
    return render_params
