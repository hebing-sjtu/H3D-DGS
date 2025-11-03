import torch
from utils.data_preprocess import Common_Param

def gs_cate(common_param:Common_Param, variables:dict, dataset:list):
    """Assign category labels to each Gaussian based on 2D masks from multiple camera
    """
    
    torch.cuda.set_device(common_param.device)
    obj_num = common_param.obj_num
    maskss = [data['masks'].int().cuda() for data in dataset if data['masks'] is not None]
    pts = variables['prev_pts']
    cate_matrix = torch.zeros((pts.shape[0], obj_num+1)).cuda()
    mask_indicess = [[torch.nonzero(masks[i]) for i in range(masks.shape[0])] for masks in maskss]
    gs = torch.concatenate((pts,torch.ones(pts.shape[0],1).float().cuda()),axis=1)[:,:,None]
    for cam_id,cam in enumerate(common_param.mask_id):
        pts2d = torch.matmul(torch.tile(common_param.proj[None, cam],(pts.shape[0],1,1)).cuda(), gs).squeeze()
        ptsuv = pts2d / pts2d[:,2].unsqueeze(-1)
        pt_rounded =torch.stack((torch.round(ptsuv[:,1]), torch.round(ptsuv[:,0])),dim=-1) #from xy -> hw
        pt_trans = pt_rounded[:,0] * common_param.w + pt_rounded[:,1]
        for i in range(obj_num+1):
            mask_indice = mask_indicess[cam_id][i]
            if mask_indice.shape[0] > 0:
                # isin func not supported for parallel in channel dimension
                mask_trans = mask_indice[:,0] * common_param.w + mask_indice[:,1]
                exists = torch.isin(pt_trans, mask_trans)
                indices = torch.where(exists)[0]
                cate_matrix[indices,i] += 1
    
    gs_cate = torch.argmax(cate_matrix,dim=1).int().cuda()
    variables['gs_cate'] = gs_cate