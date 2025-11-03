import os
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import pickle

from tqdm import tqdm
from utils.data_preprocess import get_common_param
from utils.rotation import build_c2r_from_vector
from utils.optical_utils import *

def unbias_rot(cluster_center, radius, trans, c, f):
    '''
    dim order xy/wh
    camid int, range from 0 to cam_num
    cluster_center: [N,2] (y,x)
    radius: [N,2]
    trans:  [N,2]
    c:      [1,2] torch on correct device
    f:      scalar torch tensor on correct device
    '''
    device = cluster_center.device
    if not torch.is_tensor(f):
        f = torch.tensor(f, device=device, dtype=torch.float32)
    f = f.view(1)  # [1]
    ## vector from camera center to cluster center
    vec = torch.cat([cluster_center - c, f.expand(cluster_center.shape[0]).unsqueeze(1)], dim=1)  # [N,3]
    c2r = build_c2r_from_vector(vec)
    
    ## Convert to homogeneous terms
    zeros_N1 = torch.zeros((radius.shape[0], 1), device=device, dtype=torch.float32)
    radius3 = torch.cat([radius, zeros_N1], dim=1)  # [N,3]
    trans3  = torch.cat([trans,  zeros_N1], dim=1)  # [N,3]

    radius3 = torch.matmul(c2r, radius3.unsqueeze(-1)).squeeze(-1)
    trans3  = torch.matmul(c2r, trans3 .unsqueeze(-1)).squeeze(-1)

    radius3[..., 2] = 0
    trans3[...,  2] = 0
    
    ## calculate angle
    r_norm2 = torch.clamp(torch.sum(radius3 * radius3, dim=1), min=1e-8)
    cross_z = torch.cross(trans3, radius3, dim=1)[:, 2]
    sin_theta = cross_z / r_norm2
    sin_theta = torch.clamp(sin_theta, -1.0, 1.0)

    theta = torch.arcsin(sin_theta)
    return theta


def tensor_to_image(tensor):
    # Convert tensor from (1, 3, H, W) to (H, W, 3)
    image = np.transpose(tensor[0].cpu().numpy(), (1, 2, 0))
    # Convert from range [0, 1] to [0, 255]
    image = (image * 255).astype(np.uint8)
    return image



def even_sample(flow, mask_points, mask, grid_size, args):
    device = flow.device
    
    if mask_points.numel() == 0:
        return torch.empty((0, 2), dtype=torch.long, device=device)
    
    min_h, max_h = mask_points[:, 0].min(), mask_points[:, 0].max()
    min_w, max_w = mask_points[:, 1].min(), mask_points[:, 1].max()

    rand_int_h = torch.randint(0, grid_size // 2, (1,)).item() if args.random_shift else 0
    rand_int_w = torch.randint(0, grid_size // 2, (1,)).item() if args.random_shift else 0

    start_i = min(min_h + rand_int_h, max_h - grid_size)
    start_j = min(min_w + rand_int_w, max_w - grid_size)
    
    # interesting region
    roi_mask = mask[start_i:max_h + 1, start_j:max_w + 1]
    roi_H, roi_W = roi_mask.shape
    
    # padding to make sure full grids
    num_h = (roi_H + grid_size - 1) // grid_size
    num_w = (roi_W + grid_size - 1) // grid_size
    pad_h = num_h * grid_size - roi_H
    pad_w = num_w * grid_size - roi_W
    roi_mask = F.pad(roi_mask, (0, pad_w, 0, pad_h))  # [roi_H+pad, roi_W+pad]
    
    # reshape and permute to get blocks [num_h, num_w, grid_size, grid_size]
    blocks = roi_mask.view(num_h, grid_size, num_w, grid_size).permute(0, 2, 1, 3).contiguous()

    counts = blocks.sum(dim=(-1, -2))
    valid = counts >= max(1, (grid_size // 2)**2)
    
    ys = torch.arange(grid_size, device=device).view(1, 1, grid_size, 1)
    xs = torch.arange(grid_size, device=device).view(1, 1, 1, grid_size)
    sum_y = (blocks * ys).sum(dim=(-1, -2))
    sum_x = (blocks * xs).sum(dim=(-1, -2))
    mean_y = sum_y / (counts + 1e-8)
    mean_x = sum_x / (counts + 1e-8)
    
    # add offsets to get absolute coordinates
    grid_y_offset = torch.arange(num_h, device=device)[:, None] * grid_size
    grid_x_offset = torch.arange(num_w, device=device)[None, :] * grid_size
    center_y = mean_y + grid_y_offset
    center_x = mean_x + grid_x_offset

    valid_y = center_y[valid]
    valid_x = center_x[valid]

    if valid_y.numel() == 0:
        return torch.empty((0, 2), dtype=torch.long, device=device)

    # convert to original image coordinates
    center_points = torch.stack([valid_y + start_i, valid_x + start_j], dim=-1).long()

    return center_points

def process_frame(args, frame_id, gpu_id):
    torch.cuda.set_device(gpu_id)
    common_param = args.common_param
    train_idxs = common_param.train_cam_id
    train_cam_num = len(train_idxs)
    dataset_train_cam_id = common_param.dataset_train_cam_id
    ks = common_param.k
    camera_view_param_list = [None] * train_cam_num
    
    for train_idx in train_idxs:
        k = ks[train_idx]
        cx, cy = k[0, 2], k[1, 2]
        c = torch.tensor([cx, cy], device=f"cuda:{gpu_id}").unsqueeze(0)
        f = (k[0, 0] + k[1, 1]) / 2
        camera_view_param_list[train_idx] = (c, f)

    frame_result = []
    for object_id in range(1, args.object_num + 1):
        view_result = []
        for i, view_id in enumerate(train_idxs):
            dataset_train_idx = dataset_train_cam_id[i]
            ext = "jpg" if common_param.is_sport else "png"
            imfile1 = f"{args.root_path}/imgs/{dataset_train_idx}/{frame_id:04d}.{ext}"
            imfile2 = f"{args.root_path}/imgs/{dataset_train_idx}/{frame_id+1:04d}.{ext}"
            if not os.path.exists(imfile1):
                view_result.append(None)
                continue
            # image1 = read_image_totorch(imfile1).cuda(gpu_id)
            # image2 = read_image_totorch(imfile2).cuda(gpu_id)
            # image1_np = tensor_to_image(image1)
            # image2_np = tensor_to_image(image2)
            # gray1 = cv2.cvtColor(image1_np, cv2.COLOR_RGB2GRAY)
            # gray2 = cv2.cvtColor(image2_np, cv2.COLOR_RGB2GRAY)
            gray1 = cv2.imread(imfile1, cv2.IMREAD_GRAYSCALE)
            gray2 = cv2.imread(imfile2, cv2.IMREAD_GRAYSCALE)

            if common_param.is_sport:
                big_mask_pth1 = f"{args.root_path}/mask_00/{dataset_train_idx}/{frame_id:04d}.png"
                mask_pth1 = f"{args.root_path}/mask_0{object_id}/{dataset_train_idx}/{frame_id:04d}.png"
            else:
                big_mask_pth1 = f"{args.root_path}/mask_00/{dataset_train_idx}/{frame_id:05d}.png"
                mask_pth1 = f"{args.root_path}/mask_0{object_id}/{dataset_train_idx}/{frame_id:05d}.png"
            
            big_mask1 = 1-read_mask_totorch(big_mask_pth1).cuda()
            mask1 = read_mask_totorch(mask_pth1).cuda()
            
            if not os.path.exists(mask_pth1):
                view_result.append(None)
                continue

            if gray1 is None or gray2 is None:
                view_result.append(None); continue      
            
            if args.flow_type == "Farneback":
                flow = cv2.calcOpticalFlowFarneback(gray2, gray1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            elif args.flow_type == "DeepFlow":
                dis = cv2.optflow.createOptFlow_DeepFlow()
                flow = dis.calc(gray2, gray1, None, )
            elif args.flow_type == "FB":
                flow = cv2.optflow.calcOpticalFlowSparseToDense(gray2, gray1, None)
            elif args.flow_type == "TBL":
                optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()
                flow = optical_flow.calc(gray2, gray1, None)
            elif args.flow_type == "DIS":
                optical_flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
                flow = optical_flow.calc(gray2, gray1, None)


            flow = torch.tensor(flow).cuda().permute(2, 0, 1).unsqueeze(0)
            N,C,H,W = flow.shape
            flow_show = flow * big_mask1
            flow *= mask1
            segmented_img = flow_show #blurred_flow  # [0].permute(1,2,0).cpu().numpy()

            real_positions = []
            real_position_flows = []
            angles = []

            mask = mask1[0][0]
            mask_points = torch.nonzero(mask == 1, as_tuple=False)
            if mask_points.shape[0] == 0:
                view_result.append(None)
                continue
            
            grid_size = args.grid_size
            sample_num = mask_points.shape[0] // grid_size**2
            if args.even_sample:
                grid_points = even_sample(flow, mask_points, mask, grid_size, args)
                sample_num = len(grid_points)
                if sample_num == 0:
                    sample_num = 1
                    cent_point = torch.mean(mask_points.clone().float(), dim=0).long()
                    grid_points = cent_point.unsqueeze(0)
            else:
                if sample_num == 0:
                    sample_num = 1
                    cent_point = torch.mean(mask_points.clone().float(), dim=0).long()
                    grid_points = cent_point.unsqueeze(0)
                else:
                    perm = torch.randperm(mask_points.shape[0])
                    grid_points = mask_points[perm[:sample_num]]

                # 将列表转换为 tensor
            selected_coords = grid_points
            
            radius = args.radius
            
            h_mins = torch.max(torch.stack((selected_coords[:,0]-radius, torch.zeros(sample_num,device=selected_coords.device))),dim=0)[0].long() #(n)
            h_maxs = torch.min(torch.stack((selected_coords[:,0]+radius, H*torch.ones(sample_num,device=selected_coords.device))),dim=0)[0].long()
            w_mins = torch.max(torch.stack((selected_coords[:,1]-radius, torch.zeros(sample_num,device=selected_coords.device))),dim=0)[0].long() 
            w_maxs = torch.min(torch.stack((selected_coords[:,1]+radius, W*torch.ones(sample_num,device=selected_coords.device))),dim=0)[0].long()
            
            H, W = mask1.shape[-2:]
            Y_all, X_all = torch.meshgrid(torch.arange(H, device='cuda'), torch.arange(W, device='cuda'), indexing='ij')

            
            for i, (y, x) in enumerate(selected_coords):
                h_min, h_max = h_mins[i], h_maxs[i]
                w_min, w_max = w_mins[i], w_maxs[i]

                # extract flow vectors in the subregion
                flow_vectors = segmented_img[0, :, h_min:h_max, w_min:w_max].permute(1, 2, 0).reshape(-1, 2)
                mask_subregion = mask1[0, 0, h_min:h_max, w_min:w_max].flatten()

                # calculate circle mask
                X = X_all[h_min:h_max, w_min:w_max]
                Y = Y_all[h_min:h_max, w_min:w_max]

                circle_mask = (X - x) ** 2 + (Y - y) ** 2 <= radius ** 2
                valid_mask = mask_subregion.bool() & circle_mask.flatten()

                # calculate mean flow and angle
                connect_vectors = torch.stack([Y.flatten() - y, X.flatten() - x], dim=1).float()
                connect_vectors = connect_vectors[valid_mask]
                motions = -flow_vectors[valid_mask][:, [1, 0]]

                cluster_center = torch.tensor([[y, x]], device='cuda', dtype=torch.float32)
                theta = unbias_rot(cluster_center, connect_vectors, motions, 
                                   camera_view_param_list[view_id][0], 
                                   camera_view_param_list[view_id][1])


                if theta.numel() > 0:
                    theta_mean = torch.nanmean(theta)
                    angles.append(theta_mean.item())

                # print(flow_vectors.shape)
                flow_mean = torch.mean(flow_vectors[valid_mask], dim=0)
                real_position_flows.append(flow_mean.cpu().numpy())
                real_positions.append((y.item(), x.item()))

            if len(real_positions) > 0:
                if len(real_positions) == 1 and len(angles) == 0:
                    angles = [0]
                real_positions = np.array(real_positions)
                real_position_flows = np.array(real_position_flows)
                angles = np.array(angles).reshape(-1, 1)
                # print(real_positions.shape, real_position_flows.shape, angles.shape)
                result_array = np.concatenate((real_positions, real_position_flows, angles), axis=1)
            else:
                result_array = None
            view_result.append(result_array)
        frame_result.append(view_result)

    return (frame_id, frame_result)


def worker(gpu_id, fid_list, args):
    torch.cuda.set_device(gpu_id)
    out = {}
    for fid in fid_list:
        k, v = process_frame(args, fid, gpu_id)
        out[str(k)] = v
    return out


def demo(args):
    num_timesteps = len(os.listdir(f'{args.dataset_dir}/{args.seq}/imgs/0'))
    frame_ids = list(range(num_timesteps - 1))
    num_gpus = torch.cuda.device_count()
    time_result = {}

    # 将帧平均分配到不同GPU
    gpu_assignments = {gpu_id: [] for gpu_id in range(num_gpus)}
    for i, fid in enumerate(frame_ids):
        gpu_assignments[i % num_gpus].append(fid)

    # 使用 torch.multiprocessing.Pool
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=num_gpus) as pool:
        jobs = []
        for gpu_id, fid_list in gpu_assignments.items():
            jobs.append(pool.apply_async(worker, (gpu_id, fid_list, args)))
        pool.close()
        merged = {}
        for j in tqdm(jobs):
            merged.update(j.get())

    with open(args.result_dict_path, 'wb') as file:
        pickle.dump(merged, file)

    print("✅ Parallel demo finished.")


if __name__ == '__main__':
    # RAFT 的配置
    parser = argparse.ArgumentParser()
    parser.add_argument('--radius', type=int, default=8, help="optical flow radius")
    
    args = parser.parse_args()
    # args.model = "/home/unoC/workplace/2d23d/pretrained/models/raft-sintel.pth"
    args.alternate_corr = False
    args.flow_type = "DIS" #"FB"#"TBL"#"DIS" #"Farneback"

    ############################################################################### 配置

    args.save_flow = True
    args.grid_size = 32
    args.random_shift = True
    args.even_sample = True
    ############################################################################### 

    args.object_num = 3  # 
    args.dataset_dir = 'datasets/Neural3D/sear_steak'
    seq_list = ["sear_steak"]

    for seq in seq_list:
        args.seq = seq
        args.root_path = os.path.join(args.dataset_dir, seq)
        args.common_param = get_common_param(args.dataset_dir, seq, is_sport=False)
        os.makedirs(os.path.join(args.root_path, 'flows'), exist_ok=True)  
        args.result_dict_path = os.path.join(args.root_path, 'flows', 
                                            f"model_{args.flow_type}_radius_{args.radius}.pkl")
        demo(args)
