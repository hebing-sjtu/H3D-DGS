import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from utils.data_reader import load_ply
from utils.format_converter import variables2rendervar, params2rendervar
from train import get_common_param

def render(dataset_dir, seq, exp, output_dir, is_sport=False, cuda_id=0, ds_ratio=2, num_timesteps = 300, depth_flag=False, only_depth_flag=True):
    
    if num_timesteps == 300:
        num_timesteps = len(os.listdir(f'{dataset_dir}/{seq}/imgs/0'))
    common_param = get_common_param(dataset_dir, seq, ds_ratio, cuda_id, is_sport=is_sport, sparse_mask=False, sparse_flow=False, sparse_mask_id=None, sparse_flow_id=None, sparse_mask_num=6, sparse_flow_num=6)
    test_cam_id = common_param.test_cam_id

    for t in tqdm(range(num_timesteps), desc=f"Render progress for {exp} / {seq}"):
        with torch.no_grad():
            _, variables = load_ply(common_param, exp, seq, t)
            rendervar = variables2rendervar(variables)
            for cam_id in test_cam_id:
                cam = common_param.cams[cam_id]
                im, _, depth, = Renderer(raster_settings=cam)(**rendervar)
                if not (depth_flag and only_depth_flag):
                    img_outpath = f'{output_dir}/{cam_id}/img'
                    os.makedirs(img_outpath, exist_ok=True)
                    im = im.cpu().numpy()
                    im = np.transpose(im, (1, 2, 0))
                    im = np.clip(im, 0, 1)
                    # print(im.max())
                    plt.imsave(f"{img_outpath}/{str(t).zfill(4)}.png", im)
                if depth_flag:
                    depth_outpath = f'{output_dir}/{cam_id}/depth'
                    os.makedirs(depth_outpath, exist_ok=True)
                    # print(depth.max(), depth.min(), depth.mean(), depth.std())
                    depth = (depth.squeeze().cpu().numpy() + 8.)/ 16.0
                    depth = np.clip(depth, 0, 1)
                    plt.imsave(f"{depth_outpath}/{str(t).zfill(4)}.png", depth, cmap='gray')
                # break
    
def render_one_frame(dataset_dir, seq, exp, output_dir, cuda_id=0, ds_ratio=2, t=0):
    common_param = get_common_param(dataset_dir, seq, ds_ratio, cuda_id, is_sport=False, sparse_mask=False, sparse_flow=False, sparse_mask_id=None, sparse_flow_id=None, sparse_mask_num=6, sparse_flow_num=6)
    cam = common_param.cams[0]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with torch.no_grad():
        _, variables = load_ply(common_param, exp, seq, t)
        rendervar = variables2rendervar(variables)
        im, _, depth, = Renderer(raster_settings=cam)(**rendervar)
        im = im.cpu().numpy()
        im = np.transpose(im, (1, 2, 0))
        im = np.clip(im, 0, 1)
        # depth = (depth.squeeze().cpu().numpy() - 5.0)/ 20.0
        # depth = np.clip(depth, 0, 1)
        plt.imsave(f"{output_dir}/{str(t).zfill(4)}.png", im)
        # plt.imsave(f"{output_dir}/{str(t).zfill(4)}_depth.png", depth, cmap='gray')



res_update_freq = 5
gc_knn_num=3
cc_knn_num=3
iter0=10000
iter1=500
iter2=100
dataset_dir = '/data/ssd/4Ddatasets/S4D-CMU'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the model with given parameters")
    parser.add_argument('--dataset_dir', type=str, default='/data/ssd/4Ddatasets/S4D-Neural3D', help='Dataset directory')
    parser.add_argument('--sequence', type=str, default='cook_spinach_abl', help='Sequence name')
    parser.add_argument('--ds_ratio', type=int, default=2, help='Dataset ratio')
    parser.add_argument('--cuda_id', type=int, default=0, help='CUDA ID')
    parser.add_argument('--is_sport', action='store_true', default=False, help='whether the dataset is sport')

    parser.add_argument('--res_update_freq', type=int, default=5, help='Results update frequency')
    parser.add_argument('--iter1', type=int, default=500, help='Iteration 1')
    parser.add_argument('--iter2', type=int, default=100, help='Iteration 2')
    parser.add_argument('--prune_ratio', type=int, default=0.1, help='Prune ratio')
    parser.add_argument('--gc_knn_num', type=int, default=3, help='Gs-Ctrl KNN number')
    parser.add_argument('--flow_radius', type=int, default=8, help='Prune ratio')
    
    parser.add_argument('--prune', action='store_true', default=False, help='Prune')
    args = parser.parse_args()
    
    exp_name = f'GOS_{args.res_update_freq}_radius_{args.flow_radius}_gc_knn_{args.gc_knn_num}'
    # exp_name = f'GOS_{args.res_update_freq}_{args.iter1}iter1_{args.iter2}iter2_ctrl_prune_{args.prune}_{args.prune_ratio}'
    
    # seq_list = ['basketball', 'softball', 'boxes']
    # for seq in seq_list:
    # args.sequence = seq
    args.is_sport = False
    output_dir = f'./output/{exp_name}/{args.sequence}/render_result'
    render(args.dataset_dir, args.sequence, exp_name, output_dir, 
        is_sport=args.is_sport, cuda_id=args.cuda_id, ds_ratio=args.ds_ratio)

# exp = f'GOS_{res_update_freq}_{iter1}iter1_{iter2}iter2_ctrl_prune_False_0.1'   
# seq_list = ['coffee_martini', 'cook_spinach_abl', 'cut_roasted_beef', 'flame_salmon_1', 'flame_steak', 'sear_steak_abl']
# seq_list = ['basketball', 'softball', 'boxes']

# t_list = [120,120,120,300,120,120]
# for seq, t in zip(seq_list, t_list):
#     output_dir = f'./output/{exp}/{seq}/imgs'
#     render_one_frame(dataset_dir, seq, exp, output_dir, cuda_id=0, ds_ratio=2, t=t)