import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import copy
import numpy as np
from tqdm import tqdm
from lpipsPyTorch import lpips
from pytorch_msssim import ms_ssim
from torchvision.utils import save_image
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from utils.loss_function import calc_psnr, calc_ssim, calc_mse
from utils.format_converter import variables2rendervar, params2vars
from utils.data_reader import load_ply, load_res, load_ctrl_idx, get_flow
from utils.data_preprocess import get_common_param, get_dataset, get_batch, Common_Param
from utils.CTRL_pts import ctrlvar_generator, get_id_wgt, volume_warping
from utils.gs_classifier import gs_cate


def get_result(eval_data, variables=None):
    rendervar = variables2rendervar(variables)
    im, _, _, = Renderer(raster_settings=eval_data['cam'])(**rendervar)
    
    ssim1 = calc_ssim(im, eval_data['im'])
    psnr = calc_psnr(im, eval_data['im']).mean()
    lpips1 = lpips(im.unsqueeze(0), eval_data['im'].unsqueeze(0), net_type='vgg')
    msssim = ms_ssim(im.unsqueeze(0), eval_data['im'].unsqueeze(0), data_range=1, size_average=True )
    lpips2 = lpips(im.unsqueeze(0), eval_data['im'].unsqueeze(0), net_type='alex')
    dssim = (1 - msssim)/2
    return im,ssim1,psnr,lpips1,msssim,lpips2,dssim


def easy_eval(dataset_dir, seq, exp, cuda_id=0, ds_ratio=2, num_timesteps = 300, is_sport=False, only_init=False):
    
    if only_init:
        num_timesteps = 1
    elif num_timesteps == 300:
        num_timesteps = len(os.listdir(f'{dataset_dir}/{seq}/imgs/0'))
    common_param = get_common_param(dataset_dir, seq, ds_ratio, cuda_id, is_sport=is_sport, sparse_mask=False, sparse_flow=False, sparse_mask_id=None, sparse_flow_id=None, sparse_mask_num=6, sparse_flow_num=6)
    # flowsss = get_flow_v3(seq)
    ssim_list = torch.zeros((num_timesteps))
    psnr_list = torch.zeros((num_timesteps))
    lpips1_list = torch.zeros((num_timesteps))
    msssim_list = torch.zeros((num_timesteps))
    lpips2_list = torch.zeros((num_timesteps))
    dssim_list = torch.zeros((num_timesteps))
    for t in tqdm(range(num_timesteps), desc=f"Metric evaluation progress for {exp} / {seq}"):
        # only_ctrl = (t%res_update_freq != 0)
        with torch.no_grad():
            todo_eval_dataset = []
            eval_dataset = get_dataset(common_param,t,train_flag=False)
            eval_data_num = len(eval_dataset)
            _, variables4 = load_ply(common_param, exp, seq, t)
            ssim_t = torch.zeros((eval_data_num))
            psnr_t = torch.zeros((eval_data_num))
            lpips1_t = torch.zeros((eval_data_num))
            msssim_t = torch.zeros((eval_data_num))
            lpips2_t = torch.zeros((eval_data_num))
            dssim_t = torch.zeros((eval_data_num))
            for i in range(eval_data_num):
                todo_eval_dataset, eval_data = get_batch(common_param, todo_eval_dataset, eval_dataset, shuffle=False)
                im,ssim1,psnr,lpips1,msssim,lpips2,dssim = get_result(eval_data, variables4)
                im = im.cpu()
                os.makedirs(f'./output/{exp}/{seq}/{i}', exist_ok=True)
                save_image(im, f'./output/{exp}/{seq}/{i}/{str(t).zfill(4)}.png')
                ssim_t[i] = ssim1
                psnr_t[i] = psnr
                lpips1_t[i] = lpips1
                msssim_t[i] = msssim
                lpips2_t[i] = lpips2
                dssim_t[i] = dssim
            ssim_list[t] = ssim_t.mean()
            psnr_list[t] = psnr_t.mean()
            lpips1_list[t] = lpips1_t.mean()
            msssim_list[t] = msssim_t.mean()
            lpips2_list[t] = lpips2_t.mean()
            dssim_list[t] = dssim_t.mean()

    ssim_mean = ssim_list.mean()
    psnr_mean = psnr_list.mean()
    lpips1_mean = lpips1_list.mean()
    msssim_mean = msssim_list.mean()
    lpips2_mean = lpips2_list.mean()
    dssim_mean = dssim_list.mean()
    if not only_init:
        torch.save(ssim_list, f'./output/{exp}/{seq}/ssim_list.pth')
        torch.save(psnr_list, f'./output/{exp}/{seq}/psnr_list.pth')
        torch.save(lpips1_list, f'./output/{exp}/{seq}/lpips1_list.pth')
        torch.save(msssim_list, f'./output/{exp}/{seq}/msssim_list.pth')
        torch.save(lpips2_list, f'./output/{exp}/{seq}/lpips2_list.pth')
        torch.save(dssim_list, f'./output/{exp}/{seq}/dssim_list.pth')

    return ssim_mean, psnr_mean, lpips1_mean, msssim_mean, lpips2_mean, dssim_mean


def get_result_v2(eval_dataset, variables, common_param:Common_Param, save_flag=False, mse_folder='none', t_idx=0):
    rendervar = variables2rendervar(variables)
    todo_eval_dataset = []
    psnrs = []
    ms_ssims = []
    for i in range(len(common_param.test_cam_id)):
        eval_data = get_batch(common_param, todo_eval_dataset, eval_dataset)
        im, _, _, = Renderer(raster_settings=eval_data['cam'])(**rendervar)
        psnr = calc_psnr(im, eval_data['im']).mean()
        msssim = ms_ssim(im.unsqueeze(0), eval_data['im'].unsqueeze(0), data_range=1, size_average=True)
        psnrs.append(psnr)
        ms_ssims.append(msssim)
        if save_flag:
            mse = calc_mse(im, eval_data['im']).cpu()
            print(f'{mse_folder} mse: {mse.max()} suggest scale: {1/mse.max()}')
            mse = mse*3.0
            # mse = mse / mse.max()
            if not os.path.exists(f'./mseimg/{mse_folder}'):
                os.makedirs(f'./mseimg/{mse_folder}')
            save_image(mse, f'./mseimg/{mse_folder}/{t_idx}.png')
    psnr_mean = torch.stack(psnrs).mean()
    ssim_mean = torch.stack(ms_ssims).mean()
    
    return psnr_mean,ssim_mean


def select_ctrl_for_visualization(variables,idx):
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


def evaluate(dataset_dir, seq, exp, cuda_id=0, ds_ratio=2, res_update_freq=10, num_timesteps = 300):

    is_sport = os.path.basename(dataset_dir) != 'S4D-Neural3D'
    if num_timesteps == 300:
        num_timesteps = len(os.listdir(f'{dataset_dir}/{seq}/imgs/0'))
    common_param = get_common_param(dataset_dir, seq, ds_ratio, cuda_id, is_sport=is_sport, sparse_mask=False, sparse_flow=False, sparse_mask_id=None, sparse_flow_id=None, sparse_mask_num=6, sparse_flow_num=6)
    flowsss = get_flow(dataset_dir,seq)
    psnr_list = torch.zeros((3, num_timesteps))
    ssim_list = torch.zeros((3, num_timesteps))
    dssim_list = torch.zeros((3, num_timesteps))
    for t in tqdm(range(num_timesteps), desc=f"Metric evaluation progress for {exp} / {seq}"):
        only_ctrl = (t%res_update_freq != 0)
        save_flag = (t%res_update_freq == 9)
        with torch.no_grad():
            train_dataset = get_dataset(common_param,t,train_flag=True)
            eval_dataset = get_dataset(common_param,t,train_flag=False)
            
            if (t == 0) or not only_ctrl:
                _, variables1 = load_ply(common_param, exp, seq, t)
                variables2 = variables1
                variables3 = variables1
            else:
                _, params = load_res(common_param, exp, seq, t, only_ctrl=True)
                flowss = flowsss[str(t-1)]
                flowss_copy = copy.deepcopy(flowss)       
                idx = load_ctrl_idx(exp, seq, t)       
                for i in range(common_param.obj_num):
                    flowss_copy[i].pop(0)
                    
                ctrlvar_generator(common_param, variables2, flowss_copy)
                select_ctrl_for_visualization(variables2,idx)
                gs_cate(common_param, variables2, train_dataset)
                get_id_wgt(common_param, variables2, 3, 3)
                variables2 = params2vars(volume_warping(common_param, variables2, None, only_ctrl=True))

                ctrlvar_generator(common_param, variables3, flowss_copy)
                select_ctrl_for_visualization(variables3,idx)
                gs_cate(common_param, variables3, train_dataset)
                get_id_wgt(common_param, variables3, 3, 3)
                variables3 = params2vars(volume_warping(common_param, variables3, params, only_ctrl=True))

            psnr1, ssim1 = get_result_v2(eval_dataset, variables1, common_param, save_flag=save_flag,mse_folder='none', t_idx=t)
            psnr2, ssim2 = get_result_v2(eval_dataset, variables2, common_param, save_flag=save_flag,mse_folder='partctrl', t_idx=t)
            psnr3, ssim3 = get_result_v2(eval_dataset, variables3, common_param, save_flag=save_flag, mse_folder='fullctrl', t_idx=t)
            psnr_list[:,t] = torch.tensor([psnr1, psnr2, psnr3])
            ssim_list[:,t] = torch.tensor([ssim1, ssim2, ssim3])
            dssim_list[:,t] = torch.tensor([(1-ssim1)/2, (1-ssim2)/2, (1-ssim3)/2])

    np.savez('results.npz', psnr=psnr_list.cpu().numpy(), ssim=ssim_list.cpu().numpy(), dssim=dssim_list.cpu().numpy())
    psnr_mean = [psnr_list[i].mean() for i in range(3)]
    ssim_mean = [ssim_list[i].mean() for i in range(3)]
    dssim_mean = [dssim_list[i].mean() for i in range(3)]
    return psnr_mean, ssim_mean, dssim_mean

import argparse

def main(args):
    exp_name = f'GOS_{args.res_update_freq}_radius_{args.flow_radius}_gc_knn_{args.gc_knn_num}'
    # exp_name = f'GOS_{args.res_update_freq}_{args.iter1}iter1_{args.iter2}iter2_ctrl_prune_{args.prune}_{args.prune_ratio}'
    sequence = args.sequence
    dataset_dir = args.dataset_dir
    
    ssim_mean, psnr_mean, lpips1_mean, msssim_mean, lpips2_mean, dssim_mean = easy_eval(dataset_dir, sequence, exp_name, ds_ratio=args.ds_ratio, num_timesteps=args.num_timesteps, is_sport=args.is_sport, only_init=args.only_init)
    print(f'ssim: {ssim_mean}, psnr: {psnr_mean}, lpips_vgg: {lpips1_mean}, msssim: {msssim_mean}, lpips2_alex: {lpips2_mean}, dssim: {dssim_mean}')
    # evaluation for parameters in control points 
    # psnr, ssim, dssim = evaluate(dataset_dir, sequence, exp_name, num_timesteps=30)
    # nm_list = ['none','partctrl','allctrl']
    # for i in range(3):
    #     print(f'{nm_list[i]} psnr: {psnr[i]}, ssim: {ssim[i]}, dssim: {dssim[i]}')
      
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate the model")
    parser.add_argument('--dataset_dir', type=str, default='/data/ssd/4Ddatasets/S4D-Neural3D', help='Dataset directory')
    parser.add_argument('--sequence', type=str, default='cook_spinach_abl', help='Sequence name')
    parser.add_argument('--ds_ratio', type=int, default=2, help='Dataset ratio')
    parser.add_argument('--res_update_freq', type=int, default=5, help='Residual frame frequency')
    parser.add_argument('--iter0', type=int, default=10000, help='Iteration 0')
    parser.add_argument('--iter1', type=int, default=500, help='Iteration 1')
    parser.add_argument('--iter2', type=int, default=100, help='Iteration 2')
    parser.add_argument('--num_timesteps', type=int, default=9, help='Number of timesteps')
    parser.add_argument('--is_sport', action='store_true', default=False, help='whether the dataset is sport')
    parser.add_argument('--prune', action='store_true', default=False, help='Prune')
    parser.add_argument('--prune_ratio', type=int, default=0.1, help='Prune ratio')
    parser.add_argument('--gc_knn_num', type=int, default=3, help='Gs-Ctrl KNN number')
    parser.add_argument('--flow_radius', type=int, default=8, help='Prune ratio')
    
    parser.add_argument('--only_init', action='store_true', default=False, help='Eval only initial frame')
    
    args = parser.parse_args()
    main(args)
  