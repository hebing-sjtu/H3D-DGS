import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import torch
import copy
from tqdm import tqdm
from time import time 
from utils.data_preprocess import get_common_param, get_dataset, get_batch
from utils.data_reader import load_ply, get_flow
from utils.data_saver import save_res_params, save_render_ply, save_gs_cate, save_ctrl_idx
from utils.format_converter import params2cpu, vars2cpu, render_one_frame_in_train
from utils.gs_optimizer import initialize_params, initialize_optimizer, initialize_post_timestep, \
    residual_init_per_timestep, residual_init_opt, update_learning_rate, report_progress, densify, get_loss, get_expon_lr_func
from utils.gs_classifier import gs_cate
from utils.CTRL_pts import ctrlvar_generator, ctrl_select, ctrl_prune, get_id_wgt, volume_warping


def train(dataset_dir:str, seq:str, exp:str, cuda_id:int=0, ds_ratio:int=2, 
          gc_knn_num:int=3, cc_knn_num:int=3, iter0:int=10000, iter1:int=500, iter2:int=100, 
          checkpoint:bool=True, res_update_freq:int=1, prune:bool=False, prune_ratio:float=0.3, radius:int=8,
          is_sport:bool=False, only_objs:bool=False, transfromrot:bool=False, depth_flag:bool=False, no_render:bool=False):
    """Main training loop for the model.
    Args:
        dataset_dir (str): Path to the dataset directory.
        seq (str): Sequence name.
        exp (str): Experiment name.
        cuda_id (int, optional): CUDA device ID. Defaults to 0.
        ds_ratio (int, optional): Downsampling ratio for images. Defaults to 2.
        gc_knn_num (int, optional): Gs-Ctrl KNN number. Defaults to 3.
        cc_knn_num (int, optional): Ctrl-Ctrl KNN number. Defaults to 3.
        iter0 (int, optional): Number of iterations for the initial timestep. Defaults to 10000.
        iter1 (int, optional): Number of iterations for key timesteps. Defaults to 500.
        iter2 (int, optional): Number of iterations for non-key timesteps. Defaults to 100.
        checkpoint (bool, optional): Whether to restore from checkpoint. Defaults to True.
        res_update_freq (int, optional): Frequency of residual update. Defaults to 1.
        prune (bool, optional): Whether to prune control points or not. Defaults to False.
        prune_ratio (float, optional): Pruning ratio for control points. Defaults to 0.3.
        radius (int, optional): Radius for optical flow. Defaults to 8.
        is_sport (bool, optional): Whether the dataset is CMU-panoptic. Defaults to False.
        only_objs (bool, optional): Whether to only optimize objects. Defaults to False.
    """
    
    ### load dataset and initialize parameters
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(f'{current_dir}/output/{exp}/{seq}'):
        print(f'Experiment {exp} for sequence {seq} already exists.')
        
    num_timesteps = len(os.listdir(f'{dataset_dir}/{seq}/imgs/0'))
    common_param = get_common_param(dataset_dir, seq, ds_ratio, cuda_id, is_sport=is_sport, sparse_mask=False, sparse_flow=False, sparse_mask_id=None, sparse_flow_id=None, sparse_mask_num=6, sparse_flow_num=6)    
    if checkpoint:
        init_t, variables = load_ply(common_param, exp, seq, 0)
        if init_t >= num_timesteps:
            print(f'Experiment {exp} for sequence {seq} already completed.')
            return
    else:
        init_t = 0
        params, variables = initialize_params(common_param)
        optimizer = initialize_optimizer(params, variables)

    ### load preprocessed optical flow
    flowsss = get_flow(dataset_dir,seq,radius)
    
    key_count = 0
    non_key_count = 0
    key_time = 0.0
    non_key_time = 0.0
    total_time = 0.0
    
    ### main training loop
    for t in range(init_t, num_timesteps):
        train_dataset = get_dataset(common_param,t,train_flag=True,depth_flag=depth_flag)
        eval_dataset = get_dataset(common_param,t,train_flag=False)
        todo_dataset = []
        is_initial_timestep = (t == 0)
        only_ctrl = (t%res_update_freq != 0)
        
        ### generate H3D-control pts variation using optical flow
        if not is_initial_timestep:
            flowss = flowsss[str(t-1)]
            flowss_copy = copy.deepcopy(flowss)          
            # remove optical flow predicted using evaluation cameras
            # for i in range(common_param.obj_num):
            #     flowss_copy[i].pop(0)
            ctrlvar_generator(common_param, variables, flowss_copy, depths_for_H3D)
            idx1 = ctrl_select(common_param, variables, train_dataset)
            # save_ctrl_idx(idx1, exp, seq, t)
            if prune:
                idx2 = ctrl_prune(common_param, variables, ctrl_ratio=prune_ratio)
                save_ctrl_idx(idx1[idx2], exp, seq, t)
            else:
                save_ctrl_idx(idx1, exp, seq, t)
                        
            gs_cate(common_param, variables, train_dataset)
            get_id_wgt(common_param, variables, gc_knn_num, cc_knn_num)
            
            params = residual_init_per_timestep(common_param, variables, only_ctrl=only_ctrl, only_objs=only_objs)
            optimizer = residual_init_opt(params, only_ctrl=only_ctrl)
        
        ### determine number of iterations for current timestep
        if is_initial_timestep:
            num_iter_per_timestep = iter0
        elif not only_ctrl:
            num_iter_per_timestep = iter1
            key_count += 1
        else:
            num_iter_per_timestep = iter2
            non_key_count += 1
        
        progress_bar = tqdm(range(num_iter_per_timestep), desc=f'timestep {t}')

        ### learning rate schedulers
        xyz_scheduler_args = get_expon_lr_func(lr_init=0.001,
                    lr_final=0.00001,
                    lr_delay_mult=0.01,
                    max_steps=num_iter_per_timestep)
        
        col_scheduler_args = get_expon_lr_func(lr_init=0.0025,
                    lr_final=0.000025,
                    lr_delay_mult=0.01,
                    max_steps=num_iter_per_timestep)
        
        opa_scheduler_args = get_expon_lr_func(lr_init=0.05,
                    lr_final=0.0005,
                    lr_delay_mult=0.01,
                    max_steps=num_iter_per_timestep)
        step_start_time = time()
        
        ### optimization loop for current timestep
        for i in range(num_iter_per_timestep):
            todo_dataset, curr_data = get_batch(common_param, todo_dataset, train_dataset, depth_flag=depth_flag)
            loss = get_loss(params, curr_data, variables, is_initial_timestep, common_param, only_ctrl=only_ctrl, only_objs=only_objs, transfromrot=transfromrot, depth_flag=depth_flag)
            loss.backward()
            with torch.no_grad():
                if i % 100 == 0:
                    # report_progress(params, eval_dataset, progress_bar, is_initial_timestep, common_param, variables, only_ctrl=only_ctrl, only_objs=only_objs, transfromrot=transfromrot)
                    progress_bar.update(100)
                if is_initial_timestep:
                    params, variables = densify(params, variables, optimizer, i)
                optimizer.step()
                if not todo_dataset:
                    update_learning_rate(optimizer, xyz_scheduler_args, col_scheduler_args, opa_scheduler_args, i)
                optimizer.zero_grad(set_to_none=True)
        step_time = time() - step_start_time
        if not is_initial_timestep:
            if not only_ctrl:
                key_time += step_time
            else:
                non_key_time += step_time
        progress_bar.close()
        
        
        ### post-timestep initialization and saving results
        if is_initial_timestep:
            initialize_post_timestep(variables, params)
        else:
            render_params = volume_warping(common_param, variables, params, only_ctrl=only_ctrl, only_objs=only_objs)
            initialize_post_timestep(variables, render_params)
        
        ### render current timestep
        if not no_render:
            output_dir = f'./output/{exp}/{seq}/render_result'
            depths_for_H3D = render_one_frame_in_train(common_param, render_params, output_dir, t=t)
        else:
            depths_for_H3D = None
        
        ### save results
        res = params2cpu(params)
        render = vars2cpu(variables)
        save_res_params(res, exp, seq, t)
        save_render_ply(render, exp, seq, t)
        save_gs_cate(render, exp, seq, t)
        total_time += step_time
    
    print("total time for training: ", total_time)
    print("average key time for training: ", key_time/key_count)
    print("average non-key time for training: ", non_key_time/non_key_count)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the model with given parameters")
    parser.add_argument('--dataset_dir', type=str, default='/data/ssd/4Ddatasets/Neural3D', help='Dataset directory')
    parser.add_argument('--sequence', type=str, default='cook_spinach_abl', help='Sequence name')
    parser.add_argument('--ds_ratio', type=int, default=2, help='Dataset ratio')
    parser.add_argument('--cuda_id', type=int, default=0, help='CUDA ID')
    parser.add_argument('--ckpt', action='store_true', default=False, help='restore from ckpt')
    parser.add_argument('--is_sport', action='store_true', default=False, help='whether the dataset is sport')

    parser.add_argument('--res_update_freq', type=int, default=10, help='Results update frequency')
    parser.add_argument('--gc_knn_num', type=int, default=3, help='Gs-Ctrl KNN number')
    parser.add_argument('--cc_knn_num', type=int, default=3, help='Ctrl-Ctrl KNN number')
    parser.add_argument('--iter0', type=int, default=10000, help='Iteration 0')
    parser.add_argument('--iter1', type=int, default=500, help='Iteration 1')
    parser.add_argument('--iter2', type=int, default=100, help='Iteration 2')
    parser.add_argument('--prune_ratio', type=float, default=0.1, help='Prune ratio')
    parser.add_argument('--flow_radius', type=int, default=2, help='Prune ratio')
    
    parser.add_argument('--prune', action='store_true', default=False, help='Prune')
    parser.add_argument('--transfromrot', action='store_true', default=False, help='Macro rotation')
    parser.add_argument('--depth_flag', action='store_true', default=False, help='Depth flag')
    parser.add_argument('--only_objs', action='store_true', default=False, help='Depth flag')
    
    args = parser.parse_args()
    exp_name = f'GOS_{args.res_update_freq}_radius_{args.flow_radius}_gc_knn_{args.gc_knn_num}'
    # exp_name = f'GOS_{args.res_update_freq}_{args.iter1}iter1_{args.iter2}iter2_ctrl_prune_{args.prune}_{args.prune_ratio}'
    # exp_name = f'flowview_20_resper_{args.res_update_freq}_FPstride_32_16_{args.iter1}iter1_{args.iter2}iter2_ctrlselect_woc_prune03'
    train(args.dataset_dir, args.sequence, exp_name, args.cuda_id, args.ds_ratio, args.gc_knn_num, args.cc_knn_num, 
            checkpoint=args.ckpt, res_update_freq=args.res_update_freq, prune=args.prune, prune_ratio=args.prune_ratio, radius=args.flow_radius,
            iter0=args.iter0, iter1=args.iter1, iter2=args.iter2, is_sport=args.is_sport, only_objs=args.only_objs, 
            transfromrot=args.transfromrot, depth_flag=args.depth_flag)
    torch.cuda.empty_cache()