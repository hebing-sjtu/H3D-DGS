import numpy as np
import torch
import os
import json
import glob
import copy
import torch.nn.functional as F
import open3d as o3d

from PIL import Image
from random import randint
from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera
from typing import NamedTuple, Tuple


class Common_Param(NamedTuple):
    """
    A unified parameter structure for a multi-camera system.

    This structure stores all camera-related parameters that are commonly used 
    throughout training, evaluation, and rendering. It includes:

    - **Intrinsic and extrinsic parameters**:
        - k / inv_k: camera intrinsic and its inverse.
        - w2c / c2w: world-to-camera and camera-to-world transformation matrices.
        - proj / inv_proj: projection and inverse-projection matrices.
        - campos, cx, cy, h, w: camera positions and image geometry.

    - **System-level attributes**:
        - ds_ratio: downsampling ratio.
        - obj_num, cam_num: number of tracked objects and cameras.

    - **Dataset partitioning**:
        - dataset_train_cam_id, dataset_test_cam_id, dataset_edge_cam_id: 
          camera folder names for training, testing, and edge evaluation.
        - train_cam_id, test_cam_id, edge_cam_id: camera indices in the list.

    - **Additional information**:
        - mask_id, flow_id: mask and optical flow index mappings.
        - dataset_dir, seq: dataset root directory and sequence name.
        - device: computation device (e.g., "cuda:0", "cpu").
        - is_sport: whether the sequence belongs to CMU-panoptic dataset.

    This structure unifies all per-camera and global configurations into a 
    single accessible object, simplifying cross-module consistency.
    """
    k: torch.Tensor
    inv_k: torch.Tensor
    w2c: torch.Tensor
    c2w: torch.Tensor
    proj: torch.Tensor
    inv_proj: torch.Tensor
    campos: torch.Tensor
    cx: torch.Tensor
    cy: torch.Tensor
    h: int
    w: int
    ds_ratio: int
    obj_num: int
    cam_num: int
    cams: list
    dataset_train_cam_id: list
    dataset_test_cam_id: list
    dataset_edge_cam_id: list
    train_cam_id: list
    test_cam_id: list
    edge_cam_id: list
    mask_id: list
    flow_id: list
    dataset_dir: str
    seq: str
    device: str
    is_sport: bool


def get_common_param(dataset_dir:str, seq:str, 
                     ds_ratio:int=1, cuda_id:int=0, is_sport:bool=False, 
                     sparse_mask:bool=False, sparse_flow:bool=False, 
                     sparse_mask_id=None, sparse_flow_id=None, 
                     sparse_mask_num:int=6, sparse_flow_num:int=6)->Common_Param:    
    """
    Constructs a Common_Param object encapsulating camera and dataset parameters.
    Args:
        dataset_dir (str): Root directory of the dataset.
        seq (str): Sequence name within the dataset.
        ds_ratio (int, optional): Downsampling ratio for images. Defaults to 1.
        cuda_id (int, optional): CUDA device ID for computations. Defaults to 0.
        is_sport (bool, optional): Whether the sequence is from CMU-panoptic dataset. Defaults to False.
        sparse_mask (bool, optional): Whether to use sparse mask cameras. Defaults to False.
        sparse_flow (bool, optional): Whether to use sparse flow cameras. Defaults to False.
        sparse_mask_id (list, optional): Specific camera IDs for sparse masks. Defaults to None.
        sparse_flow_id (list, optional): Specific camera IDs for sparse flows. Defaults to None.
        sparse_mask_num (int, optional): Number of sparse mask cameras if not specified. Defaults to 6.
        sparse_flow_num (int, optional): Number of sparse flow cameras if not specified. Defaults to 6.
    Returns:
        Common_Param: An object containing all relevant camera and dataset parameters.
    """
    torch.cuda.set_device(cuda_id)
    cams = []
    w2cs = []
    c2ws = []
    ks = []
    inv_ks = []
    projs = []
    inv_projs = []
    campos = []
    cx = []
    cy = []
    dataset_train_cam_id = []
    dataset_test_cam_id = []
    dataset_edge_cam_id = []
    train_cam_id = []
    test_cam_id = []
    edge_cam_id = []
    mask_id = []
    flow_id = []
    side_padding = torch.zeros(3,1).float()
    bottom_tensor = torch.tensor([[0, 0, 0, 1]], dtype=torch.float32)
    bottom = np.array([[0, 0, 0, 1]], dtype=np.float32)
    
    # for CMU-panoptic dataset
    if is_sport:
        md_train = json.load(open(f"{dataset_dir}/{seq}/train_meta.json", 'r'))
        md_test = json.load(open(f"{dataset_dir}/{seq}/test_meta.json", 'r'))
        
        H = int(md_train['h']/ds_ratio)
        W = int(md_train['w']/ds_ratio)
        
        for c in range(len(md_train['fn'][0])):
            nm = md_train['fn'][0][c]
            dataset_train_cam_id.append(int(nm.split("/")[0]))
            
        for c in range(len(md_test['fn'][0])):
            nm = md_test['fn'][0][c]
            dataset_test_cam_id.append(int(nm.split("/")[0]))
            
        merged_list = dataset_train_cam_id + dataset_test_cam_id
        sorted_list = sorted(merged_list)
        
        for i,c in enumerate(sorted_list):
            if c in dataset_train_cam_id:
                train_cam_id.append(i)
            else:
                test_cam_id.append(i)
                
    # for neural 3D dataset
    else:
        poses_arr = np.load(f'{dataset_dir}/{seq}/poses_bounds.npy').astype(float)
        poses = poses_arr[:, :-2].reshape([-1, 3, 5])
        H, W, focal = poses[0, :, -1]
        H = int(H / ds_ratio)
        W = int(W / ds_ratio)
        focal = focal / ds_ratio
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
        cam_list = os.listdir(os.path.join(dataset_dir, seq, 'imgs'))
        cam_list = sorted([int(cam) for cam in cam_list])
        # print(cam_list)

        for c_idx, c in enumerate(cam_list):
            if c_idx == 0:
                dataset_test_cam_id.append(c)
                test_cam_id.append(c_idx)
            else:
                dataset_train_cam_id.append(c)
                train_cam_id.append(c_idx)
            if c_idx in [1,10,11,20]:
                dataset_edge_cam_id.append(c)
                edge_cam_id.append(c_idx)
        # print('dataset_test_cam_id',dataset_test_cam_id)
        # print('test_cam_id',test_cam_id)
        # print('dataset_train_cam_id',dataset_train_cam_id)
        # print('train_cam_id',train_cam_id)
                
    total_cam_num = len(train_cam_id) + len(test_cam_id)
    
    for c in range(total_cam_num):
        if is_sport:
            if c in dataset_train_cam_id:
                cam_id = dataset_train_cam_id.index(c)
                k, w2c = md_train['k'][0][cam_id], md_train['w2c'][0][cam_id]
            else:
                cam_id = test_cam_id.index(c)
                k, w2c = md_test['k'][0][cam_id], md_test['w2c'][0][cam_id]
            k = torch.tensor(k)/ds_ratio
            w2c = torch.tensor(w2c)
            c2w = torch.inverse(w2c)
            cam_center = c2w[:3, 3]
            cam = setup_camera(W, H, k, w2c, cam_center, near=1.0, far=100)
        else:
            pose = poses[c]
            R = pose[:3,:3]
            R = -R
            R[:,0] = -R[:,0]
            T = -pose[:3,3].dot(R)
            k = torch.tensor([[focal, 0.0, W / 2.0], [0, focal, H / 2.0], [0.0, 0.0, 1.0]])
            w2c = np.concatenate([np.concatenate([R.T, T[..., None]], -1), bottom], 0)
            w2c = torch.tensor(w2c)
            c2w = torch.inverse(w2c)
            cam_center = c2w[:3, 3]
            cam = setup_camera_v2(W, H, focal, w2c, cam_center)
        cx.append(k[0,2])
        cy.append(k[1,2])
        inv_k = torch.inverse(k)
        inv_ks.append(inv_k)
        ks.append(k)
        w2cs.append(w2c)
        c2ws.append(c2w)
        projs.append(torch.matmul(torch.concatenate((k,side_padding),dim=-1),w2c))
        inv_projs.append(torch.matmul(c2w, torch.inverse(torch.cat((torch.cat((inv_k,side_padding),dim=-1),bottom_tensor),dim=0))))
        campos.append(c2w[:3, 3])
        cams.append(cam)
    mask_dirs = glob.glob(os.path.join(dataset_dir, seq, 'mask_0*'))

    if sparse_mask:
        if sparse_mask_id is not None:
            mask_id = sparse_mask_id
        else:
            mask_id = select_cams(torch.stack(campos),train_cam_id,sparse_mask_num)
    else:
        mask_id = train_cam_id
        
    if sparse_flow:    
        if sparse_flow_id is not None:    
            flow_id = sparse_flow_id
        else:
            flow_id = select_cams(torch.stack(campos),train_cam_id,sparse_flow_num)
    else:
        flow_id = train_cam_id

    common_param = Common_Param(
        k=torch.stack(ks).float().cuda(),
        inv_k=torch.stack(inv_ks).float().cuda(),
        w2c=torch.stack(w2cs).float().cuda(),
        c2w=torch.stack(c2ws).float().cuda(),
        proj=torch.stack(projs).float().cuda(),
        inv_proj=torch.stack(inv_projs).float().cuda(),
        campos=torch.stack(campos).float().cuda(),
        cx=torch.stack(cx).float().cuda(),
        cy=torch.stack(cy).float().cuda(),
        h=H,
        w=W,
        ds_ratio=ds_ratio,
        cam_num=len(cams),
        obj_num=len(mask_dirs)-1,
        cams=cams,
        cam_list=cam_list,
        dataset_train_cam_id=dataset_train_cam_id,
        dataset_test_cam_id=dataset_test_cam_id,
        dataset_edge_cam_id=dataset_edge_cam_id,
        train_cam_id=train_cam_id,
        test_cam_id=test_cam_id,
        edge_cam_id=edge_cam_id,
        mask_id=mask_id,
        flow_id=flow_id,
        dataset_dir=dataset_dir,
        seq=seq,
        device=cuda_id,
        is_sport=is_sport,
    )
    return common_param


def get_dataset(common_param:Common_Param, t:int, train_flag:bool=True, 
                depth_flag:bool=False, want_mask:bool=False)->list:
    """
    Retrieves the dataset for a specific timestep and training/evaluation mode.

Args:
        common_param (Common_Param): Common parameters for the dataset.
        t (int): Timestep for the dataset.
        train_flag (bool, optional): Whether the dataset is for training. Defaults to True.
        depth_flag (bool, optional): Whether to include depth information. Defaults to False.
        want_mask (bool, optional): Whether to include additional mask information. Defaults to False.

    Returns:
        list: A list of data entries for the specified timestep.
    """
    
    dataset = []
    if train_flag:
        dataset_range = common_param.dataset_train_cam_id
        camidx_range = common_param.train_cam_id
    else:
        dataset_range = common_param.dataset_test_cam_id
        camidx_range = common_param.test_cam_id
        
    for i in range(len(dataset_range)):
        c = dataset_range[i]
        c_idx = camidx_range[i]

        fn = str(t).zfill(4) + '.png'
        if common_param.is_sport:
            fn = str(t).zfill(4) + '.jpg'
        img_path = os.path.join(common_param.dataset_dir,common_param.seq,'imgs', str(c), fn)
        im = np.array(copy.deepcopy(Image.open(img_path)))
        im = torch.tensor(im).float().permute(2, 0, 1) / 255

        ds4image = im.shape[-1] // common_param.w
        if ds4image != 1:
            im = F.avg_pool2d(im.unsqueeze(0), ds4image).squeeze(0)
        
        if depth_flag:
            depth_fn = str(t).zfill(4) + '-dpt_large_384.png'
            depth_path = os.path.join(common_param.dataset_dir,common_param.seq,'depths', str(c), depth_fn)
            depth = np.array(copy.deepcopy(Image.open(depth_path)))
            depth = torch.tensor(depth).unsqueeze(0).float()/65535.
            ds4depth = depth.shape[-1] // common_param.w
            if ds4image != 1:
                depth = F.avg_pool2d(depth.unsqueeze(0), ds4depth).squeeze(0)
        else:
            depth = None
            
        if c_idx in common_param.mask_id or want_mask:
            if t>0:
                mn = str(t-1).zfill(5) + '.png'
                if common_param.is_sport:
                    mn = str(t-1).zfill(4) + '.png'
            else:
                mn = str(t).zfill(5) + '.png'
                if common_param.is_sport:
                    mn = str(t).zfill(4) + '.png'
            masks = []
            for o in range(common_param.obj_num+1):
                on = str(o).zfill(2)
                mask_path = os.path.join(common_param.dataset_dir,common_param.seq,f'mask_{on}',str(c),mn)
                try:
                    masks.append(np.array(copy.deepcopy(Image.open(mask_path))).astype(np.uint8))
                except FileNotFoundError:
                    masks.append(np.zeros((common_param.h, common_param.w), dtype=np.uint8))
            masks = np.stack(masks)
            masks = torch.tensor(masks/255).int()

            ds4mask = masks.shape[-1] // common_param.w
            if ds4mask != 1:
                masks = F.max_pool2d(masks.float().unsqueeze(0), ds4mask).int().squeeze(0)
        else:
            masks = None
        dataset.append({'cam': common_param.cams[c_idx], 'im': im, 'masks': masks, 'depth': depth})
    
    return dataset


def get_batch(common_param:Common_Param, todo_dataset:list|None, 
              dataset:list, depth_flag:bool=False, shuffle:bool=True)->tuple:
    """Get a batch of data from the dataset.

    Args:
        common_param (Common_Param): An object containing all relevant camera and dataset parameters.
        todo_dataset (list | None): A list of data entries to process.
        dataset (list): The complete dataset to sample from.
        depth_flag (bool, optional): Whether to include depth information. Defaults to False.
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to True.

    Returns:
        tuple: A tuple containing the updated todo_dataset and the current data entry.
    """
    
    torch.cuda.set_device(common_param.device)
    if not todo_dataset:
        todo_dataset = dataset.copy()
    if shuffle:
        curr_data = todo_dataset.pop(randint(0, len(todo_dataset) - 1))
    else:
        curr_data = todo_dataset.pop(0)
    curr_data['cam'] = curr_data['cam']
    curr_data['im'] = curr_data['im'].cuda()
    if depth_flag:
        curr_data['depth'] = curr_data['depth'].cuda()
    return todo_dataset, curr_data


def setup_camera(w: int, h: int, k: torch.Tensor, 
                 w2c: torch.Tensor, cam_center: torch.Tensor, 
                 near: float = 0.01, far: float = 100) -> Camera:
    """Setup the camera parameters.

    Args:
        w (int): The width of the image.
        h (int): The height of the image.
        k (torch.Tensor): The camera intrinsics.
        w2c (torch.Tensor): The world-to-camera transformation matrix.
        cam_center (torch.Tensor): The camera center position.
        near (float, optional): The near clipping plane distance. Defaults to 0.01.
        far (float, optional): The far clipping plane distance. Defaults to 100.

    Returns:
        Camera: The configured camera object.
    """
    
    fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
    w2c = w2c.cuda().float()
    # cam_center = torch.inverse(w2c)[:3, 3]
    w2c = w2c.unsqueeze(0).transpose(1, 2)
    opengl_proj = torch.tensor([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
                                [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
                                [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
                                [0.0, 0.0, 1.0, 0.0]]).cuda().float().unsqueeze(0).transpose(1, 2)
    full_proj = w2c.bmm(opengl_proj)
    cam = Camera(
        image_height=h,
        image_width=w,
        tanfovx=w / (2 * fx),
        tanfovy=h / (2 * fy),
        bg=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=w2c,
        projmatrix=full_proj,
        sh_degree=0,
        campos=cam_center,
        prefiltered=False,
    )
    return cam


def setup_camera_v2(W: int, H: int, focal: float, 
                    w2c: torch.Tensor, cam_center: torch.Tensor, 
                    near: float = 0.01, far: float = 100) -> Camera:
    """Setup the camera parameters (version 2). slightly different focal length handling."""

    w2c = w2c.cuda().float().unsqueeze(0).transpose(1, 2)
    cam_center = cam_center.cuda().float()
    opengl_proj = torch.tensor([[2 * focal / W, 0.0, 0.0, 0.0],
                            [0.0, 2 * focal / H, 0.0, 0.0],
                            [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
                            [0.0, 0.0, 1.0, 0.0]]).cuda().float().unsqueeze(0).transpose(1, 2)
    full_proj = w2c.bmm(opengl_proj)
    
    cam = Camera(
        image_height=H,
        image_width=W,
        tanfovx=W / (2 * focal),
        tanfovy=H / (2 * focal),
        bg=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=w2c,
        projmatrix=full_proj,
        sh_degree=0,
        campos=cam_center,
        prefiltered=False,
        # debug=False,
    )
    return cam


def o3d_knn(q_pts: np.ndarray, t_pts: np.ndarray, 
            num_knn: int, min_contained: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Find the k-nearest neighbors using Open3D. used for gs initialization, 

    Args:
        q_pts (np.ndarray): Query points.
        t_pts (np.ndarray): Target points.
        num_knn (int): Number of nearest neighbors.
        min_contained (bool): Whether to use only points contained in the target point cloud.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The squared distances and indices of the nearest neighbors,
        consistent with process in 4DGS.
    """
    
    indices = []
    sq_dists = []
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(t_pts, np.float64))
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    if not min_contained:
        for p in q_pts:
            [_, i, d] = pcd_tree.search_knn_vector_3d(p, num_knn + 1)
            indices.append(i[1:])
            sq_dists.append(d[1:])
    else:
        for p in q_pts:
            [_, i, d] = pcd_tree.search_knn_vector_3d(p, num_knn)
            indices.append(i)
            sq_dists.append(d)
    return np.array(sq_dists), np.array(indices)


def select_cams(campos:torch.Tensor, train_cam_id:int, num_points:int)->list:
    """Select representative cameras from a set of training cameras using KMeans.

    This function clusters the provided camera centers into num_points clusters and
    returns one camera id per cluster (the camera whose center is nearest to the
    cluster centroid). The implementation converts the input tensor to numpy and
    relies on sklearn.cluster.KMeans and scipy.spatial.distance to find the
    nearest camera within each cluster.

    Args:
        campos (torch.Tensor): Tensor of camera centers with shape (C, 3) (world coordinates).
        train_cam_id (list[int]): List of camera ids/indices to consider for selection.
        num_points (int): Number of cameras to select (number of clusters).

    Returns:
        list[int]: Selected camera ids taken from train_cam_id, length == num_points.
    """
    
    from sklearn.cluster import KMeans
    from scipy.spatial import distance
    campos = campos[train_cam_id].cpu().numpy()
    kmeans = KMeans(n_clusters=num_points, random_state=0).fit(campos)
    cam_ids = []
    for i in range(num_points):
        points_cate_lb = np.where(kmeans.labels_ == i)
        cluster_points = campos[points_cate_lb]
        distances = distance.cdist(cluster_points, [kmeans.cluster_centers_[i]], 'euclidean')
        nearest_point_index = distances.argmin()
        cam_id = train_cam_id[points_cate_lb[0][nearest_point_index]]
        cam_ids.append(cam_id)
    # print(cam_ids)
    return cam_ids