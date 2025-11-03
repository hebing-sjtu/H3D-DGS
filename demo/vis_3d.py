import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
# COLMAP camera parameters
ds_ratio = 2
c = 0
npy_dir = r"C:\Users\29389\Desktop\plot"
bottom = np.array([[0, 0, 0, 1]], dtype=np.float32)
poses_arr = np.load(f'{npy_dir}/poses_bounds.npy').astype(float)
poses = poses_arr[:, :-2].reshape([-1, 3, 5])
H, W, focal = poses[0, :, -1]
H = int(H / ds_ratio)
W = int(W / ds_ratio)
focal = focal / ds_ratio
poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
focal_length = focal  # Focal length
principal_point = [W/2.- 0.5, H/2.- 0.5]  # Principal point
image_size = [W, H]  # Image size
pose = poses[c]
R = pose[:3,:3]
R = -R
R[:,0] = -R[:,0]
T = -pose[:3,3].dot(R)
w2c = np.concatenate([np.concatenate([R.T, T[..., None]], -1), bottom], 0)
c2w = np.linalg.inv(w2c)  # c2w matrix

# Convert to Open3D camera parameters
intrinsic = o3d.camera.PinholeCameraIntrinsic(image_size[0], image_size[1], focal_length, focal_length, principal_point[0], principal_point[1])
intrinsic.intrinsic_matrix = [[focal_length, 0, principal_point[0]], [0, focal_length, principal_point[1]], [0, 0, 1]]

camera = o3d.camera.PinholeCameraParameters()
camera.intrinsic = intrinsic
camera.extrinsic = np.array([[0., 0., 0., 0.], [1., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 1.]])
# view_control = o3d.visualization.ViewControl()
# view_control.convert_from_pinhole_camera_parameters(camera)
############################################################################################################

# 显示稀疏场景流 / 仅显示不可学习参数/全部参数 /gs点位置处的稠密场景流
def show_flow(prune,frame,obj):
    data = np.load(f'./flowdebug/prune{prune}/frame_{str(frame)}_obj_{str(obj)}_gslookflow.npz')
    print(data['pts'].shape)
    points0 = np.stack([data['pts'][:,0], -data['pts'][:,1], -data['pts'][:,2]], axis=1)
    t = np.stack([data['t'][:,0], -data['t'][:,1], -data['t'][:,2]], axis=1)
    # print('point_max:',points0.max(axis=0), 'point_min:',points0.min(axis=0))
    pts_col = data.get('col')
    points1 = points0 + t/2
    points2 = points0 + t
    points = np.concatenate([points0, points1, points2], axis=0)
    lines1 = np.stack((np.arange(0, len(points0)),np.arange(len(points0),2*len(points0))),axis=1)
    lines2 = np.stack((np.arange(len(points0),2*len(points0)),np.arange(2*len(points0),3*len(points0))),axis=1)
    lines = np.concatenate([lines1, lines2], axis=0)
    colors1 = np.tile(np.array([[1, 0, 0]]), (len(points0), 1))
    colors2 = np.tile(np.array([[0, 1, 0]]), (len(points0), 1))
    colors = np.concatenate([colors1, colors2], axis=0)

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(points)
    lineset.lines = o3d.utility.Vector2iVector(lines)
    lineset.colors = o3d.utility.Vector3dVector(colors)

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points0)
    point_cloud.colors = o3d.utility.Vector3dVector(pts_col)
    return point_cloud, lineset
############################################################################################################

# 显示gs中心位置
def show_gs_pts():
    data = np.load('./flowdebug/frame_1_obj_3_gslookflow.npz')
    points = data['pts']
    pts_col = data.get('col')
    if pts_col is not None:
        colors = pts_col
    else:
        colors = np.tile(np.array([[1, 0, 0]]), (len(points), 1))

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([point_cloud])

############################################################################################################

# 显示关键点与gs的拓扑关系
# gs_pts gs_num
# ctrl_pts ctrl_num
# gs_ctrl_id gs_num, knn
def show_ctrl_gs_connect(prune,frame,obj):
    knn = 3
    data = np.load(f'./flowdebug/prune{prune}/frame_{str(frame)}_obj_{str(obj)}_ctrlgsconnect.npz')
    gs_pts = data['gs_pts']
    gs_pts = np.stack([gs_pts[:,0], -gs_pts[:,1], -gs_pts[:,2]], axis=1)
    gs_col = data['gs_col']
    ctrl_pts = data['ctrl_pts']
    ctrl_pts = np.stack([ctrl_pts[:,0], -ctrl_pts[:,1], -ctrl_pts[:,2]], axis=1)
    gs_ctrl_id = data['ctrl_gs_id'].astype(int)
    points0 = gs_pts
    colors0 = gs_col
    points1 = ctrl_pts
    
    points_tmp = 1/2 * (ctrl_pts[gs_ctrl_id].reshape(-1, 3) + np.tile(points0[:,np.newaxis,:], (1, knn, 1)).reshape(-1, 3))
    points = np.concatenate([points0, points_tmp, points1], axis=0)

    line1 = np.stack((np.tile(np.arange(len(points0))[..., None], (1, knn)).reshape(-1),np.arange(len(points0), len(points0)+len(points_tmp))),axis=1)
    line_col1 = np.tile(np.array([[1, 0, 0]]), (len(line1), 1))
    line2 = np.stack((np.arange(len(points0), len(points0)+len(points_tmp)),len(points0)+len(points_tmp)+gs_ctrl_id.reshape(-1)),axis=1)
    line_col2 = np.tile(np.array([[0, 1, 0]]), (len(line2), 1))
    lines = np.concatenate([line1, line2], axis=0)
    line_col = np.concatenate([line_col1, line_col2], axis=0)

    unique_ctrl_ids = np.unique(gs_ctrl_id)
    print(gs_pts.shape, ctrl_pts.shape, unique_ctrl_ids.shape)
    effective_ctrl_pts = ctrl_pts[unique_ctrl_ids]
    colors1 = np.tile(np.array([[0, 1, 0]]), (len(effective_ctrl_pts), 1))
    pts = np.concatenate((points0,effective_ctrl_pts), axis=0)
    col = np.concatenate((colors0,colors1), axis=0)
    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(points)
    lineset.lines = o3d.utility.Vector2iVector(lines)
    lineset.colors = o3d.utility.Vector3dVector(line_col)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pts)
    point_cloud.colors = o3d.utility.Vector3dVector(col)
    return point_cloud, lineset
    # o3d.visualization.draw_geometries([lineset, point_cloud])

############################################################################################################
# Create a visualizer and add the point cloud
# vis = o3d.visualization.Visualizer()
# vis.create_window()
# vis.add_geometry(point_cloud)

# # Change the point size
# render_option = vis.get_render_option()
# render_option.point_size = 0.02  # Adjust as needed

# # Run the visualization
# vis.run()
# vis.destroy_window()
if __name__ == '__main__':
    show_gs_pts()
    # prune = '01'
    # frame = 1
    # obj = 3
    # # point_cloud, lineset = show_flow(prune,frame,obj)
    # point_cloud, lineset = show_ctrl_gs_connect(prune,frame,obj)
    # vis = o3d.visualization.Visualizer()

    # # Create a window
    # width = camera.intrinsic.width
    # height = camera.intrinsic.height
    # # print(width, height)
    # vis.create_window(width=width*5, height=height*5)

    # # Add the geometries
    # vis.add_geometry(lineset)
    # vis.add_geometry(point_cloud)
    # view_control = vis.get_view_control()

    # # Set the camera parameters
    # view_control.set_constant_z_far(1000)
    # view_control.set_constant_z_near(0.1)
    # vis.update_renderer()
    # view_control.convert_from_pinhole_camera_parameters(camera)
    # vis.run()
    # image = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    # plt.imsave('output.png', image)
    # vis.destroy_window()