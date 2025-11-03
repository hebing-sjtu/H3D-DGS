import torch

def quat_mult(q1:torch.Tensor, q2:torch.Tensor) -> torch.Tensor:
    """ Multiply two quaternions.
    """
    w1, x1, y1, z1 = q1.T
    w2, x2, y2, z2 = q2.T
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z]).T


def rotate_vector(v, q):
    """ Rotate vector v by quaternion q.
    """
    rot = build_rotation_from_quaternion(q)
    v_rot = torch.bmm(rot, v.unsqueeze(2)).squeeze(2)
    return v_rot


def bilinear_interpolate(depth:torch.Tensor, flow_uv:torch.Tensor) -> torch.Tensor:
    """ Perform bilinear interpolation on a depth map given flow UV coordinates.
    """
    h, w = depth.shape

    x = flow_uv[:, 1]
    y = flow_uv[:, 0]

    x0 = torch.floor(x).long()
    x1 = x0 + 1
    y0 = torch.floor(y).long()
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, w-2)
    x1 = torch.clamp(x1, 0, w-1)
    y0 = torch.clamp(y0, 0, h-2)
    y1 = torch.clamp(y1, 0, h-1)

    Ia = depth[y0, x0]
    Ib = depth[y1, x0]
    Ic = depth[y0, x1]
    Id = depth[y1, x1]

    wa = (x1.type(torch.float) - x) * (y1.type(torch.float) - y)
    wb = (x1.type(torch.float) - x) * (y - y0.type(torch.float))
    wc = (x - x0.type(torch.float)) * (y1.type(torch.float) - y)
    wd = (x - x0.type(torch.float)) * (y - y0.type(torch.float))

    return wa*Ia + wb*Ib + wc*Ic + wd*Id


def build_c2r_from_vector(vector:torch.Tensor) -> torch.Tensor:
    """given a vector pointing from camera center to somewhere on the image plane,
    build a camera-to-ray rotation matrix

    Args:
        vector (torch.Tensor): [B, 3] vector

    Returns:
        torch.Tensor: [B, 3, 3] camera-to-ray rotation matrix
    """
    device = vector.device
    z_ray = vector
    x_ray = torch.cross(torch.tensor([[0, 1, 0]]).float().to(device), z_ray)
    y_ray = torch.cross(z_ray, x_ray)
    z_ray = z_ray / torch.norm(z_ray, dim=1, keepdim=True)
    x_ray = x_ray / torch.norm(x_ray, dim=1, keepdim=True)
    y_ray = y_ray / torch.norm(y_ray, dim=1, keepdim=True)
    c2r = torch.stack((x_ray, y_ray, z_ray), dim=1)
    return c2r


def build_rotation_from_quaternion(q):
    """ Convert quaternion to rotation matrix.
    """
    norm = torch.sqrt(q[:, 0] * q[:, 0] + q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3])
    q = q / norm[:, None]
    rot = torch.zeros((q.size(0), 3, 3), device='cuda')
    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    rot[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rot[:, 0, 1] = 2 * (x * y - r * z)
    rot[:, 0, 2] = 2 * (x * z + r * y)
    rot[:, 1, 0] = 2 * (x * y + r * z)
    rot[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rot[:, 1, 2] = 2 * (y * z - r * x)
    rot[:, 2, 0] = 2 * (x * z - r * y)
    rot[:, 2, 1] = 2 * (y * z + r * x)
    rot[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rot


def build_quaternion_from_rotation(rot):
    """Convert rotation matrix to quaternion.
    """
    w = torch.sqrt(1 + rot[:, 0, 0] + rot[:, 1, 1] + rot[:, 2, 2]) / 2
    x = (rot[:, 2, 1] - rot[:, 1, 2]) / (4 * w)
    y = (rot[:, 0, 2] - rot[:, 2, 0]) / (4 * w)
    z = (rot[:, 1, 0] - rot[:, 0, 1]) / (4 * w)
    return torch.stack((w, x, y, z), dim=1)


def build_quaternion_from_euler(y,p,r):
    """Convert euler angles to quaternion
    """
    cy = torch.cos(y/2)
    sy = torch.sin(y/2)
    cp = torch.cos(p/2)
    sp = torch.sin(p/2)
    cr = torch.cos(r/2)
    sr = torch.sin(r/2)
    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = sy * cp * sr + cy * sp * cr
    z = sy * cp * cr - cy * sp * sr
    return torch.stack((w, x, y, z), dim=1)


def build_rotation_around_axis(theta, vector):
    """ Rodrigues' rotation formula
    Build rotation matrix around an axis by an angle.
    """
    c = torch.cos(theta)
    s = torch.sin(theta)
    x = vector[:,0]
    y = vector[:,1]
    z = vector[:,2]
    rot = torch.zeros((vector.size(0), 3, 3), device='cuda')
    rot[:, 0, 0] = c + (1 - c) * x * x
    rot[:, 0, 1] = (1 - c) * x * y - s * z
    rot[:, 0, 2] = (1 - c) * x * z + s * y
    rot[:, 1, 0] = (1 - c) * y * x + s * z
    rot[:, 1, 1] = c + (1 - c) * y * y
    rot[:, 1, 2] = (1 - c) * y * z - s * x
    rot[:, 2, 0] = (1 - c) * z * x - s * y
    rot[:, 2, 1] = (1 - c) * z * y + s * x
    rot[:, 2, 2] = c + (1 - c) * z * z
    return rot

def build_rotation_vertical_axis(theta1, theta2, vector):
    """build two rotation matrices rotating around two orthogonal axes,
    corresponding to two learnable angles theta1 and theta2.
    """
    x = torch.tensor([1,0,0])
    z = torch.tensor([0,0,1])
    z_flag = torch.sum(vector * z[None,:], dim=1) > 0
    axis0 = torch.zeros_like(vector)
    axis0[z_flag] = z
    axis0[~z_flag] = x
    vector1 = torch.cross(vector, axis0)/torch.norm(torch.cross(vector, axis0), dim=1)[:,None]
    vector2 = torch.cross(vector, vector1)/torch.norm(torch.cross(vector, vector1), dim=1)[:,None]
    rot1 = build_rotation_around_axis(theta1, vector1)  
    rot2 = build_rotation_around_axis(theta2, vector2)
    return rot1, rot2