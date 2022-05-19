import numpy as np
import math
import pykitti
from lib.kitti_raw_loader import *
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pandas as pd

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def project_3d(p2, x3d, y3d, z3d, w3d, h3d, l3d, ry3d, return_3d=False):
    """
    Projects a 3D box into 2D vertices

    Args:
        p2 (nparray): projection matrix of size 4x3
        x3d: x-coordinate of center of object
        y3d: y-coordinate of center of object
        z3d: z-cordinate of center of object
        w3d: width of object
        h3d: height of object
        l3d: length of object
        ry3d: rotation w.r.t y-axis
    """

    # compute rotational matrix around yaw axis
    R = np.array([[+math.cos(ry3d), 0, +math.sin(ry3d)],
                  [0, 1, 0],
                  [-math.sin(ry3d), 0, +math.cos(ry3d)]])

    # 3D bounding box corners
    x_corners = np.array([0, l3d, l3d, l3d, l3d,   0,   0,   0])
    y_corners = np.array([0, 0,   h3d, h3d,   0,   0, h3d, h3d])
    z_corners = np.array([0, 0,     0, w3d, w3d, w3d, w3d,   0])

    x_corners += -l3d / 2
    y_corners += -h3d / 2
    z_corners += -w3d / 2

    front_3d = np.array([l3d/2, 0, 0]).reshape((3, 1))
    front_3d = R.dot(front_3d)
    # print(ry3d)
    # print(angle_between(front_3d.reshape(-1), np.array([l3d/2, 0, 0]).reshape(-1)))
    front_3d += np.array([x3d, y3d, z3d]).reshape((3, 1))
    front_3d = np.vstack((front_3d, np.ones((front_3d.shape[-1]))))
    front_2d = p2.dot(front_3d)
    front_2d = (front_2d / front_2d[2])[:3]

    # bounding box in object co-ordinate
    corners_3d = np.array([x_corners, y_corners, z_corners])

    # rotate
    corners_3d = R.dot(corners_3d)

    # translate
    corners_3d += np.array([x3d, y3d, z3d]).reshape((3, 1))

    corners_3D_1 = np.vstack((corners_3d, np.ones((corners_3d.shape[-1]))))
    corners_2D = p2.dot(corners_3D_1)
    corners_2D = (corners_2D / corners_2D[2])[:3]

    bb3d_lines_verts_idx = [0, 1, 2, 3, 4, 5, 6, 7, 0, 5, 4, 1, 2, 7, 6, 3]

    verts3d = (corners_2D[:, bb3d_lines_verts_idx][:2]).astype(float).T

    if return_3d:
        return verts3d, corners_3d
    else:
        return verts3d, front_2d

def draw_boxes_3d(xyz_b, whl_b, ry_b, p2, img):
    draw = ImageDraw.Draw(img)
    for xyz, whl, ry in zip(xyz_b, whl_b, ry_b):
        verts3d, front_2d = project_3d(p2, xyz[0], xyz[1], xyz[2], whl[0], whl[1], whl[2], ry)
        x = np.round(min(verts3d[:, 0]))
        y = np.round(min(verts3d[:, 1]))
        x2 = np.round(max(verts3d[:, 0]))
        y2 = np.round(max(verts3d[:, 1]))
        
        draw.ellipse((front_2d[0]-10, front_2d[1]-10, front_2d[0]+10, front_2d[1]+10), fill = 'yellow')
        draw.rectangle((x, y, x2, y2), outline='red', width=5)
    return img

def draw_boxes_2d(box_b, img):
    draw = ImageDraw.Draw(img)
    for box in box_b:
        x1, y1, x2, y2 = box
        draw.rectangle((x1, y1, x2, y2), outline='red', width=5)
    return img

def compute_angle(R1, R2):
    cameraPoseVector1 = R1.T @ [0, 0, 1]
    cameraPoseVector1 /= np.linalg.norm(cameraPoseVector1)
    cameraPoseVector2 = R2.T @ [0, 0, 1]
    cameraPoseVector2 /= np.linalg.norm(cameraPoseVector2)
 
    angle = math.degrees(math.acos(np.clip(np.dot(cameraPoseVector1, cameraPoseVector2), -1, 1)))
    return angle  

def get_raw_data_from_split1_idx(permutation, mapping, split1_to_kitti, idx):
    idx = int(idx)
    kitti_idx = int(split1_to_kitti[idx])
    raw_data = mapping[permutation[kitti_idx]-1]
    return raw_data

def get_IMU_pose(date, drive, frame):
    metadata = np.genfromtxt(f'/mnt/data/raw_kitti/{date}/{date}_drive_{drive}_sync/oxts/data/{frame}.txt')
    lat = metadata[0]
    scale = np.cos(lat * np.pi / 180.)
    pose_matrix = pose_from_oxts_packet(metadata[:6], scale)
    return pose_matrix

def get_IMU_to_CAM_and_K(date):
    imu2velo = read_calib_file(f'/mnt/data/raw_kitti/{date}/calib_imu_to_velo.txt')
    velo2cam = read_calib_file(f'/mnt/data/raw_kitti/{date}/calib_velo_to_cam.txt')
    cam2cam = read_calib_file(f'/mnt/data/raw_kitti/{date}/calib_cam_to_cam.txt')
    
    k = cam2cam['K_02'].reshape(3, 3)
    velo2cam_mat = transform_from_rot_trans(velo2cam['R'], velo2cam['T'])
    imu2velo_mat = transform_from_rot_trans(imu2velo['R'], imu2velo['T'])
    cam_2rect_mat = transform_from_rot_trans(cam2cam['R_rect_00'], np.zeros(3))

    imu2cam = cam_2rect_mat @ velo2cam_mat @ imu2velo_mat

    return imu2cam, k

def get_P2(idx, mode):
    if mode == 'train':
        with open(f'/home/ms0365647/M3D-RPN/data/kitti_split1/training/calib/{idx}.txt', 'r') as f:
            p2 = f.readlines()[2]
    else:
        with open(f'/home/ms0365647/M3D-RPN/data/kitti_split1/validation/calib/{idx}.txt', 'r') as f:
            p2 = f.readlines()[2]
    p2 = list(map(float, p2.split()[1:]))
    p2 = np.array(p2).reshape(3, 4)
    return p2

def get_3d_label(idx, mode):
    labels = pd.read_csv(f'/home/ms0365647/M3D-RPN/data/kitti_split1/{mode}/label_2/{idx}.txt', header=None, delimiter=' ')
    labels.drop(columns=labels.columns[15:], inplace=True)
    labels.columns = ['class', 'truncate', 'occ', 'alpha', 'x1', 'y1', 'x2', 'y2', 'h3d', 'w3d', 'l3d', 'x3d', 'y3d', 'z3d', 'ry3d']
    bbox_3d = labels[['x3d', 'y3d', 'z3d', 'w3d', 'h3d', 'l3d', 'ry3d']].values
    bbox_3d[:, 1] -= bbox_3d[:, 4] / 2
    return bbox_3d[:, 0:3], bbox_3d[:, 3:6], bbox_3d[:, 6]

# def get_2d_label(idx, prev=False):
#     if prev:
#         labels = np.genfromtxt(f'/mnt/data/KITTI/object/training/label_2/{idx}.txt')
#     else:
#         labels = np.genfromtxt(f'/home/ms0365647/M3D-RPN/object/training/label_2/{idx}.txt')
#     num_objs = labels.shape[0]
#     box_2d = np.vstack([labels[:, 4], labels[:, 5], labels[:, 6], labels[:, 7]])
#     return box_2d.T

class RawKitti():
    def __init__(self, mode='train'):
        with open('/home/ms0365647/M3D-RPN/lib/train_rand.txt', 'r') as f:
            permutation = f.readlines()
            self.permutation = list(map(int, permutation[0].strip().split(',')))

        with open('/home/ms0365647/M3D-RPN/lib/train_mapping.txt', 'r') as f:
            self.mapping = [line.strip().split() for line in f.readlines()]
            
        with open(f'/mnt/data/KITTI/ImageSets/{mode}.txt', 'r') as f:
            self.split1_to_kitti = [line.strip() for line in f.readlines()]
        
        self.mode = mode

    def get_previous_p2(self, split1_idx, previous_num=1):
        raw_data = get_raw_data_from_split1_idx(self.permutation, self.mapping, self.split1_to_kitti, split1_idx)
        date, drive, frame = raw_data
        drive = drive.split('_')[-2]
        # print(date, drive, frame)
        imu2cam, K = get_IMU_to_CAM_and_K(date)
        ref_pose = get_IMU_pose(date, drive, frame)
        tar_pose = get_IMU_pose(date, drive, f'{int(frame)-previous_num:010d}')

        relative_pose = imu2cam @ np.linalg.inv(tar_pose) @ ref_pose @ np.linalg.inv(imu2cam)
        relative_pose = relative_pose[:3, :] # [3, 4]
        
        p2 = get_P2(split1_idx, self.mode)
        rt_2 = np.linalg.inv(K) @ p2
        rt_2 = np.vstack([rt_2, np.array([0, 0, 0, 1])])
        p_target = K @ relative_pose @ rt_2

        p2 = np.vstack([p2, np.array([0, 0, 0, 1])])
        p_target = np.vstack([p_target, np.array([0, 0, 0, 1])])
        return p2, p_target, relative_pose

if __name__ == "__main__":
    raw_kitti = RawKitti(mode='val')
    for idx in range(0, 3712):
        idx = f'{idx:06d}'
        p2, p_target, relative_pose = raw_kitti.get_previous_p2(idx, previous_num=1)
        c3d, whl, ry = get_3d_label(idx, mode='validation')

        img = Image.open(f'/home/ms0365647/M3D-RPN/data/kitti_split1/validation/image_2/{idx}.png')
        prev_img = Image.open(f'/home/ms0365647/M3D-RPN/data/kitti_split1/validation/prev_2/{idx}_01.png')
        img = draw_boxes_3d(c3d, whl, ry, p2, img)

        c3d = np.concatenate([c3d, np.ones((c3d.shape[0], 1))], axis=1).T
        c3d = relative_pose @ c3d
        print(c3d.shape)
        c3d = c3d.T[:, :3]

        prev_img = draw_boxes_3d(c3d, whl, ry, p2, prev_img)
        fig, ax = plt.subplots(2, 1)
        ax[0].imshow(img)
        ax[1].imshow(prev_img)
        plt.show()
        input()
    
    # dataset = pykitti.raw('/mnt/data/raw_kitti/', date, drive, frames=[int(frame)+1, int(frame)+2])
    
