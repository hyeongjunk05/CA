import cv2
import os
import numpy as np
import math
from tqdm import tqdm
import torch
import torch.nn as nn
import random
random.seed(1233)

def project_3d(x3d, y3d, z3d, w3d, h3d, l3d, ry3d, motion_x, motion_y, motion_z):
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

    # bounding box in object co-ordinate
    corners_3d = np.array([x_corners, y_corners, z_corners])

    # rotate
    corners_3d = R.dot(corners_3d)

    # translate
    corners_3d += np.array([x3d+motion_x, y3d+motion_y, z3d+motion_z]).reshape((3, 1))
    return corners_3d

def get_2d_corners_from_3d(corners_3d, p2):
    corners_3d = np.vstack([corners_3d, np.ones((1, 8))])
    corners_2d = p2 @ corners_3d
    corners_2d[0:2, :] /= corners_2d[2:3, :] 
    # return corners_2d
    x = corners_2d[0, :]
    min_x = np.min(x)
    max_x = np.max(x)
    y = corners_2d[1, :]
    min_y = np.min(y)
    max_y = np.max(y)

    return min_x, min_y, max_x, max_y

def warp_by_3d_motion():
    for i in tqdm(range(3769)):
        idx = f'{i:06d}'
        curr_image_path = f'/home/jsharp/M3D-RPN/data/kitti_split1/validation/image_2/{idx}.png'
        curr_image = cv2.imread(curr_image_path)
        prev_image_path = f'/home/jsharp/M3D-RPN/data/kitti_split1/validation/prev_2/{idx}_01.png'
        prev_image = cv2.imread(prev_image_path)

        calib_path = f'/home/jsharp/M3D-RPN/data/kitti_split1/validation/calib/{idx}.txt'
        with open(calib_path, 'r') as f:
            p2 = f.readlines()[2]
        p2 = list(map(float, p2.split()[1:]))
        p2 = np.array(p2).reshape(3, 4)
        # p2 = np.vstack([p2, np.array([0, 0, 0, 1])])
    
        label_path = f'/home/jsharp/M3D-RPN/output/kitti_3d_multi_view_car/results/results_1/data/{idx}.txt'

        with open(label_path, 'r') as f:
            labels = f.readlines()
        
        for label in labels:
            try:
                label = list(map(float, label.split()[1:]))
                x1, y1, x2, y2 = int(label[3]), int(label[4]), int(label[5]), int(label[6])
                w, h = x2-x1, y2-y1
                curr_image[y1:y2, x1:x2, :].fill(0)
                
                h3d, w3d, l3d = label[7], label[8], label[9]
                x3d, y3d, z3d, ry3d = label[10], label[11]-h3d/2, label[12], label[13]
                motion_x, motion_y, motion_z = label[-3], label[-2], label[-1]

                corners_3d = project_3d(x3d, y3d, z3d, w3d, h3d, l3d, ry3d, motion_x, motion_y, motion_z)
                box_2d = list(map(int, get_2d_corners_from_3d(corners_3d, p2)))
                patch = prev_image[box_2d[1]:box_2d[3], box_2d[0]:box_2d[2], :]
                patch = cv2.resize(patch, (w, h))
                curr_image[y1:y2, x1:x2, :] = patch
            except:
                continue
        cv2.imwrite(f'../warp/{idx}.png', curr_image)

def create_synthetic_motion():
    file = open('/mnt/data/KITTI/object/training/synthetic_motion/motion.txt', 'w')

    for i in tqdm(range(3768)): #3769
        idx = f'{i:06d}'
        next_idx = f'{i+1:06d}'
        curr_image_path = f'/home/jsharp/M3D-RPN/data/kitti_split1/validation/image_2/{idx}.png'
        curr_image = cv2.imread(curr_image_path)
        prev_image_path = f'/home/jsharp/M3D-RPN/data/kitti_split1/validation/prev_2/{idx}_01.png'
        prev_image = cv2.imread(prev_image_path)
        next_image = cv2.imread(f'/home/jsharp/M3D-RPN/data/kitti_split1/validation/image_2/{next_idx}.png')

        H, W, C = curr_image.shape
        # next_image = cv2.resize(next_image, (W, H))

        calib_path = f'/home/jsharp/M3D-RPN/data/kitti_split1/validation/calib/{idx}.txt'
        with open(calib_path, 'r') as f:
            p2 = f.readlines()[2]
        p2 = list(map(float, p2.split()[1:]))
        p2 = np.array(p2).reshape(3, 4)
        # p2 = np.vstack([p2, np.array([0, 0, 0, 1])])
    
        label_path = f'/home/jsharp/M3D-RPN/data/kitti_split1/validation/label_2/{next_idx}.txt'

        filtered_label = []
        with open(label_path, 'r') as f:
            labels = f.readlines()
        for label in labels:
            category = label.split()[0]
            truncation = float(label.split()[1])
            occlusion = int(label.split()[2])
            if (category == 'Car' or category == 'Pedestrian' or category == 'Cyclist') and truncation < 1e-6 and occlusion == 0:
                filtered_label.append(label)
        
        if len(filtered_label) == 0:
            file.write('\n')
            continue
        
        label_idx = random.randint(0, len(filtered_label)-1)

        label = filtered_label[label_idx]
    
        label = list(map(float, label.split()[1:]))
        next_x1, next_y1, next_x2, next_y2 = int(label[3]), int(label[4]), int(label[5]), int(label[6])
        
        h3d, w3d, l3d = label[7], label[8], label[9]
        x3d, y3d, z3d, ry3d = label[10], label[11]-h3d/2, label[12], label[13]

        corners_3d = project_3d(x3d, y3d, z3d, w3d, h3d, l3d, ry3d, 0, 0, 0)
        box_2d = list(map(int, get_2d_corners_from_3d(corners_3d, p2)))
        x1, y1, x2, y2 = box_2d[0], box_2d[1], box_2d[2], box_2d[3]
        w, h = x2-x1, y2-y1

        patch = next_image[next_y1:next_y2, next_x1:next_x2, :]
        patch = cv2.resize(patch, (w, h))

        x1_diff, x2_diff, y1_diff, y2_diff = 0, 0, 0, 0
        if x1 < 0:
            x1_diff = 0 - x1
        
        if y1 < 0:
            y1_diff = 0 - y1
        
        if x2 >= W:
            x2_diff = W - x2
        
        if y2 >= H:
            y2_diff = H - y2

        
        patch = patch[y1_diff:h+y2_diff, x1_diff:w+x2_diff, :]
        curr_image[y1+y1_diff:y2+y2_diff, x1+x1_diff:x2+x2_diff, :] = patch
        patch = curr_image[y1:y2, x1:x2, :]
        
        motion_x, motion_y, motion_z = random.random()*2-1, 0, random.random()*2-1
        file.write(f'{x1} {x2} {y1} {y2} {motion_x} {motion_y} {motion_z}\n')

        corners_3d = project_3d(x3d, y3d, z3d, w3d, h3d, l3d, ry3d, motion_x, motion_y, motion_z)
        box_2d = list(map(int, get_2d_corners_from_3d(corners_3d, p2)))
        w, h = box_2d[2] - box_2d[0], box_2d[3] - box_2d[1] 
    
        patch = cv2.resize(patch, (w, h))
        # print(box_2d)
        x1_diff, x2_diff, y1_diff, y2_diff = 0, 0, 0, 0
        if box_2d[0] < 0:
            x1_diff = 0 - box_2d[0]
        
        if box_2d[1] < 0:
            y1_diff = 0 - box_2d[1]
        
        if box_2d[2] >= W:
            x2_diff = W - box_2d[2]
        
        if box_2d[3] >= H:
            y2_diff = H - box_2d[3]

        
        patch = patch[y1_diff:h+y2_diff, x1_diff:w+x2_diff, :]
        prev_image[box_2d[1]+y1_diff:box_2d[3]+y2_diff, box_2d[0]+x1_diff:box_2d[2]+x2_diff, :] = patch
    
        cv2.imwrite(f'/mnt/data/KITTI/object/training/synthetic_motion/image_2/{idx}.png', curr_image)
        cv2.imwrite(f'/mnt/data/KITTI/object/training/synthetic_motion/prev_2/{idx}.png', prev_image)
    
    file.close()

def warp_by_flow():
    H = 512
    W = 1792
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W)
    yy = yy.view(1,1,H,W)
    grid = torch.cat((xx,yy),1).float()

    save_path = '/mnt/data/KITTI/object/testing/test'
    os.makedirs(save_path, exist_ok=True)

    for i in tqdm(range(7518)):
        idx = f'{i:06d}'

        # curr_image_path = f'/home/jsharp/M3D-RPN/data/kitti_split1/training/image_2/{idx}.png'
        # curr_image = cv2.imread(curr_image_path)
        # org_h, org_w = curr_image.shape[0], curr_image.shape[1]
        # curr_image = cv2.resize(curr_image, (flow.shape[1], flow.shape[0]))
        pre2_image = cv2.imread(f'/mnt/data/KITTI/object/testing/prev_2/{idx}_02.png')
        if pre2_image is not None:
            continue
        
        flow_path = f'/mnt/data/KITTI/object/testing/flow_npy/{idx}.npy'
        flow = np.load(flow_path)

        prev_image_path = f'/mnt/data/KITTI/object/testing/prev_2/{idx}_01.png'
        prev_image = cv2.imread(prev_image_path)
        org_h, org_w = prev_image.shape[0], prev_image.shape[1]
        prev_image = cv2.resize(prev_image, (flow.shape[1], flow.shape[0]))

        prev_image = torch.from_numpy(prev_image).permute(2, 0, 1).unsqueeze(dim=0).float()/255

        flow = torch.from_numpy(flow).permute([2, 0, 1]).unsqueeze(dim=0)
        des_pixel = grid + flow
        des_pixel[:,0,:,:] = 2.0*des_pixel[:,0,:,:].clone() / max(W-1,1)-1.0
        des_pixel[:,1,:,:] = 2.0*des_pixel[:,1,:,:].clone() / max(H-1,1)-1.0
        des_pixel = des_pixel.permute(0,2,3,1)        
        output = nn.functional.grid_sample(prev_image, des_pixel, align_corners=False)
        output = output.squeeze(dim=0).permute(1, 2, 0).numpy()*255

        output = cv2.resize(output, (org_w, org_h))

        cv2.imwrite(os.path.join(save_path, f'{idx}_02.png'), output)

def updata_flow_features():
    data_root = '/mnt/data/KITTI/object/training/flow_feature/'
    for idx in tqdm(os.listdir(data_root)):
        flow_path = os.path.join(data_root, idx)
        flow = np.load(flow_path)*20/16
        flow = cv2.resize(flow, (110, 32))
        save_path = os.path.join(data_root, idx.split('.')[0])
        np.save(save_path, flow)
        

if __name__ == '__main__':
    # warp_by_flow()
    create_synthetic_motion()
    
    
    