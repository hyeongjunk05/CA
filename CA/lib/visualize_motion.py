import cv2
import os
import numpy as np
import math
from tqdm import tqdm

category2color = {
    'Car':(0,255,0),
    'Pedestrian':(0,0,255),
    'Cyclist':(255, 165, 0)
}
category2size = {
    'Car':30,
    'Pedestrian':85,
    'Cyclist':53
}

def draw_projected_box3d(image, qs, category, thickness=1):
    ''' Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    '''
    color = category2color[category]
    text_size = category2size[category]
    qs = qs.astype(np.int32)
    min_x = np.min(qs[:, 0])
    min_y = np.min(qs[:, 1])
    cv2.rectangle(image, (min_x, min_y), (min_x + text_size, min_y - 18), color, -1)
    cv2.putText(image, category, (min_x, min_y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    
    for k in range(0,4):
       # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
       i,j=k,(k+1)%4
       # use LINE_AA for opencv3
       #cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.CV_AA)
       cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness)
       i,j=k+4,(k+1)%4 + 4
       cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness)

       i,j=k,k+4
       cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness)
    return image

def draw_projected_box3d_with_motion(image, qs, motion, category, thickness=1):
    ''' Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    '''
    color = category2color[category]
    text_size = category2size[category]
    qs = qs.astype(np.int32)
    min_x = np.min(qs[:, 0])
    min_y = np.min(qs[:, 1])
    cv2.rectangle(image, (min_x, min_y), (min_x + text_size, min_y - 18), color, -1)
    cv2.putText(image, category, (min_x, min_y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    # 計算速度
    speed = int(math.sqrt(motion[0]**2+motion[1]**2+motion[2]**2)*10*60*60/1000)
    if speed > 10:
        cv2.rectangle(image, (min_x, min_y-17), (min_x + 85, min_y - 35), color, -1)
    else:
        cv2.rectangle(image, (min_x, min_y-17), (min_x + 80, min_y - 35), color, -1)
    cv2.putText(image, f'{speed} km/hr', (min_x+3, min_y-22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    
    for k in range(0,4):
       # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
       i,j=k,(k+1)%4
       # use LINE_AA for opencv3
       #cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.CV_AA)
       cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness)
       i,j=k+4,(k+1)%4 + 4
       cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness)

       i,j=k,k+4
       cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness)
    return image


def project_3d(x3d, y3d, z3d, w3d, h3d, l3d, ry3d, motion_x, motion_y, motion_z):
    # compute rotational matrix around yaw axis
    R = np.array([[+math.cos(ry3d), 0, +math.sin(ry3d)],
                  [0, 1, 0],
                  [-math.sin(ry3d), 0, +math.cos(ry3d)]])

    # 3D bounding box corners
    x_corners = np.array([0, l3d, l3d, l3d, l3d,   0,   0,   0])
    y_corners = np.array([0, 0,   h3d, h3d,   0,   0, h3d, h3d])
    z_corners = np.array([0, 0,     0, w3d, w3d, w3d, w3d,   0])

    x_corners = np.array([l3d,   0,   0, l3d, l3d,   0,   0, l3d])
    y_corners = np.array([h3d, h3d, h3d, h3d,   0,   0,   0,   0])
    z_corners = np.array([w3d, w3d,   0,   0, w3d, w3d,   0,   0])

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
    return corners_2d
    # x = corners_2d[0, :]
    # min_x = np.min(x)
    # max_x = np.max(x)
    # y = corners_2d[1, :]
    # min_y = np.min(y)
    # max_y = np.max(y)

    # return min_x, min_y, max_x, max_y

def visualize_motion():
    # for i in tqdm(range(3769)):
    for i in tqdm(range(3768)): #7518
        idx = f'{i:06d}'
        # curr_image_path = f'/home/jsharp/M3D-RPN/data/kitti/testing/image_2/{idx}.png'
        curr_image_path = f'/mnt/data/KITTI/object/training/synthetic_motion/image_2/{idx}.png'
        curr_image = cv2.imread(curr_image_path)
        if curr_image is None:
            continue
        # prev_image_path = f'/home/jsharp/M3D-RPN/data/kitti/testing/prev_2/{idx}_01.png'
        # prev_image = cv2.imread(prev_image_path)
        prev_image_path = f'/mnt/data/KITTI/object/training/synthetic_motion/prev_2/{idx}.png'
        prev_image = cv2.imread(prev_image_path)

        # calib_path = f'/home/jsharp/M3D-RPN/data/kitti/testing/calib/{idx}.txt'
        calib_path = f'/home/jsharp/M3D-RPN/data/kitti_split1/validation/calib/{idx}.txt'
        with open(calib_path, 'r') as f:
            p2 = f.readlines()[2]
        p2 = list(map(float, p2.split()[1:]))
        p2 = np.array(p2).reshape(3, 4)
        # p2 = np.vstack([p2, np.array([0, 0, 0, 1])])
    
        # label_path = f'/home/jsharp/M3D-RPN/twcc_car_results/18.86two_frame_motion_scale_attention/data/{idx}.txt'
        # label_path = f'/home/jsharp/M3D-RPN/output/triple_frame/results/results_1/data/{idx}.txt'
        label_path = f'/mnt/data/KITTI/object/training/synthetic_motion/results/{idx}.txt'

        with open(label_path, 'r') as f:
            labels = f.readlines()
        
        for label in labels:
            try:
                category = label.split()[0]

                label = list(map(float, label.split()[1:]))
                x1, y1, x2, y2 = int(label[3]), int(label[4]), int(label[5]), int(label[6])
                w, h = x2-x1, y2-y1
                
                h3d, w3d, l3d = label[7], label[8], label[9]
                x3d, y3d, z3d, ry3d = label[10], label[11]-h3d/2, label[12], label[13]
                motion_x, motion_y, motion_z = label[-3], label[-2], label[-1]

                curr_corners_3d = project_3d(x3d, y3d, z3d, w3d, h3d, l3d, ry3d, 0, 0, 0)
                curr_corners_2d = get_2d_corners_from_3d(curr_corners_3d, p2)
                curr_corners_2d = curr_corners_2d.transpose()
                curr_image = draw_projected_box3d(curr_image, curr_corners_2d, category)
                
                prev_corners_3d = project_3d(x3d, y3d, z3d, w3d, h3d, l3d, ry3d, motion_x, motion_y, motion_z)
                prev_corners_2d = get_2d_corners_from_3d(prev_corners_3d, p2)
                prev_corners_2d = prev_corners_2d.transpose()
                prev_image = draw_projected_box3d(prev_image, prev_corners_2d, category)
                
            except:
                continue

        # cv2.imwrite(f'/home/jsharp/M3D-RPN/motion_visualization/triple_frame/{idx}.png', curr_image)
        # cv2.imwrite(f'/home/jsharp/M3D-RPN/motion_visualization/testing_results/{idx}_01.png', prev_image)
        cv2.imwrite(f'/mnt/data/KITTI/object/training/synthetic_motion/visualization/{idx}.png', curr_image)
        cv2.imwrite(f'/mnt/data/KITTI/object/training/synthetic_motion/visualization/{idx}_01.png', prev_image)
     
if __name__ == '__main__':
    visualize_motion()