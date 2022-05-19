import os
import math

def bb_intersection_over_union(boxA, boxB):
    	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

results_root = '/mnt/data/KITTI/object/training/synthetic_motion/results/'
with open('/mnt/data/KITTI/object/training/synthetic_motion/motion.txt', 'r') as f:
    motion_gts = f.readlines()

x_error = []
y_error = []
z_error = []
for i in range(3768):
    idx = f'{i:06d}'
    motion_gt = motion_gts[i]
    motion_gt = list(map(float, motion_gt.split()))
    if len(motion_gt) == 0:
        continue
    gt_x1, gt_x2, gt_y1, gt_y2, gt_motion_x, gt_motion_y, gt_motion_z = motion_gt
    # if gt_motion_z < 0:
    #     continue

    box_gt = [gt_x1, gt_y1, gt_x2, gt_y2]

    with open(os.path.join(results_root, f'{idx}.txt'), 'r') as f:
        labels = f.readlines()
    for label in labels:
        category = label.split()[0]
        if category != 'Cyclist':
            continue
        label = list(map(float, label.split()[1:]))
        x1, y1, x2, y2 = label[3], label[4], label[5], label[6]
        z3d = label[12]
        box_pred = [x1, y1, x2, y2]
        iou = bb_intersection_over_union(box_gt, box_pred)
        if iou < 0.7:
            continue
        else:
            motion_x, motion_y, motion_z = label[-3], label[-2], label[-1]
            # error = math.sqrt((gt_motion_x-motion_x)**2 + (gt_motion_y-motion_y)**2 + (gt_motion_z-motion_z)**2)
            if gt_motion_z < 0:
                k = 1
            else:
                k = 9

            for j in range(k):
                x_error.append(abs(gt_motion_x-motion_x))
                y_error.append(abs(gt_motion_y-motion_y))
                z_error.append(abs(gt_motion_z-motion_z))
            break

print(f'Sample numbers: {len(x_error)}')
print(f'x error: {sum(x_error)/len(x_error):.2f}')
print(f'y error: {sum(y_error)/len(y_error):.2f}')
print(f'z error: {sum(z_error)/len(z_error):.2f}')