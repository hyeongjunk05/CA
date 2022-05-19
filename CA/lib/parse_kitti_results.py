import numpy as np
import re

def parse_kitti_result(respath, mode=11):
    
    text_file = open(respath, 'r')

    acc = np.zeros([3, 41], dtype=float)

    lind = 0
    for line in text_file:

        parsed = re.findall('([\d]+\.?[\d]*)', line)

        for i, num in enumerate(parsed):
            acc[lind, i] = float(num)

        lind += 1

    text_file.close()

    if mode == 11:
        easy = np.mean(acc[0, 0:41:4])
        mod = np.mean(acc[1, 0:41:4])
        hard = np.mean(acc[2, 0:41:4])
    else:
        easy = np.mean(acc[0, 1:41:1])
        mod = np.mean(acc[1, 1:41:1])
        hard = np.mean(acc[2, 1:41:1])

    return easy, mod, hard

if __name__ == '__main__':
    folder = '/home/jsharp/M3D-RPN/twcc_car_results/split2_best_pretrained_2d'

    print('3D:')
    path = f'{folder}/stats_car_detection_3d.txt'
    print('car:')
    print(parse_kitti_result(path))
    path = f'{folder}/stats_pedestrian_detection_3d.txt'
    print('pedestrian:')
    print(parse_kitti_result(path))
    path = f'{folder}/stats_cyclist_detection_3d.txt'
    print('cyclist:')
    print(parse_kitti_result(path))

    print('BEV:')
    path = f'{folder}/stats_car_detection_ground.txt'
    print('car:')
    print(parse_kitti_result(path))
    path = f'{folder}/stats_pedestrian_detection_ground.txt'
    print('pedestrian:')
    print(parse_kitti_result(path))
    path = f'{folder}/stats_cyclist_detection_ground.txt'
    print('cyclist:')
    print(parse_kitti_result(path))

    print('2D:')
    path = f'{folder}/stats_car_detection.txt'
    print('car:')
    print(parse_kitti_result(path))
    path = f'{folder}/stats_pedestrian_detection.txt'
    print('pedestrian:')
    print(parse_kitti_result(path))
    path = f'{folder}/stats_cyclist_detection.txt'
    print('cyclist:')
    print(parse_kitti_result(path))
    # print(parse_kitti_result(path, mode=40))