import os
import random
import numpy as np

from utils.utils import get_classes

classes_path        = '../model_data/GTSDB_classes.txt'

trainval_percent    = 0.9
train_percent       = 0.9

GTSDB_img_path          = 'F:/GTSDB/ppmImages/'

gt_path             = 'F:/GTSDB'

GTSDB_sets          = [('GTSDB', 'train'), ('GTSDB', 'val')]

classes, _          = get_classes(classes_path)

img_width           = 1360
img_height          = 800

if __name__ == '__main__':
    random.seed(0)
    # read the gt file.
    img_list = np.genfromtxt(gt_path + '/gt.txt', delimiter=';', dtype=None, encoding=None)

    dic = {}
    for i in img_list:
        image_name = i[0]
        target = [i[1], i[2], i[3], i[4], i[5]]  # x_min, y_min, x_max, y_max, class
        if image_name in dic:
            dic[image_name].append(target)
        else:
            dic[image_name] = [target]

    train_file_name = '../GTSDB_train.txt'
    val_file_name = '../GTSDB_val.txt'

    train_file = open(train_file_name, 'w')
    val_file = open(val_file_name, 'w')

    ppm_index = 0
    for ppm in dic:
        # train data set
        if ppm_index < int(train_percent * len(dic)):
            train_file.write(GTSDB_img_path + ppm + ' ')
            s = ''
            for target in dic[ppm]:
                for item in target:
                    s += str(item) + ','
                s = s[:-1]
                s += str(' ')
            s = s[:-1]
            s += '\n'
            train_file.write(s)
        else:
            val_file.write(GTSDB_img_path + ppm + ' ')
            s = ''
            for target in dic[ppm]:
                for item in target:
                    s += str(item) + ','
                s = s[:-1]
                s += str(' ')
            s = s[:-1]
            s += '\n'
            val_file.write(s)

        ppm_index += 1

    train_file.close()
    val_file.close()

