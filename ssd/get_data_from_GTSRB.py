import os
import numpy as np

class gt_preprocessor(object):

    def __init__(self, data_path, num_classes):
        self.path_prefix = data_path
        self.num_classes = num_classes
        self.data = dict()
        self._preprocess_gt()

    def _preprocess_gt(self):
        width = 1360
        height = 800
        gt_list = np.genfromtxt(self.path_prefix + '/gt.txt',
                                delimiter=';', dtype=None, encoding=None)
        for gt_item in gt_list:
            image_name = gt_item[0]
            target = [gt_item[1] / width, gt_item[2] / height, gt_item[3] / width, gt_item[4] / height] # x_min, y_min, x_max, y_max, class
            target += (self._to_one_hot(gt_item[5]))
            if image_name in self.data:
                self.data[image_name].append(target)
            else:
                self.data[image_name] = [target]

        for key in self.data.keys():
            self.data[key] = np.asarray(self.data[key])


    def _to_one_hot(self, name):
        one_hot_vector = [0] * self.num_classes
        one_hot_vector[name] = 1

        return one_hot_vector

