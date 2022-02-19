import os
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image

class create_dataset:
    def __init__(self, load_path='F:/GTSDB/small_img/', save_path='F:/GTSDB/new_set/',
                 max_amount_per_image=5, overlap_tolerance=0.1, use_random_background=True):
        self.load_path = load_path
        self.save_path = save_path
        self.max_amount_per_image = max_amount_per_image
        self.overlap_tolerance = overlap_tolerance
        self.use_random_background = use_random_background

        self.load_data = self.get_data()

    @staticmethod
    def isOverlap(img1, img2, tolerance):
        inter_upleft = np.maximum(img1[:2], img2[:2])
        inter_botright = np.minimum(img1[2:], img2[2:])

        inter_wh = inter_botright - inter_upleft
        inter_wh = np.maximum(inter_wh, 0)
        inter = inter_wh[0] * inter_wh[1]

        if inter == 0:
            return True, 0

        area_1 = (img1[2] - img1[0]) * (img1[3] - img1[1])
        area_2 = (img2[2] - img2[0]) * (img2[3] - img2[1])

        # iou = inter / (area_1 + area_2 - inter)

        tol_to_1 = inter / area_1
        tol_to_2 = inter / area_2

        if tol_to_1 > tolerance or tol_to_2 > tolerance:
            return False, inter

        return True, 0

    def get_data(self):
        dir_list = os.listdir(self.load_path)
        dic = dict()
        index = 0
        for _dir in dir_list:
            dic[index] = []
            for root, dirs, files in os.walk(self.load_path + _dir + '/'):
                dic[index].append(files)
            index += 1

        return dic

    def get_small_img(self):
        part_number = random.randrange(1, self.max_amount_per_image)
        smallImageList = []
        for i in range(part_number):
            select_index = random.randrange(43)
            part_len = len(self.load_data[select_index][0])
            part_select = random.randrange(part_len)
            smallImageList.append(
                str(select_index if select_index > 9 else '0' + str(select_index)) + '/' +
                self.load_data[select_index][0][part_select])
        return smallImageList

    def generate(self):
        smallImageList = self.get_small_img()

        if self.use_random_background:
            img = Image.new('RGB', (300, 300))
            for x in range(300):
                for y in range(300):
                    img.putpixel((x, y), (random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255)))
        else:
            img = Image.new('RGB', (300, 300), (127, 127, 127))

        notOverlap_flag = False
        target = []
        image_query = []

        while not notOverlap_flag:

            target.clear()
            image_query.clear()

            for smallImage in smallImageList:
                tImg = Image.open(self.load_path + smallImage)
                resize = random.uniform(1.0, 1.5)
                w, h = (int(tImg.width * resize), int(tImg.height * resize))
                tImg = tImg.resize((w, h))
                x, y = (random.randrange(-20, 280), random.randrange(-20, 280))
                image_query.append(tImg)
                # img.paste(tImg, (x, y))

                leftCol = x if x > 0 else 0
                topRow = y if y > 0 else 0
                rightCol = (x + w) if (x + w) < 300 else 300
                bottomRow = (y + h) if (y + h) < 300 else 300

                target.append([(leftCol, topRow, rightCol, bottomRow), smallImage[:2]])

            notOverlap_flag = True

            for a in range(len(target)):
                for b in range(a + 1, len(target)):
                    over = self.isOverlap(target[a][0], target[b][0], self.overlap_tolerance)
                    if not over[0]:
                        notOverlap_flag = False

        for (i, pos) in zip(image_query, target):
            img.paste(i, (pos[0][0], pos[0][1]))

        return img, target

    def create(self, amount=10):
        try:
            os.mkdir(self.save_path + 'images')
        except FileExistsError as e:
            print('The \'images\' folder is existed, skipping create it.')

        gt = open(self.save_path + 'gt.txt', 'w')

        for i in range(amount):
            img, target = self.generate()
            img.save(self.save_path + 'images/' + str(i) + '.jpg')
            gt.write(self.save_path + 'images/' + str(i) + '.jpg' + ' ')
            s = ''
            for tar in target:
                for item in tar[0]:
                    s += str(item)
                    s += ','
                s += str(int(tar[1])) + ' '

            s = s[:-1]

            gt.write(s + '\n')

        print('Done!')
        return True







# for tar in l:
#     ax = plt.gca()
#     ax.add_patch(plt.Rectangle((tar[0][0], tar[0][1]), tar[0][2] - tar[0][0], tar[0][3] - tar[0][1], color="blue", fill=False, linewidth=1))
#     ax.text(tar[0][0], tar[0][1], tar[1], bbox={'facecolor':'blue', 'alpha':0.5})
#
# plt.show()
# print(l)