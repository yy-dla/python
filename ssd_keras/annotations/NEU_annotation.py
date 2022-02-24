import os
import random
import xml.etree.ElementTree as ET

from ssd_keras.utils.utils import get_classes

classes_path = '../model_data/NEU_classes.txt'

trainval_percent = 0.9
train_percent = 0.9

NEU_path = 'F:/NEU-DET/'
image_path = 'F:/NEU-DET/IMAGES/'

classes, _ = get_classes(classes_path)

def convert_annotation(image_id, list_file):
    in_file = open(os.path.join(NEU_path, 'ANNOTATIONS/%s.xml' % image_id), encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0
        if obj.find('difficult') != None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)),
             int(float(xmlbox.find('ymin').text)),
             int(float(xmlbox.find('xmax').text)),
             int(float(xmlbox.find('ymax').text)))
        list_file.write(' ' + ','.join([str(a) for a in b]) + ',' + str(cls_id))


def convert_annotation_1(ann_name):
    in_file = open(NEU_path + 'ANNOTATIONS/' + ann_name)
    tree = ET.parse(in_file)
    root = tree.getroot()

    target_list = []

    _file_name = tree.find('filename').text

    for obj in root.iter('object'):
        difficult = 0
        if obj.find('difficult') != None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)),
             int(float(xmlbox.find('ymin').text)),
             int(float(xmlbox.find('xmax').text)),
             int(float(xmlbox.find('ymax').text)))

        target_list.append((b, cls_id))

    return _file_name, target_list


if __name__ == '__main__':
    ann_file_list = os.walk(NEU_path + 'ANNOTATIONS/')

    train_file_name = '../NEU_train.txt'
    val_file_name = '../NEU_val.txt'

    train_file = open(train_file_name, 'w')
    val_file = open(val_file_name, 'w')

    files = []

    for _, _, files in ann_file_list:
        pass

    gt_list = []

    for file_name in files:
        gt_list.append(convert_annotation_1(file_name))

    random.shuffle(gt_list)

    index = int(len(gt_list) * train_percent)

    for train_item in gt_list[:index]:
        s = ''
        if str(train_item[0]).endswith('.jpg'):
            s += image_path + str(train_item[0]) + ' '
        else:
            s += image_path + str(train_item[0]) + '.jpg '
        for target in train_item[1]:
            s += str(target[0][0]) + ',' + \
                 str(target[0][1]) + ',' + \
                 str(target[0][2]) + ',' + \
                 str(target[0][3]) + ',' + \
                 str(target[1]) + ' '
        s = s[:-1]
        s += '\n'

        train_file.write(s)

    for val_item in gt_list[index:]:
        s = ''
        if str(val_item[0]).endswith('.jpg'):
            s += image_path + str(val_item[0]) + ' '
        else:
            s += image_path + str(val_item[0]) + '.jpg '
        for target in val_item[1]:
            s += str(target[0][0]) + ',' + \
                 str(target[0][1]) + ',' + \
                 str(target[0][2]) + ',' + \
                 str(target[0][3]) + ',' + \
                 str(target[1]) + ' '
        s = s[:-1]
        s += '\n'

        val_file.write(s)

    train_file.close()
    val_file.close()

