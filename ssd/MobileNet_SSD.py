import keras
from keras.layers import Conv2D, DepthwiseConv2D, Input, merge, Flatten, Concatenate, Reshape, Softmax, Permute
from keras import Model
import tensorflow
import numpy as np
import pandas as pd
from ssd_layer import PriorBox


def create_ssd_anchors(num_layers=6,
                       min_scale=0.2,
                       max_scale=0.95):
    box_specs_list = []
    scales = [min_scale + (max_scale - min_scale) * i / (num_layers - 1)
              for i in range(num_layers)] + [1.0]
    return zip(scales[:-1], scales[1:])


def build_model(input_shape=(300, 300, 3), classes=44):
    net = {}

# Main Net
    input_tensor = Input(shape=input_shape)
    img_size = (input_shape[1], input_shape[0])

    net['input'] = input_tensor

    net['conv_0'] = Conv2D(32, (3, 3), activation='relu', padding='same', strides=2)(net['input'])

    net['conv_1_dw'] = DepthwiseConv2D((3, 3), strides=1, padding='same', activation='relu')(net['conv_0'])
    net['conv_1'] = Conv2D(64, (1, 1), activation='relu')(net['conv_1_dw'])

    net['conv_2_dw'] = DepthwiseConv2D((3, 3), strides=2, padding='same', activation='relu')(net['conv_1'])
    net['conv_2'] = Conv2D(128, (1, 1), activation='relu')(net['conv_2_dw'])

    net['conv_3_dw'] = DepthwiseConv2D((3, 3), strides=1, padding='same', activation='relu')(net['conv_2'])
    net['conv_3'] = Conv2D(128, (1, 1), activation='relu')(net['conv_3_dw'])

    net['conv_4_dw'] = DepthwiseConv2D((3, 3), strides=2, padding='same', activation='relu')(net['conv_3'])
    net['conv_4'] = Conv2D(256, (1, 1), activation='relu')(net['conv_4_dw'])

    net['conv_5_dw'] = DepthwiseConv2D((3, 3), strides=1, padding='same', activation='relu')(net['conv_4'])
    net['conv_5'] = Conv2D(256, (1, 1), activation='relu')(net['conv_5_dw'])

    net['conv_6_dw'] = DepthwiseConv2D((3, 3), strides=2, padding='same', activation='relu')(net['conv_5'])
    net['conv_6'] = Conv2D(512, (1, 1), activation='relu')(net['conv_6_dw'])

    net['conv_7_dw'] = DepthwiseConv2D((3, 3), strides=1, padding='same', activation='relu')(net['conv_6'])
    net['conv_7'] = Conv2D(512, (1, 1), activation='relu')(net['conv_7_dw'])

    net['conv_8_dw'] = DepthwiseConv2D((3, 3), strides=1, padding='same', activation='relu')(net['conv_7'])
    net['conv_8'] = Conv2D(512, (1, 1), activation='relu')(net['conv_8_dw'])

    net['conv_9_dw'] = DepthwiseConv2D((3, 3), strides=1, padding='same', activation='relu')(net['conv_8'])
    net['conv_9'] = Conv2D(512, (1, 1), activation='relu')(net['conv_9_dw'])

    net['conv_10_dw'] = DepthwiseConv2D((3, 3), strides=1, padding='same', activation='relu')(net['conv_9'])
    net['conv_10'] = Conv2D(512, (1, 1), activation='relu')(net['conv_10_dw'])

    net['conv_11_dw'] = DepthwiseConv2D((3, 3), strides=1, padding='same', activation='relu')(net['conv_10'])
    net['conv_11'] = Conv2D(512, (1, 1), activation='relu')(net['conv_11_dw'])

    net['conv_12_dw'] = DepthwiseConv2D((3, 3), strides=2, padding='same', activation='relu')(net['conv_11'])
    net['conv_12'] = Conv2D(1024, (1, 1), activation='relu')(net['conv_12_dw'])

    net['conv_13_dw'] = DepthwiseConv2D((3, 3), strides=1, padding='same', activation='relu')(net['conv_12'])
    net['conv_13'] = Conv2D(1024, (1, 1), activation='relu')(net['conv_13_dw'])

    net['conv_14_1'] = Conv2D(256, (1, 1), activation='relu')(net['conv_13'])

    net['conv_14_2'] = Conv2D(512, (3, 3), strides=2, padding='same', activation='relu')(net['conv_14_1'])

    net['conv_15_1'] = Conv2D(128, (1, 1), activation='relu')(net['conv_14_2'])

    net['conv_15_2'] = Conv2D(256, (3, 3), strides=2, padding='same', activation='relu')(net['conv_15_1'])

    net['conv_16_1'] = Conv2D(128, (1, 1), activation='relu')(net['conv_15_2'])

    net['conv_16_2'] = Conv2D(256, (3, 3), strides=2, padding='same', activation='relu')(net['conv_16_1'])

    net['conv_17_1'] = Conv2D(64, (1, 1), activation='relu')(net['conv_16_2'])

    net['conv_17_2'] = Conv2D(128, (3, 3), strides=2, padding='same', activation='relu')(net['conv_17_1'])

# mbox loc
    net['conv_11_mbox_loc'] = Conv2D(12, (1, 1))(net['conv_11'])
    net['conv_11_mbox_loc_flat'] = Flatten()(net['conv_11_mbox_loc'])

    net['conv_13_mbox_loc'] = Conv2D(24, (1, 1))(net['conv_13'])
    net['conv_13_mbox_loc_flat'] = Flatten()(net['conv_13_mbox_loc'])

    net['conv_14_2_mbox_loc'] = Conv2D(24, (1, 1))(net['conv_14_2'])
    net['conv_14_2_mbox_loc_flat'] = Flatten()(net['conv_14_2_mbox_loc'])

    net['conv_15_2_mbox_loc'] = Conv2D(24, (1, 1))(net['conv_15_2'])
    net['conv_15_2_mbox_loc_flat'] = Flatten()(net['conv_15_2_mbox_loc'])

    net['conv_16_2_mbox_loc'] = Conv2D(24, (1, 1))(net['conv_16_2'])
    net['conv_16_2_mbox_loc_flat'] = Flatten()(net['conv_16_2_mbox_loc'])

    net['conv_17_2_mbox_loc'] = Conv2D(24, (1, 1))(net['conv_17_2'])
    net['conv_17_2_mbox_loc_flat'] = Flatten()(net['conv_17_2_mbox_loc'])

    net['mbox_loc'] = Concatenate(axis=1, name='mbox_loc')([
        net['conv_11_mbox_loc_flat'],
        net['conv_13_mbox_loc_flat'],
        net['conv_14_2_mbox_loc_flat'],
        net['conv_15_2_mbox_loc_flat'],
        net['conv_16_2_mbox_loc_flat'],
        net['conv_17_2_mbox_loc_flat']
    ])

    net['mbox_loc'] = Reshape((net['mbox_loc'].shape[-1] // 4, 4), name='mbox_loc_final')(net['mbox_loc'])

# mbox conf
    net['conv_11_mbox_conf'] = Conv2D(classes * 3, (1, 1))(net['conv_11'])
    # net['conv_11_mbox_conf'] = Permute((2, 3, 1))(net['conv_11_mbox_conf'])
    net['conv_11_mbox_conf_flat'] = Flatten()(net['conv_11_mbox_conf'])

    net['conv_13_mbox_conf'] = Conv2D(classes * 6, (1, 1))(net['conv_13'])
    # net['conv_13_mbox_conf'] = Permute((2, 3, 1))(net['conv_13_mbox_conf'])
    net['conv_13_mbox_conf_flat'] = Flatten()(net['conv_13_mbox_conf'])

    net['conv_14_2_mbox_conf'] = Conv2D(classes * 6, (1, 1))(net['conv_14_2'])
    # net['conv_14_2_mbox_conf'] = Permute((2, 3, 1))(net['conv_14_2_mbox_conf'])
    net['conv_14_2_mbox_conf_flat'] = Flatten()(net['conv_14_2_mbox_conf'])

    net['conv_15_2_mbox_conf'] = Conv2D(classes * 6, (1, 1))(net['conv_15_2'])
    # net['conv_15_2_mbox_conf'] = Permute((2, 3, 1))(net['conv_15_2_mbox_conf'])
    net['conv_15_2_mbox_conf_flat'] = Flatten()(net['conv_15_2_mbox_conf'])

    net['conv_16_2_mbox_conf'] = Conv2D(classes * 6, (1, 1))(net['conv_16_2'])
    # net['conv_16_2_mbox_conf'] = Permute((2, 3, 1))(net['conv_16_2_mbox_conf'])
    net['conv_16_2_mbox_conf_flat'] = Flatten()(net['conv_16_2_mbox_conf'])

    net['conv_17_2_mbox_conf'] = Conv2D(classes * 6, (1, 1))(net['conv_17_2'])
    # net['conv_17_2_mbox_conf'] = Permute((2, 3, 1))(net['conv_17_2_mbox_conf'])
    net['conv_17_2_mbox_conf_flat'] = Flatten()(net['conv_17_2_mbox_conf'])

    net['mbox_conf'] = Concatenate(axis=1, name='mbox_conf_concat')([
        net['conv_11_mbox_conf_flat'],
        net['conv_13_mbox_conf_flat'],
        net['conv_14_2_mbox_conf_flat'],
        net['conv_15_2_mbox_conf_flat'],
        net['conv_16_2_mbox_conf_flat'],
        net['conv_17_2_mbox_conf_flat']
    ])

    # net['mbox_conf'] = Reshape((net['mbox_loc'].shape[-1] // 4, 43), name='mbox_conf_reshape')(
    #     net['mbox_conf'])
    net['mbox_conf'] = Reshape((-1, 44), name='mbox_conf_reshape')(
        net['mbox_conf'])

    net['mbox_conf'] = Softmax(name='mbox_conf_softmax')(net['mbox_conf'])

# prior box
    net['conv_11_mbox_priorbox'] = PriorBox((300, 300), min_size=60.0, aspect_ratios=[2], variances=[0.1, 0.1, 0.2, 0.2],
                                            name='conv_11_mbox_priorbox')(
        net['conv_11'])
    net['conv_13_mbox_priorbox'] = PriorBox((300, 300), min_size=105.0, max_size=150.0, aspect_ratios=[2, 3],
                                            variances=[0.1, 0.1, 0.2, 0.2], name='conv_13_mbox_priorbox')(
        net['conv_13'])
    net['conv_14_2_mbox_priorbox'] = PriorBox((300, 300), 150.0, max_size=195.0, aspect_ratios=[2, 3],
                                              variances=[0.1, 0.1, 0.2, 0.2], name='conv_14_2_mbox_priorbox')(
        net['conv_14_2'])
    net['conv_15_2_mbox_priorbox'] = PriorBox((300, 300), 195.0, max_size=240.0, aspect_ratios=[2, 3],
                                              variances=[0.1, 0.1, 0.2, 0.2], name='conv_15_2_mbox_priorbox')(
        net['conv_15_2'])
    net['conv_16_2_mbox_priorbox'] = PriorBox((300, 300), 240.0, max_size=285.0, aspect_ratios=[2, 3],
                                              variances=[0.1, 0.1, 0.2, 0.2], name='conv_16_2_mbox_priorbox')(
        net['conv_16_2'])
    net['conv_17_2_mbox_priorbox'] = PriorBox((300, 300), 285.0, max_size=300.0, aspect_ratios=[2, 3],
                                              variances=[0.1, 0.1, 0.2, 0.2], name='conv_17_2_mbox_priorbox')(
        net['conv_17_2'])

    net['mbox_priorbox'] = Concatenate(axis=1, name='mbox_priorbox')([
        net['conv_11_mbox_priorbox'],
        net['conv_13_mbox_priorbox'],
        net['conv_14_2_mbox_priorbox'],
        net['conv_15_2_mbox_priorbox'],
        net['conv_16_2_mbox_priorbox'],
        net['conv_17_2_mbox_priorbox']
    ])

# final
    net['predictions'] = Concatenate(axis=2, name='predictions')([
        net['mbox_loc'],
        net['mbox_conf'],
        net['mbox_priorbox']
    ])

    model = Model(net['input'], net['predictions'])

    return model


