from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Input, Flatten, Concatenate, Reshape,\
    Softmax, BatchNormalization, ReLU, ZeroPadding2D
from tensorflow.keras import Model
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
    input_tensor = Input(shape=input_shape, name='input_1')
    img_size = (input_shape[1], input_shape[0])

    net['input_1'] = input_tensor

    net['conv1'] = Conv2D(32, (3, 3), padding='same', strides=2, use_bias=False, name='conv1')(net['input_1'])
    net['conv1_bn'] = BatchNormalization(name='conv1_bn')(net['conv1'])
    net['conv1_relu'] = ReLU(6.0, name='conv1_relu')(net['conv1_bn'])

    net['conv_dw_1'] = DepthwiseConv2D((3, 3), strides=1, padding='same', use_bias=False, name='conv_dw_1')(net['conv1_relu'])
    net['conv_dw_1_bn'] = BatchNormalization(name='conv_dw_1_bn')(net['conv_dw_1'])
    net['conv_dw_1_relu'] = ReLU(6.0, name='conv_dw_1_relu')(net['conv_dw_1_bn'])

    net['conv_pw_1'] = Conv2D(64, (1, 1), use_bias=False, name='conv_pw_1')(net['conv_dw_1_relu'])
    net['conv_pw_1_bn'] = BatchNormalization(name='conv_pw_1_bn')(net['conv_pw_1'])
    net['conv_pw_1_relu'] = ReLU(6.0, name='conv_pw_1_relu')(net['conv_pw_1_bn'])

    net['conv_pad_2'] = ZeroPadding2D(((0, 1), (0, 1)), name='conv_pad_2')(net['conv_pw_1_relu'])

    net['conv_dw_2'] = DepthwiseConv2D((3, 3), strides=2, padding='valid', use_bias=False, name='conv_dw_2')(net['conv_pad_2'])
    net['conv_dw_2_bn'] = BatchNormalization(name='conv_dw_2_bn')(net['conv_dw_2'])
    net['conv_dw_2_relu'] = ReLU(6.0, name='conv_dw_2_relu')(net['conv_dw_2_bn'])

    net['conv_pw_2'] = Conv2D(128, (1, 1), use_bias=False, name='conv_pw_2')(net['conv_dw_2_relu'])
    net['conv_pw_2_bn'] = BatchNormalization(name='conv_pw_2_bn')(net['conv_pw_2'])
    net['conv_pw_2_relu'] = ReLU(6.0, name='conv_pw_2_relu')(net['conv_pw_2_bn'])

    net['conv_dw_3'] = DepthwiseConv2D((3, 3), strides=1, padding='same', use_bias=False, name='conv_dw_3')(net['conv_pw_2_relu'])
    net['conv_dw_3_bn'] = BatchNormalization(name='conv_dw_3_bn')(net['conv_dw_3'])
    net['conv_dw_3_relu'] = ReLU(6.0, name='conv_dw_3_relu')(net['conv_dw_3_bn'])

    net['conv_pw_3'] = Conv2D(128, (1, 1), use_bias=False, name='conv_pw_3')(net['conv_dw_3_relu'])
    net['conv_pw_3_bn'] = BatchNormalization(name='conv_pw_3_bn')(net['conv_pw_3'])
    net['conv_pw_3_relu'] = ReLU(6.0, name='conv_pw_3_relu')(net['conv_pw_3_bn'])

    net['conv_pad_4'] = ZeroPadding2D(((0, 1), (0, 1)), name='conv_pad_4')(net['conv_pw_3_relu'])

    net['conv_dw_4'] = DepthwiseConv2D((3, 3), strides=2, padding='valid', use_bias=False, name='conv_dw_4')(net['conv_pad_4'])
    net['conv_dw_4_bn'] = BatchNormalization(name='conv_dw_4_bn')(net['conv_dw_4'])
    net['conv_dw_4_relu'] = ReLU(6.0, name='conv_dw_4_relu')(net['conv_dw_4_bn'])

    net['conv_pw_4'] = Conv2D(256, (1, 1), use_bias=False, name='conv_pw_4')(net['conv_dw_4_relu'])
    net['conv_pw_4_bn'] = BatchNormalization(name='conv_pw_4_bn')(net['conv_pw_4'])
    net['conv_pw_4_relu'] = ReLU(6.0, name='conv_pw_4_relu')(net['conv_pw_4_bn'])

    net['conv_dw_5'] = DepthwiseConv2D((3, 3), strides=1, padding='same', use_bias=False, name='conv_dw_5')(net['conv_pw_4_relu'])
    net['conv_dw_5_bn'] = BatchNormalization(name='conv_dw_5_bn')(net['conv_dw_5'])
    net['conv_dw_5_relu'] = ReLU(6.0, name='conv_dw_5_relu')(net['conv_dw_5_bn'])

    net['conv_pw_5'] = Conv2D(256, (1, 1), use_bias=False, name='conv_pw_5')(net['conv_dw_5_relu'])
    net['conv_pw_5_bn'] = BatchNormalization(name='conv_pw_5_bn')(net['conv_pw_5'])
    net['conv_pw_5_relu'] = ReLU(6.0, name='conv_pw_5_relu')(net['conv_pw_5_bn'])

    net['conv_pad_6'] = ZeroPadding2D(((0, 1), (0, 1)), name='conv_pad_6')(net['conv_pw_5_relu'])

    net['conv_dw_6'] = DepthwiseConv2D((3, 3), strides=2, padding='valid', use_bias=False, name='conv_dw_6')(net['conv_pad_6'])
    net['conv_dw_6_bn'] = BatchNormalization(name='conv_dw_6_bn')(net['conv_dw_6'])
    net['conv_dw_6_relu'] = ReLU(6.0, name='conv_dw_6_relu')(net['conv_dw_6_bn'])

    net['conv_pw_6'] = Conv2D(512, (1, 1), use_bias=False, name='conv_pw_6')(net['conv_dw_6_relu'])
    net['conv_pw_6_bn'] = BatchNormalization(name='conv_pw_6_bn')(net['conv_pw_6'])
    net['conv_pw_6_relu'] = ReLU(6.0, name='conv_pw_6_relu')(net['conv_pw_6_bn'])

    net['conv_dw_7'] = DepthwiseConv2D((3, 3), strides=1, padding='same', use_bias=False, name='conv_dw_7')(net['conv_pw_6_relu'])
    net['conv_dw_7_bn'] = BatchNormalization(name='conv_dw_7_bn')(net['conv_dw_7'])
    net['conv_dw_7_relu'] = ReLU(6.0, name='conv_dw_7_relu')(net['conv_dw_7_bn'])

    net['conv_pw_7'] = Conv2D(512, (1, 1), use_bias=False, name='conv_pw_7')(net['conv_dw_7_relu'])
    net['conv_pw_7_bn'] = BatchNormalization(name='conv_pw_7_bn')(net['conv_pw_7'])
    net['conv_pw_7_relu'] = ReLU(6.0, name='conv_pw_7_relu')(net['conv_pw_7_bn'])

    net['conv_dw_8'] = DepthwiseConv2D((3, 3), strides=1, padding='same', use_bias=False, name='conv_dw_8')(net['conv_pw_7_relu'])
    net['conv_dw_8_bn'] = BatchNormalization(name='conv_dw_8_bn')(net['conv_dw_8'])
    net['conv_dw_8_relu'] = ReLU(6.0, name='conv_dw_8_relu')(net['conv_dw_8_bn'])

    net['conv_pw_8'] = Conv2D(512, (1, 1), use_bias=False, name='conv_pw_8')(net['conv_dw_8_relu'])
    net['conv_pw_8_bn'] = BatchNormalization(name='conv_pw_8_bn')(net['conv_pw_8'])
    net['conv_pw_8_relu'] = ReLU(6.0, name='conv_pw_8_relu')(net['conv_pw_8_bn'])

    net['conv_dw_9'] = DepthwiseConv2D((3, 3), strides=1, padding='same', use_bias=False, name='conv_dw_9')(net['conv_pw_8_relu'])
    net['conv_dw_9_bn'] = BatchNormalization(name='conv_dw_9_bn')(net['conv_dw_9'])
    net['conv_dw_9_relu'] = ReLU(6.0, name='conv_dw_9_relu')(net['conv_dw_9_bn'])

    net['conv_pw_9'] = Conv2D(512, (1, 1), use_bias=False, name='conv_pw_9')(net['conv_dw_9_relu'])
    net['conv_pw_9_bn'] = BatchNormalization(name='conv_pw_9_bn')(net['conv_pw_9'])
    net['conv_pw_9_relu'] = ReLU(6.0, name='conv_pw_9_relu')(net['conv_pw_9_bn'])

    net['conv_dw_10'] = DepthwiseConv2D((3, 3), strides=1, padding='same', use_bias=False, name='conv_dw_10')(net['conv_pw_9_relu'])
    net['conv_dw_10_bn'] = BatchNormalization(name='conv_dw_10_bn')(net['conv_dw_10'])
    net['conv_dw_10_relu'] = ReLU(6.0, name='conv_dw_10_relu')(net['conv_dw_10_bn'])

    net['conv_pw_10'] = Conv2D(512, (1, 1), use_bias=False, name='conv_pw_10')(net['conv_dw_10_relu'])
    net['conv_pw_10_bn'] = BatchNormalization(name='conv_pw_10_bn')(net['conv_pw_10'])
    net['conv_pw_10_relu'] = ReLU(6.0, name='conv_pw_10_relu')(net['conv_pw_10_bn'])

    net['conv_dw_11'] = DepthwiseConv2D((3, 3), strides=1, padding='same', use_bias=False, name='conv_dw_11')(net['conv_pw_10_relu'])
    net['conv_dw_11_bn'] = BatchNormalization(name='conv_dw_11_bn')(net['conv_dw_11'])
    net['conv_dw_11_relu'] = ReLU(6.0, name='conv_dw_11_relu')(net['conv_dw_11_bn'])

    net['conv_pw_11'] = Conv2D(512, (1, 1), use_bias=False, name='conv_pw_11')(net['conv_dw_11_relu'])
    net['conv_pw_11_bn'] = BatchNormalization(name='conv_pw_11_bn')(net['conv_pw_11'])
    net['conv_pw_11_relu'] = ReLU(6.0, name='conv_pw_11_relu')(net['conv_pw_11_bn'])

    net['conv_pad_12'] = ZeroPadding2D(((0, 1), (0, 1)), name='conv_pad_12')(net['conv_pw_11_relu'])

    net['conv_dw_12'] = DepthwiseConv2D((3, 3), strides=2, padding='valid', use_bias=False, name='conv_dw_12')(net['conv_pad_12'])
    net['conv_dw_12_bn'] = BatchNormalization(name='conv_dw_12_bn')(net['conv_dw_12'])
    net['conv_dw_12_relu'] = ReLU(6.0, name='conv_dw_12_relu')(net['conv_dw_12_bn'])

    net['conv_pw_12'] = Conv2D(1024, (1, 1), use_bias=False, name='conv_pw_12')(net['conv_dw_12_relu'])
    net['conv_pw_12_bn'] = BatchNormalization(name='conv_pw_12_bn')(net['conv_pw_12'])
    net['conv_pw_12_relu'] = ReLU(6.0, name='conv_pw_12_relu')(net['conv_pw_12_bn'])

    net['conv_dw_13'] = DepthwiseConv2D((3, 3), strides=1, padding='same', use_bias=False, name='conv_dw_13')(net['conv_pw_12_relu'])
    net['conv_dw_13_bn'] = BatchNormalization(name='conv_dw_13_bn')(net['conv_dw_13'])
    net['conv_dw_13_relu'] = ReLU(6.0, name='conv_dw_13_relu')(net['conv_dw_13_bn'])

    net['conv_pw_13'] = Conv2D(1024, (1, 1), use_bias=False, name='conv_pw_13')(net['conv_dw_13_relu'])
    net['conv_pw_13_bn'] = BatchNormalization(name='conv_pw_13_bn')(net['conv_pw_13'])
    net['conv_pw_13_relu'] = ReLU(6.0, name='conv_pw_13_relu')(net['conv_pw_13_bn'])

    net['conv_14_1'] = Conv2D(256, (1, 1), use_bias=False, name='conv_14_1')(net['conv_pw_13_relu'])
    net['conv_14_1_bn'] = BatchNormalization(name='conv_14_1_bn')(net['conv_14_1'])
    net['conv_14_1_relu'] = ReLU(name='conv_14_1_relu')(net['conv_14_1_bn'])

    net['conv_14_2'] = Conv2D(512, (3, 3), strides=2, padding='same', use_bias=False, name='conv_14_2')(net['conv_14_1_relu'])
    net['conv_14_2_bn'] = BatchNormalization(name='conv_14_2_bn')(net['conv_14_2'])
    net['conv_14_2_relu'] = ReLU(name='conv_14_2_relu')(net['conv_14_2_bn'])

    net['conv_15_1'] = Conv2D(128, (1, 1), use_bias=False, name='conv_15_1')(net['conv_14_2_relu'])
    net['conv_15_1_bn'] = BatchNormalization(name='conv_15_1_bn')(net['conv_15_1'])
    net['conv_15_1_relu'] = ReLU(name='conv_15_1_relu')(net['conv_15_1_bn'])

    net['conv_15_2'] = Conv2D(256, (3, 3), strides=2, padding='same', use_bias=False, name='conv_15_2')(net['conv_15_1_relu'])
    net['conv_15_2_bn'] = BatchNormalization(name='conv_15_2_bn')(net['conv_15_2'])
    net['conv_15_2_relu'] = ReLU(name='conv_15_2_relu')(net['conv_15_2_bn'])

    net['conv_16_1'] = Conv2D(128, (1, 1), use_bias=False, name='conv_16_1')(net['conv_15_2_relu'])
    net['conv_16_1_bn'] = BatchNormalization(name='conv_16_1_bn')(net['conv_16_1'])
    net['conv_16_1_relu'] = ReLU(name='conv_16_1_relu')(net['conv_16_1_bn'])

    net['conv_16_2'] = Conv2D(256, (3, 3), strides=2, padding='same', use_bias=False, name='conv_16_2')(net['conv_16_1_relu'])
    net['conv_16_2_bn'] = BatchNormalization(name='conv_16_2_bn')(net['conv_16_2'])
    net['conv_16_2_relu'] = ReLU(name='conv_16_2_relu')(net['conv_16_2_bn'])

    net['conv_17_1'] = Conv2D(64, (1, 1), use_bias=False, name='conv_17_1')(net['conv_16_2_relu'])
    net['conv_17_1_bn'] = BatchNormalization(name='conv_17_1_bn')(net['conv_17_1'])
    net['conv_17_1_relu'] = ReLU(name='conv_17_1_relu')(net['conv_17_1_bn'])

    net['conv_17_2'] = Conv2D(128, (3, 3), strides=2, padding='same', use_bias=False, name='conv_17_2')(net['conv_17_1_relu'])
    net['conv_17_2_bn'] = BatchNormalization(name='conv_17_2_bn')(net['conv_17_2'])
    net['conv_17_2_relu'] = ReLU(name='conv_17_2_relu')(net['conv_17_2_bn'])

# mbox loc
    net['conv_11_mbox_loc'] = Conv2D(12, (1, 1))(net['conv_pw_11_relu'])
    net['conv_11_mbox_loc_flat'] = Flatten()(net['conv_11_mbox_loc'])

    net['conv_13_mbox_loc'] = Conv2D(24, (1, 1))(net['conv_pw_13_relu'])
    net['conv_13_mbox_loc_flat'] = Flatten()(net['conv_13_mbox_loc'])

    net['conv_14_2_mbox_loc'] = Conv2D(24, (1, 1))(net['conv_14_2_relu'])
    net['conv_14_2_mbox_loc_flat'] = Flatten()(net['conv_14_2_mbox_loc'])

    net['conv_15_2_mbox_loc'] = Conv2D(24, (1, 1))(net['conv_15_2_relu'])
    net['conv_15_2_mbox_loc_flat'] = Flatten()(net['conv_15_2_mbox_loc'])

    net['conv_16_2_mbox_loc'] = Conv2D(24, (1, 1))(net['conv_16_2_relu'])
    net['conv_16_2_mbox_loc_flat'] = Flatten()(net['conv_16_2_mbox_loc'])

    net['conv_17_2_mbox_loc'] = Conv2D(24, (1, 1))(net['conv_17_2_relu'])
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
    net['conv_11_mbox_conf'] = Conv2D(classes * 3, (1, 1))(net['conv_pw_11_relu'])
    # net['conv_11_mbox_conf'] = Permute((2, 3, 1))(net['conv_11_mbox_conf'])
    net['conv_11_mbox_conf_flat'] = Flatten()(net['conv_11_mbox_conf'])

    net['conv_13_mbox_conf'] = Conv2D(classes * 6, (1, 1))(net['conv_pw_13_relu'])
    # net['conv_13_mbox_conf'] = Permute((2, 3, 1))(net['conv_13_mbox_conf'])
    net['conv_13_mbox_conf_flat'] = Flatten()(net['conv_13_mbox_conf'])

    net['conv_14_2_mbox_conf'] = Conv2D(classes * 6, (1, 1))(net['conv_14_2_relu'])
    # net['conv_14_2_mbox_conf'] = Permute((2, 3, 1))(net['conv_14_2_mbox_conf'])
    net['conv_14_2_mbox_conf_flat'] = Flatten()(net['conv_14_2_mbox_conf'])

    net['conv_15_2_mbox_conf'] = Conv2D(classes * 6, (1, 1))(net['conv_15_2_relu'])
    # net['conv_15_2_mbox_conf'] = Permute((2, 3, 1))(net['conv_15_2_mbox_conf'])
    net['conv_15_2_mbox_conf_flat'] = Flatten()(net['conv_15_2_mbox_conf'])

    net['conv_16_2_mbox_conf'] = Conv2D(classes * 6, (1, 1))(net['conv_16_2_relu'])
    # net['conv_16_2_mbox_conf'] = Permute((2, 3, 1))(net['conv_16_2_mbox_conf'])
    net['conv_16_2_mbox_conf_flat'] = Flatten()(net['conv_16_2_mbox_conf'])

    net['conv_17_2_mbox_conf'] = Conv2D(classes * 6, (1, 1))(net['conv_17_2_relu'])
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
    net['mbox_conf'] = Reshape((-1, classes), name='mbox_conf_reshape')(
        net['mbox_conf'])

    net['mbox_conf'] = Softmax(name='mbox_conf_softmax')(net['mbox_conf'])

# prior box
    net['conv_11_mbox_priorbox'] = PriorBox(img_size, min_size=100, aspect_ratios=[2], variances=[0.1, 0.1, 0.2, 0.2],
                                            name='conv_11_mbox_priorbox')(
        net['conv_pw_11_relu'])
    net['conv_13_mbox_priorbox'] = PriorBox(img_size, min_size=170, max_size=240, aspect_ratios=[2, 3],
                                            variances=[0.1, 0.1, 0.2, 0.2], name='conv_13_mbox_priorbox')(
        net['conv_pw_13_relu'])
    net['conv_14_2_mbox_priorbox'] = PriorBox(img_size, min_size=240, max_size=310, aspect_ratios=[2, 3],
                                              variances=[0.1, 0.1, 0.2, 0.2], name='conv_14_2_mbox_priorbox')(
        net['conv_14_2_relu'])
    net['conv_15_2_mbox_priorbox'] = PriorBox(img_size, min_size=310, max_size=420, aspect_ratios=[2, 3],
                                              variances=[0.1, 0.1, 0.2, 0.2], name='conv_15_2_mbox_priorbox')(
        net['conv_15_2_relu'])
    net['conv_16_2_mbox_priorbox'] = PriorBox(img_size, min_size=420, max_size=490, aspect_ratios=[2, 3],
                                              variances=[0.1, 0.1, 0.2, 0.2], name='conv_16_2_mbox_priorbox')(
        net['conv_16_2_relu'])
    net['conv_17_2_mbox_priorbox'] = PriorBox(img_size, min_size=490, max_size=img_size[0], aspect_ratios=[2, 3],
                                              variances=[0.1, 0.1, 0.2, 0.2], name='conv_17_2_mbox_priorbox')(
        net['conv_17_2_relu'])

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

    model = Model(net['input_1'], net['predictions'])

    return model


