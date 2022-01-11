import tensorflow as tf
import numpy as np
import re
import math
import struct


kSignMask = 0x8000000000000000
kExponentMask = 0x7ff0000000000000
kExponentShift = 52
kExponentBias = 1023
kExponentIsBadNum = 0x7ff
kFractionMask = 0x000fffffffc00000
kFractionShift = 22
kFractionRoundingMask = 0x003fffff
kFractionRoundingThreshold = 0x00200000


def convert_h5_to_c(model_name_list: list, model, save_file_path="config.h"):

    file = open(save_file_path, 'w', encoding='utf-8')

    for layer_name in model_name_list:

        layer_weight = model.get_layer(layer_name).get_weights()[1]
        layer_bias = model.get_layer(layer_name).get_weights()[0]

        if layer_name.find('depthwise') != -1:
            # depthwise conv convert
            file.write("const float %s_weight[%d][%d][%d] = {\n" % (layer_name, layer_weight.shape[2], 3, 3))
            for k in range(layer_weight.shape[2]):
                file.write('    {\n')
                for i in range(3):
                    file.write('        {')
                    for j in range(3):
                        file.write(str(layer_weight[i][j][k][0]) + ', ')
                    file.write('}, \n')
                file.write('    }, \n')
            file.write('};\n\n')
        elif layer_weight.shape[0] == 1:
            # pointwise conv convert
            file.write('const float %s_weight[%d][%d] = {\n' % (layer_name, layer_weight.shape[3], layer_weight.shape[2]))
            for j in range(layer_weight.shape[3]):
                file.write('    {')
                for i in range(layer_weight.shape[2]):
                    for k in range(1):
                        for l in range(1):
                            file.write(str(layer_weight[k][l][i][j]) + ', ')
                file.write('}, \n')
            file.write('};\n\n')
        elif layer_name.find('dense') != -1:
            # fully connect convert
            file.write('const float %s_weight[%d][%d] = {\n' % (layer_name, layer_weight.shape[1], layer_weight.shape[0]))
            for j in range(layer_weight.shape[1]):
                file.write('{')
                for i in range(layer_weight.shape[0]):
                    file.write(str(layer_weight[i][j]) + ', ')
                file.write('},\n')
            file.write('};\n\n')
        elif layer_name.find('batch_normalization') != -1:
            # BN convert
            file.write('const float %s_parameters[%d][4] = {\n' % (layer_name, layer_weight.shape[0]))
            for i in range(layer_weight.shape[0]):
                file.write('{%f, %f, %f, %f}, \n' % (model.get_layer(layer_name).get_weights()[0][i],
                                                     model.get_layer(layer_name).get_weights()[1][i],
                                                     model.get_layer(layer_name).get_weights()[2][i],
                                                     model.get_layer(layer_name).get_weights()[3][i]
                                                    )
                           )
            file.write('};\n\n')
        else:
            # traditional conv convert
            file.write('const float %s_weight[%d][3][9] = {\n' % (layer_name, layer_weight.shape[3]))
            for l in range(layer_weight.shape[3]):
                file.write('    {\n')
                for k in range(3):
                    file.write("        { ")
                    for j in range(3):
                        for i in range(3):
                            file.write(str(layer_weight[j][i][k][l]) + ', ')
                    file.write('}, \n')
                file.write('    }, \n')
            file.write('};\n\n')

        # write bias
        if layer_name.find('conv2d') != -1 or layer_name.find('dense') != -1:
            file.write('const float %s_bias[%d] = { \n    ' % (layer_name, len(layer_bias)))
            for i in range(len(layer_bias)):
                file.write(str(layer_bias[i]) + ', ')
            file.write('\n}; \n\n')

    file.close()


def generate_c_array(model_name_list: list, model):
    for layer_name in model_name_list:
        layer_weight = model.get_layer(layer_name).get_weights()[1]
        layer_bias = model.get_layer(layer_name).get_weights()[0]

        # print('float ***' + layer_name + '_weight;')
        # print('float ***' + layer_name + '_bias;')
        # print(layer_weight.shape)
        # print(layer_bias.shape)
        if layer_name.find('depthwise') != -1:
            print("""
this->%s_weight = new float** [%d];
for (int i = 0; i < %d; i++) {
    %s_weight[i] = new float* [3];
    for (int j = 0; j < 3; j++) {
        %s_weight[i][j] = new float[3];
    }
}
for (int i = 0; i < %d; i++) {
    for (int j = 0; j < %d; j++) {
        for (int k = 0; k < %d; k++) {
            %s_weight[i][j][k] = %s_weight[i][j][k];
        }
    } 
}     
            """ % (layer_name[6:],
                   layer_weight.shape[2],
                   layer_weight.shape[2],
                   layer_name[6:],
                   layer_name[6:],
                   layer_weight.shape[2],
                   3,
                   3,
                   layer_name[6:],
                   layer_name), end=""
                  )
        elif layer_weight.shape[0] == 1:
            print("""
this->%s_weight = new float* [%d];
for (int i = 0; i < %d; i++) {
    this->%s_weight[i] = new float [%d];
}
for (int i = 0; i < %d; i++) {
    for (int j = 0; j < %d; j++) {
        this->%s_weight[i][j] = %s_weight[i][j];
    } 
}     
            """ % (layer_name[6:],
                   layer_weight.shape[3],
                   layer_weight.shape[3],
                   layer_name[6:],
                   layer_weight.shape[2],
                   layer_weight.shape[3],
                   layer_weight.shape[2],
                   layer_name[6:],
                   layer_name)
                  )
        else:
            print("""
this->conv2d_weight = new float** [32];
for (int i = 0; i < 32; i++) {
    this->conv2d_weight[i] = new float* [3];
    for (int j = 0; j < 3; j++) {
        this->conv2d_weight[i][j] = new float[9];
    }
}
for (int i = 0; i < 32; i++) {
    for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 9; k++) {
            this->conv2d_weight[i][j][k] = quant_conv2d_weight[i][j][k];
        }
    }
}
""", end="")

        print("""
this->%s_bias = new float[%d];
for(int i = 0; i < %d; i++) {
    this->%s_bias[i] = %s_bias[i];
}
""" % (
            layer_name[6:],
            layer_bias.shape[0],
            layer_bias.shape[0],
            layer_name[6:],
            layer_name
        ), end="")



def generate_c_define(model_name_list: list, model):
    for layer_name in model_name_list:

        layer_bias = model.get_layer(layer_name).get_weights()[0]
        print('float ***' + layer_name[6:] + '_weight;')

    for layer_name in model_name_list:
        layer_weight = model.get_layer(layer_name).get_weights()[1]
        print('float *' + layer_name[6:] + '_bias;')


def generate_bn_init_array(model_name_list: list, model):
    i = 0
    for layer_name in model_name_list:
        if layer_name.find("batch_normalization") != -1:
            print("this->BN_%d.init(%d, quant_batch_normalization_%d_parameters);" % (i, model.get_layer(layer_name).get_weights()[0].shape[0], i))
            i += 1


def generate_bn_define(model_name_list: list, model):
    i = 0
    for layer_name in model_name_list:
        if layer_name.find("batch_normalization") != -1:
            print("BatchNormalization BN_%d;" % i)
            i += 1


def dec2INT8InHex(num: int):
    return str(hex(num))[2:].rjust(2, '0') if num >= 0 else str(hex((-num) | 0x80))[2:].rjust(2, '0')


def dec2INT32InHex(num: int):
    return str(hex(num))[2:].rjust(8, '0') if num >= 0 else str(hex((-num) | 0x80000000))[2:].rjust(8, '0')


from tensorflow.lite.python import schema_py_generated as schema_fb
import flatbuffers


def OutputsOffset(subgraph, j):
    o = flatbuffers.number_types.UOffsetTFlags.py_type(subgraph._tab.Offset(8))
    if o != 0:
        a = subgraph._tab.Vector(o)
        return a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4)
    return 0


def buffer_change_output_tensor_to(model_buffer, new_tensor_i):
    root = schema_fb.Model.GetRootAsModel(model_buffer, 0)
    output_tensor_index_offset = OutputsOffset(root.Subgraphs(0), 0)

    # Flatbuffer scalars are stored in little-endian.
    new_tensor_i_bytes = bytes([
        new_tensor_i & 0x000000FF,
        (new_tensor_i & 0x0000FF00) >> 8,
        (new_tensor_i & 0x00FF0000) >> 16,
        (new_tensor_i & 0xFF000000) >> 24
        ])
    # Replace the 4 bytes corresponding to the first output tensor index
    return model_buffer[:output_tensor_index_offset] + new_tensor_i_bytes + model_buffer[
                                                                            output_tensor_index_offset + 4:]


def get_M_list(tflite_model: tf.lite.Interpreter, index, shift=30):

    Sw = tflite_model.get_tensor_details()[index]['quantization_parameters']['scales'].astype(np.float64)

    Sx = tflite_model.get_tensor_details()[index - 1]['quantization_parameters']['scales'][0].astype(np.float64)
    Sa = tflite_model.get_tensor_details()[index + 1]['quantization_parameters']['scales'][0].astype(np.float64)

    Sw_list = []
    shift_list = []
    fraction_list = []
    fraction_after_shift = []
    M_list = []

    for i in Sw:
        Sw_list.append(float(i))

    for Sw_index in Sw_list:
        M = (Sx * Sw_index) / Sa
        M_list.append(M)
        t = InterFrExp(M)
        fraction_list.append(t[0])
        shift_list.append(t[1])

    shift_min = max(shift_list)
    final_shift = shift + abs(shift_min)

    for i in range(len(fraction_list)):
        if shift_list[i] != shift_min:
            t_shift = abs(shift_list[i]) - abs(shift_min)
            fraction_after_shift.append(dec2INT32InHex(fraction_list[i] >> t_shift))
        else:
            fraction_after_shift.append(dec2INT32InHex(fraction_list[i]))

    return fraction_after_shift, final_shift


def convert_h5_to_c_quant(model_path, save_file_path="config_quant.h", shift=30):
    f_generate = open(save_file_path, "w")

    tflite_model = tf.lite.Interpreter(model_path=model_path)
    tflite_model.allocate_tensors()

    layer_pos_list = [
        (32, 29), (34, 28), (36, 17), (38, 9), (40, 8), (42, 7), (44, 6), (46, 5), (48, 4), (50, 3), (52, 27), (54, 26), (56, 25), (58, 24), (60, 23), (62, 22), (64, 21), (66, 20), (68, 19), (70, 18), (72, 16), (74, 15), (76, 14), (78, 13), (80, 12), (82, 11), (84, 10), (88, 30)
    ]

    for layer_index in range(len(layer_pos_list)):
        if layer_index == 0: # t conv
            f_generate.write("static unsigned int t_conv_quant[] = {\n")

            layer = tflite_model.get_tensor(layer_pos_list[layer_index][0])

            strl = []
            for c in range(32):
                s = ""
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            # s += dec2INT8InHex(layer[c][j][k][i])
                            s += dec2INT8InHex(layer[c][i][j][k])
                            # print(layer[c][j][k][i])
                strl.append(s)

            ss = ''
            for i in strl:
                ss += i

            res = re.findall(r'.{8}', ss)
            for i in res:
                f_generate.write('0x' + i + ',\n')

            layer = tflite_model.get_tensor(layer_pos_list[layer_index][1])

            for i in layer:
                f_generate.write('0x' + dec2INT32InHex(i) + ',\n')

            M_list, shift_final = get_M_list(tflite_model, layer_pos_list[layer_index][0], shift)

            for m in M_list:
                f_generate.write('0x' + m + ',\n')

            f_generate.write('};\n')

            f_generate.write('static const int t_conv_shift = ' + str(shift_final) + ';\n')

        elif layer_index == 27: # fc

            f_generate.write("static unsigned int fc_quant[] = {\n")

            w_layer = tflite_model.get_tensor(layer_pos_list[layer_index][0])
            b_layer = tflite_model.get_tensor(layer_pos_list[layer_index][1])
            M_list, shift_final = get_M_list(tflite_model, layer_pos_list[layer_index][0], shift)

            # first 32
            gt = []
            s = ""
            for c in range(int(1024 / 32)):
                tl = []
                g = []
                for nn in range(32):
                    for cc in range(c * 32, c * 32 + 32):
                        tl.append(dec2INT8InHex(w_layer[nn, cc]))
                    g.clear()
                    for i in range(int(len(tl) / 4)):
                        gl = []
                        for temp in range(4):
                            gl.append(tl[i * 4 + temp])
                        gl.reverse()
                        g.append(gl)

                for gs in g:
                    ts = ""
                    for i in range(4):
                        ts += gs[i]
                    gt.append(ts)

                    # t_r = re.findall(r'.{8}', st)

            for bb in range(32):
                s += dec2INT32InHex(b_layer[bb])

            for mm in range(32):
                s += M_list[0]

            res = re.findall(r'.{8}', s)

            res = gt + res

            for i in res:
                f_generate.write('0x' + i + ',\n')

            # remain 11
            gt = []
            s = ""
            for c in range(int(1024 / 32)):
                tl = []
                g = []
                for nn in range(32, 64):
                    for cc in range(c * 32, c * 32 + 32):
                        if nn < 43:
                            tl.append(dec2INT8InHex(w_layer[nn, cc]))
                        else:
                            tl.append(dec2INT8InHex(0))
                    g.clear()
                    for i in range(int(len(tl) / 4)):
                        gl = []
                        for temp in range(4):
                            gl.append(tl[i * 4 + temp])
                        gl.reverse()
                        g.append(gl)

                for gs in g:
                    ts = ""
                    for i in range(4):
                        ts += gs[i]
                    gt.append(ts)

                    # t_r = re.findall(r'.{8}', st)

            for bb in range(32, 64):
                if bb < 43:
                    s += dec2INT32InHex(b_layer[bb])
                else:
                    s += dec2INT32InHex(0)

            for mm in range(32, 64):
                if mm < 43:
                    s += M_list[0]
                else:
                    s += dec2INT32InHex(0)

            res = re.findall(r'.{8}', s)

            res = gt + res

            for i in res:
                f_generate.write('0x' + i + ',\n')

            f_generate.write("};\n")

            f_generate.write("static const int dense_shift = " + str(shift_final) + ';\n')

        elif layer_index % 2 == 1: # dw conv
            f_generate.write("static unsigned int dw_conv_" + str(int((layer_index - 1) / 2)) + "_quant[] = {\n")

            w_layer = tflite_model.get_tensor(layer_pos_list[layer_index][0])
            b_layer = tflite_model.get_tensor(layer_pos_list[layer_index][1])
            M_list, shift_final = get_M_list(tflite_model, layer_pos_list[layer_index][0], shift)

            for n in range(int(w_layer.shape[3] / 32)):
                s = ""
                for c in range(32):
                    for i in range(3):
                        for j in range(3):
                            s += dec2INT8InHex(w_layer[0][i][j][n * 32 + c])
                res = re.findall(r'.{8}', s)

                for i in res:
                    f_generate.write('0x' + i + ',\n')

                for i in range(32):
                    f_generate.write('0x' + dec2INT32InHex(b_layer[n * 32 + i]) + ',\n')

                for i in range(32):
                    f_generate.write('0x' + M_list[n * 32 + i] + ',\n')

            f_generate.write('};\n')

            f_generate.write('static const int dw_conv_' + str(int((layer_index - 1) / 2)) + '_shift = ' + str(shift_final) + ';\n')

        elif layer_index % 2 == 0: # pw conv
            f_generate.write("static unsigned int pw_conv_" + str(int((layer_index - 2) / 2)) + "_quant[] = {\n")

            w_layer = tflite_model.get_tensor(layer_pos_list[layer_index][0])
            b_layer = tflite_model.get_tensor(layer_pos_list[layer_index][1])
            M_list, shift_final = get_M_list(tflite_model, layer_pos_list[layer_index][0], shift)

            N = w_layer.shape[0]
            channel = w_layer.shape[3]

            for n in range(int(N / 32)):
                gt = []
                s = ""
                for c in range(int(channel / 32)):
                    tl = []
                    g = []
                    for nn in range(n * 32, n * 32 + 32):
                        for cc in range(c * 32, c * 32 + 32):
                            tl.append(dec2INT8InHex(w_layer[nn, 0, 0, cc]))
                        g = []
                        for i in range(int(len(tl) / 4)):
                            gl = []
                            for temp in range(4):
                                gl.append(tl[i * 4 + temp])
                            gl.reverse()
                            g.append(gl)

                    for gs in g:
                        ts = ""
                        for i in range(4):
                            ts += gs[i]
                        gt.append(ts)

                        # t_r = re.findall(r'.{8}', st)

                for bb in range(n * 32, n * 32 + 32):
                    s += dec2INT32InHex(b_layer[bb])

                for mm in range(n * 32, n * 32 + 32):
                    s += M_list[mm]

                res = re.findall(r'.{8}', s)

                res = gt + res

                for i in res:
                    f_generate.write('0x' + i + ',\n')

            f_generate.write('};\n')

            f_generate.write('static const int pw_conv_' + str(int((layer_index - 1) / 2)) + '_shift = ' + str(shift_final) + ';\n')

    f_generate.close()


def InterFrExp(input: float):
    shift = 0

    pack = struct.pack('d', input)
    u = np.long(struct.unpack('q', pack)[0])
    # u = int(u, 16)

    if (u & ~kSignMask) == 0:
        shift = 0
        return

    exponent_part = ((u & kExponentMask) >> kExponentShift)

    if exponent_part == kExponentIsBadNum:
        shift = 0x7fffffff
        if u & kFractionMask:
            return 0, shift
        else:
            if u & kSignMask:
                return 0, shift
            else:
                return 0x7fffffff, shift

    shift = (exponent_part - kExponentBias) + 1

    fraction = np.long(0)

    fraction += 0x40000000 + ((u & kFractionMask) >> kFractionShift)

    if (u & kFractionRoundingMask) > kFractionRoundingThreshold:
        fraction += 1

    if u & kSignMask:
        fraction *= -1

    return fraction, shift


def conv2d(f_w, f_h, channel, N, w, b, M, shift, fmap, out):
    for n in range(N):
        for f_x in range(0, f_w, 2):
            for f_y in range(0, f_h, 2):
                local_sum = 0
                for c in range(channel):
                    for k_x in range(3):
                        for k_y in range(3):
                            if (f_x + k_x) != -1 and (f_y + k_y) != -1 and (f_x + k_x) < f_w and (f_y + k_y) < f_h:
                                local_sum += w[n][k_x][k_y][c] * fmap[f_x + k_x][f_y + k_y][c]
                            else:
                                local_sum += 0
                local_sum += b[n]
                local_sum = local_sum if local_sum > 0 else 0

                out[n][f_x >> 1][f_y >> 1] = (np.int64(local_sum) * M[n] >> 31) >> (abs(shift[n]))


def dw_conv2d(f_w, f_h, channel, stride, w, b, M, shift, fmap, out):
    if stride == 1:
        for c in range(channel):
            for f_x in range(0, f_w, stride):
                for f_y in range(0, f_h, stride):
                    local_sum = 0
                    for k_x in range(3):
                        for k_y in range(3):
                            if (f_x - 1 + k_x) != -1 and (f_y - 1 + k_y) != -1 and (f_x - 1 + k_x) < f_w and (f_y - 1 + k_y) < f_h:
                                local_sum += w[0][k_x][k_y][c] * fmap[c][f_x - 1 + k_x][f_y - 1 + k_y]
                            else:
                                local_sum += 0
                    local_sum += b[c]
                    local_sum = local_sum if local_sum > 0 else 0

                    out[c][f_x][f_y] = (np.int64(local_sum) * M[c] >> 31) >> (abs(shift[c]))
    elif stride == 2:
        for c in range(channel):
            for f_x in range(0, f_w, stride):
                for f_y in range(0, f_h, stride):
                    local_sum = 0
                    for k_x in range(3):
                        for k_y in range(3):
                            if (f_x + k_x) != -1 and (f_y + k_y) != -1 and (f_x + k_x) < f_w and (f_y + k_y) < f_h:
                                local_sum += w[0][k_x][k_y][c] * fmap[c][f_x + k_x][f_y + k_y]
                            else:
                                local_sum += 0
                    local_sum += b[c]
                    local_sum = local_sum if local_sum > 0 else 0

                    out[c][f_x >> 1][f_y >> 1] = (np.int64(local_sum) * M[c] >> 31) >> (abs(shift[c]))


def pw_conv2d(f_w, f_h, channel, N, w, b, M, shift, fmap, out):
    for n in range(N):
        for f_x in range(f_w):
            for f_y in range(f_h):
                local_sum = 0
                for c in range(channel):
                    local_sum += w[n][0][0][c] * fmap[c][f_x][f_y]

                local_sum += b[n]
                local_sum = local_sum if local_sum > 0 else 0

                out[n][f_x][f_y] = (np.int64(local_sum) * M[n] >> 31) >> (abs(shift[n]))


def get_param(tflite_model: tf.lite.Interpreter, w_index: int, b_index: int):
    w = np.asarray(tflite_model.get_tensor(w_index))
    b = np.asarray(tflite_model.get_tensor(b_index))
    Sx = tflite_model.get_tensor_details()[w_index - 1]['quantization_parameters']['scales'].astype(float)
    Sw_scale_list = tflite_model.get_tensor_details()[w_index]['quantization_parameters']['scales'].astype(float)
    Sa = tflite_model.get_tensor_details()[w_index + 1]['quantization_parameters']['scales'].astype(float)

    M_list = []
    shift_list = []

    for i in Sw_scale_list:
        M, shift = InterFrExp((Sx * i) / Sa)
        M_list.append(M)
        shift_list.append(shift)

    return w, b, M_list, shift_list


def model_invoke(image_array, model_path='./model/newer/MobileNet.tflite'):
    tflite_model = tf.lite.Interpreter(model_path=model_path)
    tflite_model.allocate_tensors()

    o1 = [[[0 for i in range(112)] for j in range(112)] for k in range(1024)]
    o1 = np.asarray(o1)
    o1.astype(np.int64)

    o2 = [[[0 for i in range(112)] for j in range(112)] for k in range(1024)]
    o2 = np.asarray(o2)
    o2.astype(np.int64)

    w, b, M, shift = get_param(tflite_model, 32, 29)
    conv2d(224, 224, 3, 32, w, b, M, shift, image_array, o1)

    w, b, M, shift = get_param(tflite_model, 34, 28)
    dw_conv2d(112, 112, 32, 1, w, b, M, shift, o1, o2)

    w, b, M, shift = get_param(tflite_model, 36, 17)
    pw_conv2d(112, 112, 32, 64, w, b, M, shift, o2, o1)

    w, b, M, shift = get_param(tflite_model, 38, 9)
    dw_conv2d(112, 112, 64, 2, w, b, M, shift, o1, o2)

    w, b, M, shift = get_param(tflite_model, 40, 8)
    pw_conv2d(56, 56, 64, 128, w, b, M, shift, o2, o1)

    w, b, M, shift = get_param(tflite_model, 42, 7)
    dw_conv2d(56, 56, 128, 1, w, b, M, shift, o1, o2)

    w, b, M, shift = get_param(tflite_model, 44, 6)
    pw_conv2d(56, 56, 128, 128, w, b, M, shift, o2, o1)

    w, b, M, shift = get_param(tflite_model, 46, 5)
    dw_conv2d(56, 56, 128, 2, w, b, M, shift, o1, o2)

    w, b, M, shift = get_param(tflite_model, 48, 4)
    pw_conv2d(28, 28, 128, 256, w, b, M, shift, o2, o1)

    w, b, M, shift = get_param(tflite_model, 50, 3)
    dw_conv2d(28, 28, 256, 1, w, b, M, shift, o1, o2)

    w, b, M, shift = get_param(tflite_model, 52, 27)
    pw_conv2d(28, 28, 256, 256, w, b, M, shift, o2, o1)

    w, b, M, shift = get_param(tflite_model, 54, 26)
    dw_conv2d(28, 28, 256, 2, w, b, M, shift, o1, o2)

    w, b, M, shift = get_param(tflite_model, 56, 25)
    pw_conv2d(14, 14, 256, 512, w, b, M, shift, o2, o1)

    w, b, M, shift = get_param(tflite_model, 58, 24)
    dw_conv2d(14, 14, 512, 1, w, b, M, shift, o1, o2)

    w, b, M, shift = get_param(tflite_model, 60, 23)
    pw_conv2d(14, 14, 512, 512, w, b, M, shift, o2, o1)

    w, b, M, shift = get_param(tflite_model, 62, 22)
    dw_conv2d(14, 14, 512, 1, w, b, M, shift, o1, o2)

    w, b, M, shift = get_param(tflite_model, 64, 21)
    pw_conv2d(14, 14, 512, 512, w, b, M, shift, o2, o1)

    w, b, M, shift = get_param(tflite_model, 66, 20)
    dw_conv2d(14, 14, 512, 1, w, b, M, shift, o1, o2)

    w, b, M, shift = get_param(tflite_model, 68, 19)
    pw_conv2d(14, 14, 512, 512, w, b, M, shift, o2, o1)

    w, b, M, shift = get_param(tflite_model, 70, 18)
    dw_conv2d(14, 14, 512, 1, w, b, M, shift, o1, o2)

    w, b, M, shift = get_param(tflite_model, 72, 16)
    pw_conv2d(14, 14, 512, 512, w, b, M, shift, o2, o1)

    w, b, M, shift = get_param(tflite_model, 74, 15)
    dw_conv2d(14, 14, 512, 1, w, b, M, shift, o1, o2)

    w, b, M, shift = get_param(tflite_model, 76, 14)
    pw_conv2d(14, 14, 512, 512, w, b, M, shift, o2, o1)

    w, b, M, shift = get_param(tflite_model, 78, 13)
    dw_conv2d(14, 14, 512, 2, w, b, M, shift, o1, o2)

    w, b, M, shift = get_param(tflite_model, 80, 12)
    pw_conv2d(7, 7, 512, 1024, w, b, M, shift, o2, o1)

    w, b, M, shift = get_param(tflite_model, 82, 11)
    dw_conv2d(7, 7, 1024, 1, w, b, M, shift, o1, o2)

    w, b, M, shift = get_param(tflite_model, 84, 10)
    pw_conv2d(7, 7, 1024, 1024, w, b, M, shift, o2, o1)

    return o1




