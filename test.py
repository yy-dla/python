
import matplotlib.pyplot as plt
import tensorflow as tf
# import keras
from tensorflow import keras
import numpy as np
import tensorflow_model_optimization as tfmot
from PIL import Image
import time
from utils import *


# img = Image.open('test.jpg')
# array2 = np.asarray(img)
#
# o = model_invoke(array2)


convert_h5_to_c_quant('model/newer/MobileNet.tflite', shift=31)

# test_image = np.expand_dims(Image.open('test.jpg'), axis=0).astype(np.float32)
#
# with open('model/newer/MobileNet.tflite', 'rb') as f:
#     model_buffer = f.read()
#
# model_buffer = buffer_change_output_tensor_to(model_buffer, 90)
#
# inter = tf.lite.Interpreter(model_content=model_buffer)
# inter.allocate_tensors()
#
# input_index = inter.get_input_details()[0]['index']
# output_index = inter.get_output_details()[0]['index']
#
# inter.set_tensor(input_index, test_image)
# inter.invoke()
#
# pred = inter.get_tensor(output_index)
#
# layer_index = 32
#
# Sx = float(inter.get_tensor_details()[layer_index - 1]['quantization_parameters']['scales'][0])
# # Sw = 2.5518790e-05
# Sa = float(inter.get_tensor_details()[layer_index + 1]['quantization_parameters']['scales'][0])
#
# n = 31
# l = []
# Sw_list = []
# shift_list = []
# fraction_list = []
# fraction_after_shift = []
# final_shift = 0
# M_list = []
#
# for i in inter.get_tensor_details()[layer_index]['quantization_parameters']['scales']:
#     Sw_list.append(float(i))
#
# for Sw in Sw_list:
#     M = (Sx * Sw) / Sa
#     M_list.append(M)
#     t = InterFrExp(M)
#     fraction_list.append(t[0])
#     shift_list.append(t[1])
#
# shift_min = max(shift_list)
# final_shift = n + abs(shift_min)
#
# for i in range(len(fraction_list)):
#     if shift_list[i] != shift_min:
#         t_shift = abs(shift_list[i]) - abs(shift_min)
#         fraction_after_shift.append(fraction_list[i] >> t_shift)
#     else:
#         fraction_after_shift.append(fraction_list[i])
#
# for i in range(len(fraction_list)):
#     print(hex(fraction_list[i]), hex(fraction_after_shift[i]), dec2INT32InHex(fraction_after_shift[i]), M_list[i])

# l = [
#     0.00096228503   ,
#     0.000549665     ,
#     0.0016514199    ,
#     0.00077855925   ,
#     0.0021553254    ,
#     0.0007260751    ,
#     0.00055243896   ,
#     0.0014726268    ,
#     0.000580271     ,
#     0.0010022782    ,
#     0.0006962323    ,
#     0.0011823799    ,
#     0.00045308666   ,
#     0.0010650893    ,
#     0.00093918375   ,
#     0.0004825684    ,
#     0.00087372446   ,
#     0.0005584907    ,
#     0.0011186119    ,
#     0.0009384513    ,
#     0.0008300605    ,
#     0.00066576165   ,
#     0.0010732117    ,
#     0.00063461985   ,
#     0.00043228443   ,
#     0.0012655397    ,
#     0.00071948784   ,
#     0.00091362663   ,
#     0.0006244027    ,
#     0.0008822114    ,
#     0.0012508034    ,
#     0.00057771325   ,
# ]
# for i in l:
#     t = InterFrExp(i)
#     print(hex(t[0]), t[1])


