from utils import convert_h5_to_c, generate_c_array, generate_c_define, generate_bn_define, generate_bn_init_array
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import os

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

with tfmot.quantization.keras.quantize_scope():
    model = tf.keras.models.load_model('./model/newer/MobileNet_quantized_weights.h5')

layers_name_list = []
for i in model.get_config()['layers']:
    layers_name_list.append(i['config']['name'])

# print(layers_name_list)

conv_layer_name = [j for j in layers_name_list if (j.find('conv2d') != -1 and j.find('conv2d_input') == -1) or j.find('batch_normalization') != -1 or j.find('dense') != -1]

# print(model.get_layer('quant_dense').get_weights()[1].shape)

print(conv_layer_name)

convert_h5_to_c(conv_layer_name, model)
# generate_c_array(conv_layer_name, model)
# generate_c_define(conv_layer_name, model)\
# generate_bn_define(layers_name_list, model)
# generate_bn_init_array(layers_name_list, model)