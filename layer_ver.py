import tensorflow as tf
import numpy as np
import keras
import tensorflow_model_optimization as tfmot
import platform
import time
import matplotlib.pyplot as plt
from PIL import Image
import os

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

with tfmot.quantization.keras.quantize_scope():
    model = tf.keras.models.load_model('./model/newer/MobileNet_quantized.h5')

# model.summary()

layer = keras.Model(
    inputs=model.input,
    outputs=model.get_layer('quant_re_lu').output
    # outputs=model.get_layer('quant_conv2d').output
    # outputs=model.output
)

# layer2 = keras.Model(
#     inputs=model.input,
#     outputs=model.get_layer('quant_dense').output
# )

# layer.summary()

img = keras.preprocessing.image.load_img('test.jpg')
data = keras.preprocessing.image.img_to_array(img)
plt.imshow(img)
data = np.expand_dims(data, axis=0)

# print(data)

result = layer.predict(data)
result2 = layer2.predict(data)

pre = model.predict(data)

print(result2.shape)
#
# print(data.shape)
# print(data[0,0,:,0])
print(pre)
