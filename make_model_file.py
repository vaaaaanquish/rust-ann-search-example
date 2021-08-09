# convert resnet50 -> embedding_model
#
# Reference
# https://leimao.github.io/blog/Save-Load-Inference-From-TF2-Frozen-Graph/
# https://zenn.dev/dskkato/articles/tf2-rust-python

import tensorflow as tf
from keras.models import Model
from tensorflow.python.framework.convert_to_constants import \
    convert_variables_to_constants_v2

model = tf.keras.applications.resnet50.ResNet50(weights='imagenet')
embedding_model = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
resnet = tf.TensorSpec(embedding_model.input_shape, tf.float32, name="resnet")
concrete_function = tf.function(lambda x: embedding_model(x)).get_concrete_function(resnet)
frozen_model = convert_variables_to_constants_v2(concrete_function)
tf.io.write_graph(frozen_model.graph, '/app/model', "model.pb", as_text=False)
