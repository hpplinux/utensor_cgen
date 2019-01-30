import tensorflow as tf
model = '/home/hp/utensor_cgen/tests/deep_cnn/cifar10_cnn.pb'
with tf.Session() as sess:
    with open(model, 'rb') as model_file:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(model_file.read())
        print(graph_def)
