# https://blog.csdn.net/u014061630/article/details/80557028
import tensorflow as tf

model = ('/home/hp/utensor_cgen/tests/deep_cnn/cifar10_cnn.pb') 
graph = tf.get_default_graph()
graph_def = graph.as_graph_def()
graph_def.ParseFromString(tf.gfile.FastGFile(model, 'rb').read())
tf.import_graph_def(graph_def, name='graph')
summaryWriter = tf.summary.FileWriter('/home/hp/log/', graph)
