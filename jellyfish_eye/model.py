import tensorflow as tf

from jellyfish_eye.utilities import summary_image, summary_image_collection, summary_scalar


def inference(inputs, is_training):
    outputs = summary_image('input', tf.reshape(inputs, (-1, 100, 100, 4)))
    outputs = summary_image_collection('convolution-1', tf.contrib.layers.max_pool2d(tf.contrib.layers.conv2d(outputs, 32, 10), 2))
    outputs = summary_image_collection('convolution-2', tf.contrib.layers.max_pool2d(tf.contrib.layers.conv2d(outputs, 64, 10), 2))
    outputs = tf.contrib.layers.flatten(outputs)
    outputs = tf.contrib.layers.stack(outputs, tf.contrib.layers.fully_connected, (1024, 512))
    outputs = tf.contrib.layers.dropout(outputs, is_training=is_training)

    return tf.contrib.layers.linear(outputs, 3)


def loss(logits, labels):
    return summary_scalar('loss', tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)))


def train(loss):
    return tf.train.AdamOptimizer().minimize(loss)


def accuracy(logits, labels):
    return tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, labels, 1), tf.float32))
