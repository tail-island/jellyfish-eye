import tensorflow as tf


def summary_image(name, variable):
    tf.summary.image(name, tf.slice(variable, (0, 0, 0, 0), (1, -1, -1, -1)))
    return variable


def summary_image_collection(name, variable):
    h = tf.shape(variable)[1]
    w = tf.shape(variable)[2]
    r = tf.shape(variable)[3] // 8  # 畳込みの出力チャンネル数は、8の倍数だと仮定します。
    c = 8
    
    image = tf.slice(variable, (0, 0, 0, 0), (1, -1, -1, -1))
    image = tf.reshape(image, (h, w, r, c))
    image = tf.transpose(image, (2, 0, 3, 1))
    image = tf.image.adjust_contrast(image, 1)
    image = tf.reshape(image, (1, r * h, c * w, 1))
    
    tf.summary.image(name, image)
    
    return variable


def summary_scalar(name, variable):
    tf.summary.scalar(name, variable)
    return variable
