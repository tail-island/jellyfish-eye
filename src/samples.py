# import tensorflow as tf

# from tensorflow.examples.tutorials.mnist import input_data

# # MNISTデータを取得します。
# train_data_set, validation_data_set, test_data_set = input_data.read_data_sets("MNIST_data/")

# # 画像と正解ラベル、トレーニング中かをを入れる変数を作成します。
# images = tf.placeholder(tf.float32, (None, 784))
# labels = tf.placeholder(tf.int32, (None,))
# is_training = tf.placeholder_with_default(False, ())

# # ニューラル・ネットワークを定義します。TensorFlowでは、ニューラル・ネットワークの出力はlogitと呼びます。
# logits = tf.reshape(images, (-1, 28, 28, 1))
# logits = tf.contrib.layers.conv2d(logits, 32, 5)
# logits = tf.contrib.layers.max_pool2d(logits, 2)
# logits = tf.contrib.layers.conv2d(logits, 64, 5)
# logits = tf.contrib.layers.max_pool2d(logits, 2)
# logits = tf.contrib.layers.flatten(logits)
# logits = tf.contrib.layers.fully_connected(logits, 128)
# logits = tf.contrib.layers.dropout(logits, is_training=is_training)
# logits = tf.contrib.layers.linear(logits, 10)

# # logitとlabelsの誤差を計算します。TensorFlowでは、lossと呼びます。
# loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))

# # どのように訓練するのかを定義します。lossが小さくなるように、学習率を自動で設定してくれるAdamOptimizerで訓練します。
# train = tf.train.AdamOptimizer().minimize(loss)

# # 正解率を計算します。
# accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, labels, 1), tf.float32))

# # セッションを作成します。
# with tf.Session() as session:
#     # tf.contrib.layersで変数を使用しているので、初期化します。
#     session.run(tf.global_variables_initializer())

#     # 3000回、訓練します。
#     for i in range(3000):
#         # 1回の訓練では、100個のデータを使用します。
#         images_value, labels_value = train_data_set.next_batch(100)
#         session.run(train, feed_dict={images: images_value, labels: labels_value, is_training: True})

#         # 精度の推移を知るために、100回に一回、訓練データと検証データでの精度を出力します。
#         if i % 100 == 0:
#             print("accuracy of train data: {0}".format(session.run(accuracy, feed_dict={images: images_value, labels: labels_value})))
#             print("accuracy of validation data: {0}".format(session.run(accuracy, feed_dict={images: validation_data_set.images, labels: validation_data_set.labels})))

#     # テスト・データでの精度を出力します。
#     print("accuracy of test data: {0}".format(session.run(accuracy, feed_dict={images: test_data_set.images, labels: test_data_set.labels})))


import matplotlib.pyplot as plot
import scipy as sp

xs = sp.arange(-10, 10, 0.1)
ys = [max(x, 0) for x in xs]

plot.plot(xs, ys)
plot.ylim(-10, 10)
plot.show()
