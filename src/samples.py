# weights = (4, 2, 2)
# threshold = 3

# def formal_neuron(*xs):
#     return sum(weights[i] for i, x in enumerate(xs) if x) > threshold

####

# from itertools import chain

# def and_(*xs):
#     return sum((1, 1)[i] for i, x in enumerate(xs) if x) > 1.5

# def or_(*xs):
#     return sum((1, 1)[i] for i, x in enumerate(xs) if x) > 0.5

# def not_(*xs):
#     return sum((-1,)[i] for i, x in enumerate(xs) if x) > -0.5

# def xor_(*xs):
#     def next_layer(*next_xs):
#         return sum((1, 1, -2)[i] for i, x in enumerate(next_xs) if x) > 0.5

#     return next_layer(*chain(xs, ((sum((1, 1)[i] for i, x in enumerate(xs) if x) > 1.5),)))

####

# import matplotlib.pyplot as plot
# import matplotlib.animation as animation

# from itertools import starmap
# from random import shuffle


# weights = (0, 1)
# bias = -0.5


# def perceptron(*xs):
#     return 1 if sum(x * weight for x, weight in zip(xs, weights)) + bias >= 0 else 0


# motorcycles = (
#     (1, ( 999, 1501200)),  # ホンダ CBR1000RR（ロスホワイト）
#     (1, ( 999, 1468800)),  # ホンダ CBR1000RR
#     (1, ( 999, 1674000)),  # ホンダ CBR1000RR<ABS>（ロスホワイト）
#     (1, ( 999, 2030400)),  # ホンダ CBR1000RR SP
#     (1, ( 599, 1334880)),  # ホンダ CBR600RR<ABS>（ロスホワイト）
#     (1, ( 599, 1162080)),  # ホンダ CBR600RR（ロスホワイト）
#     (1, ( 599, 1302480)),  # ホンダ CBR600RR<ABS>（グラファイトブラック）
#     (1, ( 599, 1129680)),  # ホンダ CBR600RR（グラファイトブラック）
#     (1, ( 998, 2430000)),  # ヤマハ YZF-R1（ライトれディッシュイエローソリット1）
#     (1, ( 998, 2376000)),  # ヤマハ YZF-R1（ディープパープリッシュブルーメタリックC）
#     (1, ( 998, 3186000)),  # ヤマハ YZF-R1M
#     (1, ( 998, 1782000)),  # カワサキ Ninja ZX-10R
#     (1, ( 599,  924480)),  # カワサキ Ninja ZX-6R
#     (1, ( 999, 1695600)),  # スズキ GSX-R1000
#     (1, ( 999, 1760400)),  # スズキ GSX-R1000 ABS
#     (1, ( 750, 1544400)),  # スズキ GSX-R750
#     (1, ( 599, 1425600)),  # スズキ GSX-R600

#     (0, ( 845, 1069200)),  # ヤマハ MT-09 TRACER ABS
#     (0, ( 745,  743040)),  # ホンダ NC750X
#     (0, ( 745,  793800)),  # ホンダ NC750X<ABS>
#     (0, ( 745,  859680)),  # ホンダ NC750X DCS<ABS>
#     (0, ( 745,  924480)),  # ホンダ NC750X DCS<ABS> E Package
#     (0, ( 845, 1004400)),  # ヤマハ MT-09 ABS
#     (0, ( 998, 1382400)),  # ホンダ CRF1000L（ヴィクトリーレッド、パールグレアホワイト）
#     (0, ( 998, 1350000)),  # ホンダ CRF1000L（キャンディープロミネンスレッド、デジタルシルバーメタリック）
#     (0, ( 998, 1490400)),  # ホンダ CRF1000L DCS（ヴィクトリーレッド、パールグレアホワイト）
#     (0, ( 998, 1458000)),  # ホンダ CRF1000L DCS（キャンディープロミネンスレッド、デジタルシルバーメタリック）
#     (0, ( 998, 1115640)),  # スズキ GSX-S1000 ABS
#     (0, ( 998, 1166400)),  # スズキ GSX-S1000F ABS
#     (0, ( 688,  760320)),  # ヤマハ MT-07 ABS
#     (0, ( 688,  710640)),  # ヤマハ MT-07
#     (0, ( 845, 1042200)),  # ヤマハ XSR900
#     (0, (1164, 1172880)))  # カワサキ ZRX1200 DAEG


# def train():
#     global weights, bias

#     min_xs = tuple(min(starmap(lambda _, xs: xs[i], motorcycles)) for i in range(2))
#     max_xs = tuple(max(starmap(lambda _, xs: xs[i], motorcycles)) for i in range(2))
#     normalized_motorcycles = [(label, tuple((xs[i] - min_xs[i]) / (max_xs[i] - min_xs[i]) for i in range(2))) for label, xs in motorcycles]

#     shuffle(normalized_motorcycles)
#     test_data, train_data = normalized_motorcycles[:5], normalized_motorcycles[5:]

#     learning_rate = 0.01

#     # アニメーション用の変数です。
#     figure = plot.figure()
#     images = []

#     plot.plot([xs[0] for label, xs in train_data if label == 0],
#               [xs[1] for label, xs in train_data if label == 0],
#               'bo',
#               marker='.')
#     plot.plot([xs[0] for label, xs in train_data if label == 1],
#               [xs[1] for label, xs in train_data if label == 1],
#               'ro',
#               marker='.')
#     plot.plot([xs[0] for label, xs in test_data if label == 0],
#               [xs[1] for label, xs in test_data if label == 0],
#               'bo',
#               marker='+')
#     plot.plot([xs[0] for label, xs in test_data if label == 1],
#               [xs[1] for label, xs in test_data if label == 1],
#               'ro',
#               marker='+')

#     for i in range(100):
#         images.append(plot.plot([-(weights[0] / weights[1]) * i - (bias / weights[1]) for i in range(2)], 'g'))

#         for label, xs in train_data:
#             result = perceptron(*xs)
#             if (result != label):
#                 weights = tuple(weights[i] + learning_rate * (label - result) * xs[i] for i in range(2))
#                 bias = bias + learning_rate * (label - result)

#     for label, xs in test_data:
#         print("{0}: {1}".format(label, perceptron(*xs)))

#     artist_animation = animation.ArtistAnimation(figure, images, interval=1, repeat_delay=1000)
#     artist_animation.save('perceptron.gif', writer='imagemagick')

#     plot.show()


# if __name__ == '__main__':
#     train()

# import matplotlib.pyplot as plot  # 動かす前に、pip install matplotlibしておいてください。
# import numpy as np

# from tensorflow.examples.tutorials.mnist import input_data

# train_data_set, validation_data_set, test_data_set = input_data.read_data_sets("MNIST_data/")

# print(len(train_data_set.images))
# print(len(train_data_set.labels))

# images, labels = train_data_set.next_batch(5)

# print(images)
# print(labels)

# for image in images:
#     plot.imshow(np.reshape(image, (28, 28)))
#     plot.show()

# import tensorflow as tf

# a = tf.Variable(1)
# b = tf.Variable(2)
# c = a + b
# d = tf.Variable(3)
# e = c + d

# # print(e)
# # print(e.op)

# with tf.Session() as session:
#     session.run(tf.global_variables_initializer())
    
#     result = session.run(e)
#     print(result)

########

# import tensorflow as tf

# from tensorflow.examples.tutorials.mnist import input_data

# # MNISTデータを取得します。
# train_data_set, validation_data_set, test_data_set = input_data.read_data_sets("MNIST_data/")

# # 画像と正解ラベル、トレーニング中かをを入れる変数を作成します。
# images = tf.placeholder(tf.float32, (None, 784))
# labels = tf.placeholder(tf.int32, (None,))

# # ニューラル・ネットワークを定義します。TensorFlowでは、ニューラル・ネットワークの出力はlogitと呼びます。
# logits = tf.contrib.layers.linear(tf.contrib.layers.fully_connected(images, 128), 10)

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
#         session.run(train, feed_dict={images: images_value, labels: labels_value})

#         # 精度の推移を知るために、100回に一回、訓練データと検証データでの精度を出力します。
#         if i % 100 == 0:
#             print("accuracy of train data: {0}".format(session.run(accuracy, feed_dict={images: images_value, labels: labels_value})))
#             print("accuracy of validation data: {0}".format(session.run(accuracy, feed_dict={images: validation_data_set.images, labels: validation_data_set.labels})))

#     # テスト・データでの精度を出力します。
#     print("accuracy of test data: {0}".format(session.run(accuracy, feed_dict={images: test_data_set.images, labels: test_data_set.labels})))

########

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

# MNISTデータを取得します。
train_data_set, validation_data_set, test_data_set = input_data.read_data_sets("MNIST_data/")

# 画像と正解ラベル、トレーニング中かをを入れる変数を作成します。
images = tf.placeholder(tf.float32, (None, 784))
labels = tf.placeholder(tf.int32, (None,))
is_training = tf.placeholder_with_default(False, ())

# ニューラル・ネットワークを定義します。TensorFlowでは、ニューラル・ネットワークの出力はlogitと呼びます。
logits = tf.reshape(images, (-1, 28, 28, 1))
logits = tf.contrib.layers.conv2d(logits, 32, 5)
logits = tf.contrib.layers.max_pool2d(logits, 2)
logits = tf.contrib.layers.conv2d(logits, 64, 5)
logits = tf.contrib.layers.max_pool2d(logits, 2)
logits = tf.contrib.layers.flatten(logits)
logits = tf.contrib.layers.fully_connected(logits, 128)
logits = tf.contrib.layers.dropout(logits, is_training=is_training)
logits = tf.contrib.layers.linear(logits, 10)

# logitとlabelsの誤差を計算します。TensorFlowでは、lossと呼びます。
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))

# どのように訓練するのかを定義します。lossが小さくなるように、学習率を自動で設定してくれるAdamOptimizerで訓練します。
train = tf.train.AdamOptimizer().minimize(loss)

# 正解率を計算します。
accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, labels, 1), tf.float32))

# セッションを作成します。
with tf.Session() as session:
    # tf.contrib.layersで変数を使用しているので、初期化します。
    session.run(tf.global_variables_initializer())

    # 3000回、訓練します。
    for i in range(3000):
        # 1回の訓練では、100個のデータを使用します。
        images_value, labels_value = train_data_set.next_batch(100)
        session.run(train, feed_dict={images: images_value, labels: labels_value, is_training: True})

        # 精度の推移を知るために、100回に一回、訓練データと検証データでの精度を出力します。
        if i % 100 == 0:
            print("accuracy of train data: {0}".format(session.run(accuracy, feed_dict={images: images_value, labels: labels_value})))
            print("accuracy of validation data: {0}".format(session.run(accuracy, feed_dict={images: validation_data_set.images, labels: validation_data_set.labels})))

    # テスト・データでの精度を出力します。
    print("accuracy of test data: {0}".format(session.run(accuracy, feed_dict={images: test_data_set.images, labels: test_data_set.labels})))
