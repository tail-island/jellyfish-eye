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

import matplotlib.pyplot as plot
import matplotlib.animation as animation

from itertools import starmap
from random import shuffle


weights = (0, 1)
bias = -0.5


def perceptron(*xs):
    return 1 if sum(x * weight for x, weight in zip(xs, weights)) + bias >= 0 else 0


motorcycles = (
    (1, ( 999, 1501200)),  # ホンダ CBR1000RR（ロスホワイト）
    (1, ( 999, 1468800)),  # ホンダ CBR1000RR
    (1, ( 999, 1674000)),  # ホンダ CBR1000RR<ABS>（ロスホワイト）
    (1, ( 999, 2030400)),  # ホンダ CBR1000RR SP
    (1, ( 599, 1334880)),  # ホンダ CBR600RR<ABS>（ロスホワイト）
    (1, ( 599, 1162080)),  # ホンダ CBR600RR（ロスホワイト）
    (1, ( 599, 1302480)),  # ホンダ CBR600RR<ABS>（グラファイトブラック）
    (1, ( 599, 1129680)),  # ホンダ CBR600RR（グラファイトブラック）
    (1, ( 998, 2430000)),  # ヤマハ YZF-R1（ライトれディッシュイエローソリット1）
    (1, ( 998, 2376000)),  # ヤマハ YZF-R1（ディープパープリッシュブルーメタリックC）
    (1, ( 998, 3186000)),  # ヤマハ YZF-R1M
    (1, ( 998, 1782000)),  # カワサキ Ninja ZX-10R
    (1, ( 599,  924480)),  # カワサキ Ninja ZX-6R
    (1, ( 999, 1695600)),  # スズキ GSX-R1000
    (1, ( 999, 1760400)),  # スズキ GSX-R1000 ABS
    (1, ( 750, 1544400)),  # スズキ GSX-R750
    (1, ( 599, 1425600)),  # スズキ GSX-R600

    (0, ( 845, 1069200)),  # ヤマハ MT-09 TRACER ABS
    (0, ( 745,  743040)),  # ホンダ NC750X
    (0, ( 745,  793800)),  # ホンダ NC750X<ABS>
    (0, ( 745,  859680)),  # ホンダ NC750X DCS<ABS>
    (0, ( 745,  924480)),  # ホンダ NC750X DCS<ABS> E Package
    (0, ( 845, 1004400)),  # ヤマハ MT-09 ABS
    (0, ( 998, 1382400)),  # ホンダ CRF1000L（ヴィクトリーレッド、パールグレアホワイト）
    (0, ( 998, 1350000)),  # ホンダ CRF1000L（キャンディープロミネンスレッド、デジタルシルバーメタリック）
    (0, ( 998, 1490400)),  # ホンダ CRF1000L DCS（ヴィクトリーレッド、パールグレアホワイト）
    (0, ( 998, 1458000)),  # ホンダ CRF1000L DCS（キャンディープロミネンスレッド、デジタルシルバーメタリック）
    (0, ( 998, 1115640)),  # スズキ GSX-S1000 ABS
    (0, ( 998, 1166400)),  # スズキ GSX-S1000F ABS
    (0, ( 688,  760320)),  # ヤマハ MT-07 ABS
    (0, ( 688,  710640)),  # ヤマハ MT-07
    (0, ( 845, 1042200)),  # ヤマハ XSR900
    (0, (1164, 1172880)))  # カワサキ ZRX1200 DAEG


def train():
    global weights, bias

    min_xs = tuple(min(starmap(lambda _, xs: xs[i], motorcycles)) for i in range(2))
    max_xs = tuple(max(starmap(lambda _, xs: xs[i], motorcycles)) for i in range(2))
    normalized_motorcycles = [(label, tuple((xs[i] - min_xs[i]) / (max_xs[i] - min_xs[i]) for i in range(2))) for label, xs in motorcycles]

    shuffle(normalized_motorcycles)
    test_data, train_data = normalized_motorcycles[:5], normalized_motorcycles[5:]

    learning_rate = 0.01

    # アニメーション用の変数です。
    figure = plot.figure()
    images = []

    plot.plot([xs[0] for label, xs in train_data if label == 0],
              [xs[1] for label, xs in train_data if label == 0],
              'bo',
              marker='.')
    plot.plot([xs[0] for label, xs in train_data if label == 1],
              [xs[1] for label, xs in train_data if label == 1],
              'ro',
              marker='.')
    plot.plot([xs[0] for label, xs in test_data if label == 0],
              [xs[1] for label, xs in test_data if label == 0],
              'bo',
              marker='+')
    plot.plot([xs[0] for label, xs in test_data if label == 1],
              [xs[1] for label, xs in test_data if label == 1],
              'ro',
              marker='+')

    for i in range(100):
        images.append(plot.plot([-(weights[0] / weights[1]) * i - (bias / weights[1]) for i in range(2)], 'g'))

        for label, xs in train_data:
            result = perceptron(*xs)
            if (result != label):
                weights = tuple(weights[i] + learning_rate * (label - result) * xs[i] for i in range(2))
                bias = bias + learning_rate * (label - result)

    for label, xs in test_data:
        print("{0}: {1}".format(label, perceptron(*xs)))

    artist_animation = animation.ArtistAnimation(figure, images, interval=1, repeat_delay=1000)
    artist_animation.save('perceptron.gif', writer='imagemagick')

    plot.show()


if __name__ == '__main__':
    train()
