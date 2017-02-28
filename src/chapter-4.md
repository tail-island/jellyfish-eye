# プログラムの作成

このようにスゴイ深層学習を、実際のプログラムで試してみたくなりませんか？　私は試してみたくなっちゃったので、今回は、ロボットに一般物体認識させるプログラムを作ってきました。

で、その作ってきたソース・コードは、テンプレートとして使用できるんじゃないかなーと考えてています。本章で紹介するソース・コードを少しだけ修正すれば、皆様が深層学習でやりたいことを実現できるんじゃないかなーと。

## ソース・コード

そんなことを言われても、現物を見なければ信用できない？　その通りですね。ソース・コードは[GitHub](https://github.com/tail-island/jellyfish-eye/)で管理していますので、[Git for Windows](https://git-for-windows.github.io/)をセットアップして、以下のコマンドでダウンロードしてみてください。

```bash
> cd TensorFlow  # 「2. インストール」で作成した、TensorFlow用の仮想環境
> git clone https://github.com/tail-island/jellyfish-eye.git
```

なお、プロジェクト名のjellyfish-eyeは、一部のクラゲは高機能な目を持っていて、脳がないにもかかわらず目で見た情報を処理して活用しているらしいことに由来しています。脳を持たないロボットがセンサーの情報にもとづいて自律行動しちゃうぜって感じですね。

## Pythonのパッケージ

Pythonでは、ファイルがパッケージの単位になります。`./aaa.py`の中から`./bbb.py`に含まれる関数を使う際には、`import bbb`してあげなければならないわけですね。

今回は、あとでコードを活用できるように、複数のファイルに分割しました。その際に、管理が楽になるように専用のパッケージを作成しています。`./xxx/ccc.py`のようにして、`import xxx.ccc`にしたわけですな。

このパッケージ名（ディレクトリ名）は、Pythonの文法に合うようにプロジェクト名の`-`を`_`に置換して、`jellyfish_eye`としました。

```bash
> mkdir jellyfish_eye
```

## データ

私の手持ちのロボットはTurtleBotというロボットで、これには深度センサー（MicrosoftのKinectのような、ドット単位で、センサーから物体までの距離を計測できるもの）が付いています。

＊＊＊絵＊＊＊

この深度センサーのデータを深層学習で処理して、ロボットの前に置かれたものが何かを回答することにしましょう（本稿では紹介しませんけど、実は、指定した物体の前まで移動するロボット用のプログラムも書いています。ロボットがモノを見分けるように見えて気持ち悪いデモをご覧になりたい場合は、声をかけてください）。

ロボットの回答のレベルについても、考えなければなりません。というのも、「ギター」と回答するのか「平沢唯のギー太」と回答するのかで、難しさが変わってくるんですよ。「平沢唯のギー太」で学習して「平沢唯のギー太」を見つけるのは特定物体認識と呼ばれる技術で、実は簡単で面白くない。だから、「平沢唯のギー太」で学習して、学習データには含まれていなかった「中野梓のフェンダーJAPANのムスタング」についても「ギター」と答えてくれる、一般物体認識をやることにしましょう。

以上で方針が決まりましたので、学習用のデータを作成しました。ロボットの上でプログラムを動かして、縦100ドット×横100ドット×(3原色＋距離)のデータを、「カレンダー」と「飲み物」、「ティッシュ・ペーパー」の3種類分、集めてきました（会社の机の周りにあった、適当なもの3種類です）。

データを格納しているディレクトリーの構造は、以下の通り。

```
data
├── calendar
│   ├── DISNEY_カレンダー
│   ├── UNITED_BEES_カレンダー
│   └── software_AG_カレンダー
├── drink
│   ├── GEORGIA_EUROPIAN_香るブラック_400ml
│   ├── KIRIN_真っ赤なベリーのビタミーナ
│   ├── NATIONAL_VENDING_ホット
│   ├── POKKA_SAPPORO_冴えるBLACK_275g
│   ├── POKKA_SAPPORO_冴えるBLACK_400g
│   ├── POKKA_SAPPORO_富士山麓の天然水
│   └── SUNTORY_伊右衛門_1l
└── tissue-paper
    ├── FamilyMart_フェイシャルティシュー
    ├── Kleenex_HIGH_QUALITY_FACIAL_TISSUES
    ├── Kleenex_ハイクオリティ_フェイシャルティシュー
    ├── Kleenex_ローションティシュー_エックス
    └── エリエール_贅沢保湿
```

三階層目のディレクトリーの中に、それぞれの物体のデータが入ります。様々な角度から、1物体につき

データの単位で、ファイルは分割されています（ファイルの拡張子は.txt）。そのファイルの1行は1ドットを表していて「赤<空白>青＜空白＞緑＜空白＞距離」となっています（それぞれのデータの範囲は、0.0〜1.0）。最初の行は左上のドット、次の行はその右のドットと続き、最も右下のドットまで、10000行続いています。

## jellyfish\_eye/data\_sets.py

今回は独自のデータを使用していますから、前章のMNISTの場合のような便利クラスはありません。だから、データ管理のコードを作成しました。

```python
import numpy as np
import os

from functools import partial
from itertools import chain, islice, repeat, starmap, tee


# データ管理用のクラスです。
class DataSet:
    # データをシャッフルします。
    def _shuffle(self):
        indice = np.arange(len(self.inputs))
        np.random.shuffle(indice)

        self.inputs = self.inputs[indice]
        self.labels = self.labels[indice]

    # コンストラクタ。inputs_collectionは、[[クラス0のデータ, クラス0のデータ...], [クラス1のデータ, クラス1のデータ], ...]としてください。
    def __init__(self, inputs_collection):
        self.inputs = np.array(tuple(chain.from_iterable(inputs_collection)))
        self.labels = np.array(tuple(chain.from_iterable(starmap(lambda i, inputs: repeat(i, len(inputs)), enumerate(inputs_collection)))))

        self._shuffle()
        self._batch_index = 0

    # バッチ学習用のデータを取得します。
    def next_batch(self, batch_size):
        if self._batch_index + batch_size > len(self.inputs):
            self._shuffle()
            self._batch_index = 0

        start = self._batch_index
        end = self._batch_index = self._batch_index + batch_size

        return self.inputs[start:end], self.labels[start:end]


# データを読み込みます。
def load(data_path='./data'):
    # データを1行分、読み込みます。
    def rgbz(line_string):
        # 赤と緑と青と距離に分割し、数値化します。
        return map(float, line_string.split())
    
    # データを1つ、読み込みます。
    def load_image(path):
        # 一行ずつ読み込んで、chain.from_iterableでフラットにします。
        with open(path) as file:
            return tuple(chain.from_iterable(map(rgbz, file)))
    
    # 物体のデータを読み込みます。
    def load_object(object_path):
        # 様々な角度からのデータを読み込みます。
        return map(load_image, filter(lambda path: '.txt' in path, map(partial(os.path.join, object_path), sorted(os.listdir(object_path)))))
        
    # クラス（カレンダーや飲み物）を読み込みます。
    def load_class(class_path):
        # 物体単位でデータを読み込んで、chain.from_iterableでフラットにします。
        return tuple(chain.from_iterable(map(load_object, map(partial(os.path.join, class_path), sorted(os.listdir(class_path))))))

    # 訓練データとテスト・データに分割します。
    def train_and_test(images):
        # 25番目以降を訓練データ、それより前をテスト・データとします。
        return images[25:], images[:25]

    # クラス（カレンダーや飲み物）単位でデータを読み込んで、訓練データとテスト・データに分割し、データ・セットのクラスを生成します。
    return map(DataSet, zip(*map(train_and_test, map(load_class, map(partial(os.path.join, data_path), sorted(os.listdir(data_path)))))))
```

上半分の`class DataSet`は、深層学習を使う目的がクラス分類ならば、どのようなデータでも使えると思います。お好きにコピー＆ペーストしてみてください。`_shuffle()`等では、NumPyの機能をフルに使っています。

下半分の`load()`は、対象データによって異なりますので、コピー＆ペーストは駄目。他のデータを扱う場合は、書き直さなければなりません。書き直しの負荷を減らすにはソース・コードの量を減らすのが有効で、そのためには関数型プログラミングが効く。というわけで、Pythonの公式文書の[関数型プログラミング HOWTO](https://docs/python.jp/3/howto/functional.html)を読んで、関数型で書いてみました。[関数型プログラミング HOWTO](https://docs/python.jp/3/howto/functional.html)は非常にわかりやすいので、一読すれば、上のコードのような感じに関数型プログラミングできます。

## jellyfish\_eye/model.py

次は、ニューラル・ネットワークを定義することにしましょう。以下がそのコード。

```python
import tensorflow as tf

from jellyfish_eye.utilities import summary_image, summary_image_collection, summary_scalar

# 必要なplaceholderを定義します。
inputs = tf.placeholder(tf.float32, (None, 100 * 100 * 4))
labels = tf.placeholder(tf.int32, (None,))
is_training = tf.placeholder_with_default(False, ())


# TensorFlowのドキュメントに、ニューラル・ネットワークを作成する関数名はinference（推論）にしろと書いてありました。なのでinference()。
def inference():
    # summary_image()とsummary_image_collection()は、次の項で説明します。ログを取ると思ってください。

    # 畳み込みできるように、データの形を変更します。
    outputs = summary_image('input', tf.reshape(inputs, (-1, 100, 100, 4)))
    
    # 畳み込みとプーリング（1層目）
    outputs = summary_image_collection('convolution-1', tf.contrib.layers.max_pool2d(tf.contrib.layers.conv2d(outputs, 32, 10), 2))
    
    # 畳み込みとプーリング（2層目）
    outputs = summary_image_collection('convolution-2', tf.contrib.layers.max_pool2d(tf.contrib.layers.conv2d(outputs, 64, 10), 2))
    
    # 全結合層で処理できるように、データの形を変更します。
    outputs = tf.contrib.layers.flatten(outputs)
    
    # 全結合層。1024ニューロンと512ニューロンの2層。
    outputs = tf.contrib.layers.stack(outputs, tf.contrib.layers.fully_connected, (1024, 512))
    
    # ドロップアウト。
    outputs = tf.contrib.layers.dropout(outputs, is_training=is_training)

    # 3クラスに分類します。
    return tf.contrib.layers.linear(outputs, 3)


# 損失関数。
def loss(logits):
    # summary_scalarは、次の項で説明します。ログを取るんだと思ってください。
    
    # クロス・エントロピーで、損失を計算します。
    return summary_scalar('loss', tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)))


# 訓練。
def train(loss):
    # 学習率の設定が不要なAdamOptimizerを使用して訓練します。
    return tf.train.AdamOptimizer().minimize(loss)


# 精度。
def accuracy(logits):
    return tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, labels, 1), tf.float32))
```

前章でやった畳み込みニューラル・ネットワークのコードから、ニューラル・ネットに関係する部分を抜き出して、関数化しただけですね。

で、これらの処理を別パッケージに抜き出した理由は、変更が多い部分を局所化するためです。深層学習してみたのだけど精度が低いという場合は、ニューラル・ネットワークをチューニングしなければなりません。で、チューニングで何をするかというと、`max_pool2d()`や`conv2d()`、`fully_connected()`の引数を変更することになります。畳み込みで出力されるチャンネル数を増やしたりとかね。あと、畳込み層を増やしちゃうなんてのもよくやります。これらのチューニング作業が、この`model.py`の修正だけで済むというわけ。

## jellyfish\_eye/utilities.py

