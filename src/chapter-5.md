# おわりに

ほら、深層学習って簡単でしょ？　本稿のコードをコピー＆ペーストして、データを取得する部分を書き直して、必要に応じてニューラル・ネットワークをチューニングするだけでよいんですから。

多層パーセプトロンや畳み込み、プーリング、ドロップアウトという深層学習で使われる道具についても、数学なしでその概念を理解できました[^7]。概念が分かっているのですから、ニューラル・ネットワークのチューニングは可能なはず[^8]。逆誤差伝播学習法の詳細になると数学が必要なのですけど、TensorFlowが全自動でやってくれるから詳細を理解していなくても問題ないしね[^9]。ほらやっぱり、深層学習って簡単でしょ？

でも、深層学習って画像が何なのかを判断できるるだけなのでは？　データが画像以外だったらどうするの……という皆様からの反論が聞こえてきそうですけど、深層学習は画像以外でも使えるのでご安心ください。私は先日、ひょんなことから「トイレの個室でのスマートフォン使用を、超音波センサー10個の距離情報（トイレで動画を撮るわけにはいきませんもんね）から検知する」システムのプロトタイプを作ったのですけど、適当に取った数人×数分のデータを、以下のニューラル・ネットワークに食わせて学習してみたら、70%を超える精度が出ちゃいました。遊びでやったので作業は止まってますけど、データを増やせば、実用になるんじゃないかな。

```python
def inference():
    outputs = tf.reshape(inputs, (-1, data_sets.history_size, 1, data_sets.channel_size))  # 幅を1にして、1次元データをconvolution2dできるようにします。
    outputs_collection = [tf.contrib.layers.max_pool2d(tf.contrib.layers.convolution2d(outputs, 128, (kernel_size, 1), padding='VALID'), (data_sets.history_size - kernel_size + 1, 1)) for kernel_size in (5, 4, 3, 2)]
    outputs = tf.concat(1, [tf.contrib.layers.flatten(outputs) for outputs in outputs_collection])  # 次元0はバッチなので、次元1でconcatします。
    outputs = tf.contrib.layers.stack(outputs, tf.contrib.layers.fully_connected, (1024, 512))
    outputs = tf.contrib.layers.dropout(outputs, is_training=is_training)

    return tf.contrib.layers.linear(outputs, 2)
```

ちなみに、このニューラル・ネットワークは、「[自然言語処理における畳み込みニューラルネットワークを理解する](http://tkengo.github.io/blog/2016/03/11/understanding-convolutional-neural-networks-for-nlp/)」で引用されている論文（hang, Y., & Wallace, B. (2015). A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification）の、文章をカケラも読まずに図だけを見て作ったモノです。深層学習は道具立てが簡単なので、けっこう楽ちんに小難しい論文の内容を活用できちゃうんですな。ほら、やっぱり、深層学習は簡単だ。

ぜひ、皆様も深層学習してみてください。楽しいっすよ。


[^7]: 再帰型ニューラル・ネットワークについての説明は、ごめんなさい、使い方が定まるまで待ってください……。
[^8]: 数学的に理解している人たちも、結構行き当たりばったりでチューニングしているみたいだしね。
[^9]: 少なくとも私は、本稿に書いたレベルしか理解できていません。でもだいじょーぶ。
