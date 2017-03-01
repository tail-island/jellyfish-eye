import numpy as np
import os

from functools import partial
from itertools import chain, repeat, starmap


class DataSet:
    def _shuffle(self):
        indice = np.arange(len(self.inputs))
        np.random.shuffle(indice)

        self.inputs = self.inputs[indice]
        self.labels = self.labels[indice]

    def __init__(self, inputs_collection):
        self.inputs = np.array(tuple(chain.from_iterable(inputs_collection)))
        self.labels = np.array(tuple(chain.from_iterable(starmap(lambda i, inputs: repeat(i, len(inputs)), enumerate(inputs_collection)))))

        self._shuffle()
        self._batch_index = 0

    def next_batch(self, batch_size):
        if self._batch_index + batch_size > len(self.inputs):
            self._shuffle()
            self._batch_index = 0

        start = self._batch_index
        end = self._batch_index = self._batch_index + batch_size

        return self.inputs[start:end], self.labels[start:end]


def load(data_path='./data'):
    def rgbz(line_string):
        return map(float, line_string.split())

    def load_image(path):
        with open(path) as file:
            return tuple(chain.from_iterable(map(rgbz, file)))

    def load_object(object_path):
        return map(load_image, filter(lambda path: '.txt' in path, map(partial(os.path.join, object_path), sorted(os.listdir(object_path)))))

    def load_class(class_path):
        return tuple(chain.from_iterable(map(load_object, map(partial(os.path.join, class_path), sorted(os.listdir(class_path))))))

    def train_and_test(images):
        return images[20:], images[:20]

    return map(DataSet, zip(*map(train_and_test, map(load_class, map(partial(os.path.join, data_path), sorted(os.listdir(data_path)))))))
