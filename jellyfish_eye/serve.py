import jellyfish_eye.model as model
import numpy as np
import tensorflow as tf


inputs = tf.placeholder(tf.float32, (None, 100 * 100 * 4))
is_training = tf.placeholder_with_default(False, ())

top_k = tf.nn.top_k(tf.nn.softmax(model.inference(inputs, is_training)), k=3)

supervisor = tf.train.Supervisor(logdir='logs', save_model_secs=0, save_summaries_secs=0)
session = supervisor.PrepareSession()


def classify(input):
    return session.run(top_k, feed_dict={inputs: np.array((input,))})


if __name__ == '__main__':
    import jellyfish_eye.data_sets as data_sets
    import time
    
    _, test_data_set = data_sets.load()
    classes = []

    starting_time = time.time()
    for input in test_data_set.inputs:
        classes.append(classify(input))
    finishing_time = time.time()

    for label, class_ in zip(test_data_set.labels, classes):
        print('{0} : {1}, {2}'.format(label, class_.indices[0][0], tuple(zip(class_.indices[0], class_.values[0]))))
        
    print('Elapsed time: {0:.4f} sec'.format((finishing_time - starting_time) / len(test_data_set.inputs)))
