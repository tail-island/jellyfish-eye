import jellyfish_eye.data_sets as data_sets
import jellyfish_eye.model as model
import os
import tensorflow as tf


train_data_set, test_data_set = data_sets.load()

inputs = tf.placeholder(tf.float32, (None, 100 * 100 * 4))
labels = tf.placeholder(tf.int32, (None,))
is_training = tf.placeholder_with_default(False, ())

logits = model.inference(inputs, is_training)
loss = model.loss(logits, labels)
train = model.train(loss)
accuracy = model.accuracy(logits, labels)

global_step = tf.contrib.framework.get_or_create_global_step()
inc_global_step = tf.assign(global_step, tf.add(global_step, 1))

summary = tf.summary.merge_all()
saver = tf.train.Saver()

if not os.path.exists('./checkpoints'):
    os.mkdir('./checkpoints')

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    summary_writer = tf.summary.FileWriter('./logs')
    checkpoint = tf.train.latest_checkpoint('./checkpoints')
    if checkpoint:
        saver.restore(session, checkpoint)

    while True:
        global_step_value = session.run(global_step)

        inputs_value, labels_value = train_data_set.next_batch(20)
        _, summary_value = session.run((train, summary), feed_dict={inputs: inputs_value, labels: labels_value, is_training: True})

        summary_writer.add_summary(summary_value, global_step_value)
        summary_writer.flush()

        if global_step_value % 10 == 0:
            saver.save(session, './checkpoints/model', global_step=global_step_value)
            print('global step {0:>4d}: train accuracy = {1:.4f}, test accuracy = {2:.4f}.'.format(
                global_step_value,
                session.run(accuracy, feed_dict={inputs: inputs_value, labels: labels_value}),
                session.run(accuracy, feed_dict={inputs: test_data_set.inputs, labels: test_data_set.labels})))

        session.run(inc_global_step)
