import jellyfish_eye.data_sets as data_sets
import jellyfish_eye.model as model
import tensorflow as tf


train_data_set, validation_data_set, _ = data_sets.load()

inputs = model.inputs
labels = model.labels
is_training = model.is_training

logits = model.inference()
train = model.train(model.loss(logits))
accuracy = model.accuracy(logits)

global_step = tf.contrib.framework.get_or_create_global_step()
inc_global_step = tf.assign(global_step, tf.add(global_step, 1))
summary = tf.summary.merge_all()
supervisor = tf.train.Supervisor(logdir='logs', save_model_secs=60, save_summaries_secs=60, summary_op=None)

with supervisor.managed_session() as session:
    while True:  # not supervisor.should_stop():が正しいはずなのですけど、なぜかWindows環境では動かなかった……。
        global_step_value = session.run(global_step)

        train_inputs, train_labels = train_data_set.next_batch(20)
        session.run(train, feed_dict={inputs: train_inputs, labels: train_labels, is_training: True})

        if global_step_value % 10 == 0:
            supervisor.summary_computed(session, session.run(summary, feed_dict={inputs: train_inputs, labels: train_labels, is_training: True}))

            print('global step {0:>4d}: train accuracy = {1:.4f}, validation accuracy = {2:.4f}.'.format(
                global_step_value,
                session.run(accuracy, feed_dict={inputs: train_inputs, labels: train_labels}),
                session.run(accuracy, feed_dict={inputs: validation_data_set.inputs, labels: validation_data_set.labels})))

        session.run(inc_global_step)
