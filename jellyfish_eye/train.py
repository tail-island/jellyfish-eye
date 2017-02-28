import jellyfish_eye.data_sets as data_sets
import jellyfish_eye.model as model
import tensorflow as tf


train_data_set, test_data_set = data_sets.load()

inputs = tf.placeholder(tf.float32, (None, 100 * 100 * 4))
labels = tf.placeholder(tf.int32, (None,))
is_training = tf.placeholder_with_default(False, ())

logits = model.inference(inputs, is_training)
train = model.train(model.loss(logits, labels))
accuracy = model.accuracy(logits, labels)

global_step = tf.contrib.framework.get_or_create_global_step()
inc_global_step = tf.assign(global_step, tf.add(global_step, 1))
summary = tf.summary.merge_all()
supervisor = tf.train.Supervisor(logdir='logs', save_model_secs=60, save_summaries_secs=60, summary_op=None)

with supervisor.managed_session() as session:
    while not supervisor.should_stop():
        global_step_value = session.run(global_step)

        train_inputs, train_labels = train_data_set.next_batch(20)
        session.run(train, feed_dict={inputs: train_inputs, labels: train_labels, is_training: True})
        
        if global_step_value % 10 == 0:
            supervisor.summary_computed(session, session.run(summary, feed_dict={inputs: train_inputs, labels: train_labels, is_training: True}))
            
            print('global step {0:>4d}: train accuracy = {1:.4f}, test accuracy = {2:.4f}.'.format(
                global_step_value,
                session.run(accuracy, feed_dict={inputs: train_inputs, labels: train_labels}),
                session.run(accuracy, feed_dict={inputs: test_data_set.inputs, labels: test_data_set.labels})))

        session.run(inc_global_step)
