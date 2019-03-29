#@title datasets_tutorials { display-mode: "both" }
# conding: utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow_datasets as tfds
import numpy as np
import os

# def format_tran(tfdata, batch_size=32):
#     batch_tfdata = tfdata.shuffle(1).batch(batch_size)
#     batch_imgs = tfds.as_numpy(batch_tfdata).__next__()['image']
#     batch_labels = tfds.as_numpy(batch_tfdata).__next__()['label']
#     return batch_imgs, batch_labels

tf.logging.set_verbosity(tf.logging.ERROR)

tf.app.flags.DEFINE_float('learning_rate', 3e-4, "learning rate, default is 0.0003")
tf.app.flags.DEFINE_float('scale_factor', 3000, "scale factor of l2 regularization, default is 3000")

tf.app.flags.DEFINE_integer('batch_size', 256, 'batch size, default is 256')
tf.app.flags.DEFINE_integer('num_epochs', 3000, 'number of epochs, default is 3000')

tf.app.flags.DEFINE_string('event_path', os.path.dirname(os.path.abspath(__file__)) + '/Tensorboard', 'Directory where event logs are written to')
tf.app.flags.DEFINE_string('checkpoint_path', os.path.dirname(os.path.abspath(__file__)) + '/Checkpoints', 'Directory where checkpoints are written to')
tf.app.flags.DEFINE_boolean('online_test', False, 'online test, default is False')
# tf.app.flags.DEFINE_boolean()

FLAGS = tf.app.flags.FLAGS


if __name__ == '__main__':

    filepath = '/content/sample_data/MNIST_data'
    # filepath = r'E:\Anaconda2\Programs\MNIST_data'
    mnist = input_data.read_data_sets(filepath, one_hot=False)
    # mnist = input_data.read_data_sets('/content/sample_data/MNIST', one_hot=True)
    imgs_train, labels_train = mnist.train.images, mnist.train.labels
    imgs_test, labels_test = mnist.test.images, mnist.test.labels
    num_samples = labels_test.shape[0]
    assert imgs_train.max() == 1., "warning: 'The value of the pixel is preferably between 0 and 1.'"


    graph = tf.Graph()
    with graph.as_default():
        
        # 指数衰减型 learning_rate，衰减因子 0.9，衰减步数 100
        global_step = tf.Variable(0, name='global_step', trainable=False)
        decay_steps = 100
        decay_rate = 0.95
        learning_rate = tf.train.exponential_decay(FLAGS.learning_rate,
                                                    global_step=global_step,
                                                    decay_steps=decay_steps,
                                                    decay_rate=decay_rate,
                                                    staircase=True,
                                                    name='learning_rate')
        
        # 
        x_p = tf.placeholder(tf.float32, shape=[None, 784], name='input_images')
        y_p = tf.placeholder(tf.int64, shape=[None, ], name='labels')
        batch_size = tf.placeholder(tf.int64, name='batch_size')
        # is_training = tf.placeholder(tf.bool, name='is_training')
        data = tf.data.Dataset.from_tensor_slices((x_p, y_p))
        # if is_training is not None:
        #     data = data.repeat()
        #     data_batch = data.shuffle(2).batch(FLAGS.batch_size).prefetch(FLAGS.batch_size)
        # else:
        #     num_samples = x_p.get_shape().as_list()[0]
        #     data_batch = data.batch(num_samples).prefetch(num_samples)
        data = data.repeat()
        data_batch = data.shuffle(2).batch(batch_size).prefetch(batch_size)
        iterator = data_batch.make_initializable_iterator()

        with tf.name_scope('Input'):
            x_input, y_input = iterator.get_next()
            y = tf.one_hot(y_input, depth=10)
            keep_pro = tf.placeholder(tf.float32)
            x_imgs = tf.reshape(x_input, shape=[-1, 28, 28, 1], name='input_images')
        with tf.name_scope('Conv1'):
            w_1 = tf.Variable(tf.truncated_normal([3, 3, 1, 32], stddev=0.1), name='weights_conv1')
            b_1 = tf.Variable(tf.constant(0.1, shape=[32]), name='bias_conv1')
            h_conv1 = tf.nn.relu(tf.nn.conv2d(x_imgs, w_1, strides=[1, 1, 1, 1], padding='SAME') + b_1)
            h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        with tf.name_scope('Conv2'):
            w_2 = tf.Variable(tf.truncated_normal([3, 3, 32, 64], stddev=0.1), name='weights_conv2')
            b_2 = tf.Variable(tf.constant(0.1, shape=[64]), name='bias_conv2')
            h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, w_2, strides=[1, 1, 1, 1], padding='SAME') + b_2)
            h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            # layer_shape = h_pool2.get_shape().as_list()
            # num_f = reduce(lambda a,b:a * b, layer_shape[1:])
            # h_pool2_fla = tf.reshape(h_pool2, shape=[-1, num_f])
        with tf.name_scope('FC1'):
            h_pool2_fla = tf.layers.flatten(h_pool2)
            num_f = h_pool2_fla.get_shape().as_list()[-1]
        
            w_fc1 = tf.Variable(tf.truncated_normal([num_f, 512], stddev=0.1), name='weights_fc1')
            b_fc1 = tf.Variable(tf.constant(0.1, shape=[512]), name='bias_fc1')
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_fla, w_fc1) + b_fc1)
            h_drop1 = tf.nn.dropout(h_fc1, keep_prob=keep_pro, name='Dropout')

        with tf.name_scope('Output'):
            w_fc2 = tf.Variable(tf.truncated_normal([512, 10], stddev=0.1), name='weights_fc2')
            b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]), name='bias_fc2')
            h_fc2 = tf.matmul(h_drop1, w_fc2) + b_fc2
    
    # ---------------------regularization_L2----------------
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, w_fc1)
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, w_fc2)
        regularizer = tf.contrib.layers.l2_regularizer(scale=15./FLAGS.scale_factor)
        reg_tem = tf.contrib.layers.apply_regularization(regularizer)

        with tf.name_scope('loss'):
            entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=h_fc2))
            entropy_loss = tf.add(entropy_loss, reg_tem, name='l2_loss')
        
        with tf.name_scope('accuracy'):
            prediction = tf.cast(tf.equal(tf.arg_max(h_fc2, 1), tf.argmax(y, 1)), "float")
            accuracy = tf.reduce_mean(prediction)
        
        with tf.name_scope('train'):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            train_op = optimizer.minimize(entropy_loss, global_step=global_step)

        tf.summary.image('input_images', x_imgs, max_outputs=3, collections=['train', 'test'])
        tf.summary.histogram('conv1_weights', w_1, collections=['train'])
        tf.summary.histogram('conv1_bias', b_1, collections=['train'])
        tf.summary.scalar('loss', entropy_loss, collections=['train', 'test'])
        tf.summary.scalar('accuracy', accuracy, collections=['train', 'test'])
        tf.summary.scalar('global_step', global_step, collections=['train'])
        tf.summary.scalar('learning_rate', learning_rate, collections=['train'])

        summ_train = tf.summary.merge_all('train')
        summ_test = tf.summary.merge_all('test')

        max_acc = 100.
        min_loss = 0.1
        sess = tf.Session(graph=graph)
        with sess.as_default():
            sess.run(tf.global_variables_initializer())
            sess.run(iterator.initializer, feed_dict={x_p: imgs_train, y_p: labels_train, batch_size: FLAGS.batch_size})
            # batch_imgs, batch_labels = format_tran(mnist_train, batch_size=batch_size)

            summ_train_dir = os.path.join(FLAGS.event_path, 'summaries','train')
            summ_train_Writer = tf.summary.FileWriter(summ_train_dir)
            summ_train_Writer.add_graph(sess.graph)

            summ_test_dir = os.path.join(FLAGS.event_path, 'summaries', 'test')
            summ_test_Writer = tf.summary.FileWriter(summ_test_dir)
            summ_test_Writer.add_graph(sess.graph)

            saver = tf.train.Saver(max_to_keep=1)
            print(' Training ========== (。・`ω´・) ========')

            for num in range(FLAGS.num_epochs):

                _, acc, loss, rs = sess.run([train_op, accuracy, entropy_loss, summ_train], feed_dict={keep_pro: 0.5, 
                                                                                                        batch_size: FLAGS.batch_size})
                summ_train_Writer.add_summary(rs, global_step=num)
                acc *= 100
                num_e = str(num + 1)
                print_list = [num_e, loss, acc]
                if (num + 1) % 200 == 0:
                    print('Keep on training ========== (。・`ω´・) ========')
                    # print(b_x.shape)
                    # print(b_t)
                    print('Epoch {0[0]}, train_loss is {0[1]:.4f}, accuracy is {0[2]:.2f}%.\n'.format(print_list))
                    
                    if FLAGS.online_test:
                        print(' '*12, 'Online-Testing ========== (。・`ω´・) ========')
                        sess.run(iterator.initializer, feed_dict={x_p: imgs_test, y_p: labels_test, batch_size: num_samples})
                        acc, loss, rs = sess.run([accuracy, entropy_loss, summ_test], feed_dict={keep_pro: 1., 
                                                                                                    batch_size: num_samples})
                        summ_test_Writer.add_summary(rs, global_step=num)
                        acc *= 100
                        print_list = [num_e, loss, acc]
                        # print(' '*10, b_x.shape)
                        # print(' '*10, b_t)
                        print(' '*10, 'Epoch {0[0]}, test_loss is {0[1]:.4f}, accuracy is {0[2]:.2f}%.\n'.format(print_list))
                        
                        sess.run(iterator.initializer, feed_dict={x_p: imgs_train, y_p: labels_train, batch_size: FLAGS.batch_size})

        print('\n', 'Training completed.')
        summ_train_Writer.close()
        summ_test_Writer.close()
        sess.close()
