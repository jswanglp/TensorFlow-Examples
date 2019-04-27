# conding: utf-8

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io as scio
# import time

tf.logging.set_verbosity(tf.logging.ERROR)

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

def glorot_init(shape):
    return tf.truncated_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))

def bias_init(shape):
    return tf.random_normal(shape=shape, stddev=.1)

mnist = input_data.read_data_sets("E:\Anaconda2\Programs\MNIST_data", one_hot=True)
# mnist = input_data.read_data_sets('sample_data/MNIST_data', one_hot=True)

# Training Parameters
tf.app.flags.DEFINE_float('learning_rate', 5e-4, 'learning rate, default is 5e-4.')
tf.app.flags.DEFINE_float('decay_steps', 2000, 'decay steps, default is 2000.')
tf.app.flags.DEFINE_integer('num_epochs', 50000, 'number of epochs, default is 50000.')
tf.app.flags.DEFINE_integer('batch_size', 128, 'batch size, default is 128.')
FLAGS = tf.app.flags.FLAGS

# file path
file_path = os.path.dirname(os.path.abspath(__file__))
checkpoint_path = os.path.join(file_path, 'Checkpoints')
event_path = os.path.join(file_path, 'Tensorboard')

# Network Parameters
img_dim = mnist.train.images[0].shape[-1]
gen_hidden_dim = 256
disc_hidden_dim = 256
feature_dim = 100
disc_output_dim = 1

# tf Graph input (only pictures)
graph = tf.Graph()
with graph.as_default():

    global_step = tf.Variable(0, name='global_step', trainable=False)
    init_rate = FLAGS.learning_rate
    decay_rate = 0.9
    decay_steps = FLAGS.decay_steps
    learning_rate = tf.train.exponential_decay(init_rate, global_step=global_step,
                                                decay_steps=decay_steps, 
                                                decay_rate=decay_rate, 
                                                staircase=True, 
                                                name='exponential_decay')

    with tf.name_scope('Input'):
        gen_input = tf.placeholder(tf.float32, [None, feature_dim], name='random_noises')
        disc_input = tf.placeholder(tf.float32, [None, img_dim], name='real_images')

    with tf.name_scope('Weights_and_biases'):
        weights = {
            'gen_h': tf.Variable(glorot_init([feature_dim, gen_hidden_dim]), name='gen_w1'),
            'gen_o': tf.Variable(glorot_init([gen_hidden_dim, img_dim]), name='gen_w2'),
            'disc_h': tf.Variable(glorot_init([img_dim, disc_hidden_dim]), name='disc_w1'),
            'disc_o': tf.Variable(glorot_init([disc_hidden_dim, disc_output_dim]), name='disc_w2'),
            }
        biases = {
            'gen_b1': tf.Variable(bias_init([gen_hidden_dim]), name='gen_b1'),
            'gen_b2': tf.Variable(bias_init([img_dim]), name='gen_b2'),
            'disc_b1': tf.Variable(bias_init([disc_hidden_dim]), name='disc_b1'),
            'disc_b2': tf.Variable(bias_init([disc_output_dim]), name='disc_b2'),
            }
    with tf.name_scope('Generator_and_Discriminator'):
        # Building the generator
        def generator(x):
            layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['gen_h']),
                                            biases['gen_b1']))
            layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['gen_o']),
                                            biases['gen_b2']))
            return layer_2

        # Building the discriminator
        def discriminator(x):
            layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['disc_h']),
                                           biases['disc_b1']))
            layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['disc_o']),
                                           biases['disc_b2']))
            return layer_2

    with tf.name_scope('Main_structure'):
        # Construct model
        gen_sample = generator(gen_input)
        gen_imgs = tf.reshape(gen_sample, [-1, 28, 28, 1], name='gen_imgs')
        disc_fake = discriminator(gen_sample)
        disc_real = discriminator(disc_input)

    with tf.name_scope('Loss'):
        gen_loss = -tf.reduce_mean(tf.log(disc_fake + 1e-10))
        disc_loss = -tf.reduce_mean(tf.log(disc_real + 1e-10) + tf.log(1. - disc_fake))

    with tf.name_scope('Train'):
        # Build Optimizers
        optimizer_gen = tf.train.AdamOptimizer(learning_rate=learning_rate)
        optimizer_disc = tf.train.AdamOptimizer(learning_rate=learning_rate)
        # By default in TensorFlow, all variables are updated by each optimizer, so we
        # need to precise for each one of them the specific variables to update.
        # Generator Network Variables
        gen_vars = [weights['gen_h'], weights['gen_o'],
                    biases['gen_b1'], biases['gen_b2']]
        # Discriminator Network Variables
        disc_vars = [weights['disc_h'], weights['disc_o'],
                    biases['disc_b1'], biases['disc_b2']]

        trainop_gen = optimizer_gen.minimize(gen_loss, global_step=global_step, var_list=gen_vars)
        # trainop_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars)
        trainop_disc = optimizer_disc.minimize(disc_loss, var_list=disc_vars)

    # summaries
    tf.summary.histogram('gen_w1', weights['gen_h'], collections=['gen'])
    tf.summary.histogram('gen_b1', biases['gen_b1'], collections=['gen'])
    tf.summary.histogram('disc_w2', weights['disc_o'], collections=['disc'])
    tf.summary.histogram('disc_b2', biases['disc_b2'], collections=['disc'])
    tf.summary.image('gen_imgs', gen_imgs, collections=['gen'])
    tf.summary.scalar('gen_loss', gen_loss, collections=['gen'])
    tf.summary.scalar('disc_loss', disc_loss, collections=['disc'])
    tf.summary.scalar('global_step', global_step, collections=['gen'])
    tf.summary.scalar('learning_rate', learning_rate, collections=['gen'])

    gen_sum = tf.summary.merge_all('gen')
    disc_sum = tf.summary.merge_all('disc')

    saver = tf.train.Saver(max_to_keep=1)

    sess = tf.Session(graph=graph)
    with sess.as_default():
        # Initialize the variables (i.e. assign their default value)
        sess.run(tf.global_variables_initializer())

        gen_path = os.path.join(event_path, 'gen')
        gen_writer = tf.summary.FileWriter(gen_path)
        gen_writer.add_graph(sess.graph)

        disc_path = os.path.join(event_path, 'disc')
        disc_writer = tf.summary.FileWriter(disc_path)
        disc_writer.add_graph(sess.graph)

        # Start Training
        gen_loss_list = []
        disc_loss_list = []
        for epoch in range(FLAGS.num_epochs):
            batch, _ = mnist.train.next_batch(FLAGS.batch_size)
            random_noise = np.random.uniform(-1., 1., size=[FLAGS.batch_size, feature_dim])
            _, _, l_g, l_d, s_g, s_d = sess.run([trainop_gen, trainop_disc, gen_loss, disc_loss, gen_sum, disc_sum], 
                                                feed_dict={gen_input: random_noise, disc_input: batch})

            # s_g, s_d = sess.run([gen_sum, disc_sum], feed_dict={gen_input: random_noise, 
            #                                                     disc_input: batch})
            gen_writer.add_summary(s_g, global_step=epoch)
            disc_writer.add_summary(s_d, global_step=epoch)
            if (epoch + 1) % 100 == 0:
                gen_loss_list.append(l_g)
                disc_loss_list.append(l_d)
            # # Train a classifier first
            # _, l_d = sess.run([trainop_disc, disc_loss], feed_dict={gen_input: random_noise, 
            #                                                         disc_input: batch})
            # if (epoch+1) % 2 == 0:
            #     _, l_g = sess.run([trainop_gen, gen_loss], feed_dict={gen_input: random_noise,
            #                                                             disc_input: batch})
            print_list = [epoch+1, l_g, l_d]
            if (epoch+1) % 2000 == 0 or (epoch+1) == 1:
                print('Epoch {0[0]}: Generator Loss: {0[1]:.4f}, Discriminator Loss: {0[2]:.4f}.'.format(print_list))

        data_name = os.path.join(event_path, 'loss_data.mat')
        loss_data = {'gen_loss': np.array(gen_loss_list),
                    'disc_loss': np.array(disc_loss_list)}
        scio.savemat(data_name, loss_data)
        # loss_data = scio.loadmat(data_name)
        x_data = np.linspace(1, FLAGS.num_epochs + 1, FLAGS.num_epochs / 100)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 9))
        par1 = ax.twinx()
        p1, = ax.plot(x_data, loss_data['gen_loss'], 'r', label='gen_loss')
        p2, = par1.plot(x_data, loss_data['disc_loss'], 'b', label='disc_loss')
        ax.set_xlabel('epochs', fontsize=14)
        ax.set_ylabel('gen_loss', fontsize=14)
        par1.set_ylabel('disc_loss', fontsize=14)
        ax.yaxis.label.set_color(p1.get_color())
        par1.yaxis.label.set_color(p2.get_color())
        ax.tick_params(axis='y', colors=p1.get_color(), fontsize=14)
        par1.tick_params(axis='y', colors=p2.get_color(), fontsize=14)
        ax.tick_params(axis='x', fontsize=14)
        lines = [p1, p2]
        ax.legend(lines, [l.get_label() for l in lines], fontsize=14, loc='upper center')
        img_name = os.path.join(event_path, 'gen_loss_and_disc_loss.jpg')
        plt.savefig(img_name)
        plt.show()

        # Generate images from noise, using the generator network.
        n = 6
        canvas = np.empty((28 * n, 28 * n))
        for i in range(n):
            # Noise input.
            z = np.random.uniform(-1., 1., size=[n, feature_dim])
            # Generate image from noise.
            g = sess.run(gen_sample, feed_dict={gen_input: z})
            # Reverse colours for better display
            g = -1 * (g - 1)
            for j in range(n):
                # Draw the generated digits
                canvas[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = g[j].reshape([28, 28])

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(n, n))
        ax.imshow(canvas, cmap='gray')
        ax.set_xticks([]), ax.set_yticks([])
        img_name1 = os.path.join(event_path, 'generated_images_by_GAN1.jpg')
        plt.savefig(img_name1)
        plt.show()

        fig, AX = plt.subplots(nrows=6, ncols=6, figsize=(10, 10))
        fig.subplots_adjust(wspace=0.1, hspace=0.1)
        # Noise input.
        z = np.random.uniform(-1., 1., size=[36, feature_dim])
        # Generate image from noise.
        g = sess.run(gen_sample, feed_dict={gen_input: z})
        g = (1 - g)
        for i in range(6):
            for j in range(6):
                ax = AX[i, j]
                ax.imshow(g[i*6 + j].reshape([28, 28]), 'gray')
                ax.set_xticks([]), ax.set_yticks([])
        img_name2 = os.path.join(event_path, 'generated_images_by_GAN2.jpg')
        plt.savefig(img_name2)
        plt.show()


    gen_writer.close()
    disc_writer.close()
    sess.close()


