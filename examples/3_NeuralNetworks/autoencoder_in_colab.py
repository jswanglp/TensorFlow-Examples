#@title Autoencoder { display-mode: "both" }
# ex-2_3 autoencoder
# conding: utf-8
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import cv2

tf.logging.set_verbosity(tf.logging.ERROR)

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
path_data = '/content/MNIST_data' #@param {type: "string"}
mnist = input_data.read_data_sets(path_data, one_hot=True)
# mnist = input_data.read_data_sets("E:\Anaconda2\Programs\MNIST_data", one_hot=True)

# Training Parameters
learning_rate = 0.003 #@param {type: "number"}
num_steps = 50000 #@param {type: "integer"}
batch_size = 48 #@param {type: "integer"}

display_step = 1000
examples_to_show = 10

# Network Parameters
num_hidden_1 = 128 #@param {type: "integer"}
num_hidden_2 = 100 #@param {type: "integer"}
num_sq = int(np.sqrt(num_hidden_2))
num_input = 784 # MNIST data input (img shape: 28*28)

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, num_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([num_input])),
}

# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Encoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2


# Building the decoder
def decoder(x):
    # Decoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start Training
# Start a new TF session
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Training
    start_time = time.time()
    for i in range(1, num_steps+1):
        # Prepare Data
        # Get the next batch of MNIST data (only images are needed, not labels)
        batch_x, _ = mnist.train.next_batch(batch_size)

        # Run optimization op (backprop) and cost op (to get loss value)
        _, l = sess.run([optimizer, loss], feed_dict={X: batch_x})
        # Display logs per step
        if i % display_step == 0:
            end_time = time.time()
            running_time = end_time - start_time
            start_time = end_time
            print('Step %i: Minibatch Loss: %f, ' % (i, l) + 'running time: {0:.2f}s.'.format(running_time))

    # Testing
    # Encode and decode images from test set and visualize their reconstruction.
    n = 4
    canvas_orig = np.empty((28 * n, 28 * n))
    canvas_recon = np.empty((28 * n, 28 * n))
    img_features = np.empty((n * n, num_sq, num_sq))
    for i in range(n):
        # MNIST test set
        batch_x, _ = mnist.test.next_batch(n)
        # Encode and decode the digit image
        f, g = sess.run([encoder_op, decoder_op], feed_dict={X: batch_x})
        # Display image features
        for j in range(n):
            img_features[i * n + j] = f[j].reshape([num_sq, num_sq])
        # Display original images
        for j in range(n):
            # Draw the original digits
            canvas_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                batch_x[j].reshape([28, 28])
        # Display reconstructed images
        for j in range(n):
            # Draw the reconstructed digits
            canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                g[j].reshape([28, 28])

    fig = plt.figure(1, figsize=(10, 5))
    image_names = ["Original Images", "Reconstructed Images"]
    images_o_r = [canvas_orig, canvas_recon]
    AX = [fig.add_subplot(i) for i in range(121, 123)]
    for na, img, ax in zip(image_names, images_o_r, AX):
        ax.imshow(img, origin="upper", cmap="gray")
        ax.set_title(na)
        ax.set_xticks([]), ax.set_yticks([])
        ax.grid()
    plt.show()

    fig_f = plt.figure(2, figsize=(5, 5))
    G = gridspec.GridSpec(n, n)
    G.wspace, G.hspace = 0.05, 0.05
    for i in range(n):
        for j in range(n):
            plt.subplot(G[i, j])
            plt.imshow(img_features[i * n + j], cmap='gray')
            plt.xticks([]), plt.yticks([])

    plt.show()
    
    path_test = '/content/GoogleDrive/MATLAB/number2.jpg' #@param {type: "string"}
    img_test = cv2.imread(path_test, cv2.IMREAD_GRAYSCALE)
    img_re = cv2.resize(img_test, dsize=(28, 28), interpolation=cv2.INTER_LINEAR)
    img_t = (255 - img_re.reshape(1,-1)) / 255
    img_enco, img_deco = sess.run([encoder_op, decoder_op], feed_dict={X: img_t})
    fig = plt.figure(3, figsize=(10, 5))
    image_names = ["Original Images", "Reconstructed Images"]
    images_o_r = [img_re, img_deco.reshape(28, 28)]
    AX = [fig.add_subplot(i) for i in range(121, 123)]
    for na, img, ax in zip(image_names, images_o_r, AX):
        ax.imshow(img, origin="upper", cmap="gray")
        ax.set_title(na)
        ax.set_xticks([]), ax.set_yticks([])
        ax.grid()
    plt.show()
    
sess.close()
