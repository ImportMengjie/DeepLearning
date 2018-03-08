import numpy as np
import tensorflow as tf
import csv


def get_weight(shape):
    return tf.Variable(tf.truncated_normal(shape))


def conv2d(input, filter):
    return tf.nn.conv2d(input=input, filter=filter, strides=[1, 1, 1, 1], padding='SAME')


x_data = tf.placeholder('float32', (None))
y_data = tf.placeholder('int32', (None))

x = tf.reshape(x_data/255., (-1, 48, 48, 1))
y = tf.one_hot(y_data, 6)

conv1_filter = get_weight((3, 3, 1, 16))
conv1_bias = get_weight((1, 16))
conv1 = conv2d(x, conv1_filter)
conv1_out = tf.nn.relu(conv1+conv1_bias)

conv2_filter = get_weight((3, 3, 16, 8))
conv2_bias = get_weight((1, 8))
conv2 = conv2d(conv1_out, conv2_filter)
conv2_out = tf.nn.relu(conv2+conv2_bias)

conv3_filter = get_weight((3, 3, 8, 4))
conv3_bias = get_weight((1, 4))
conv3 = conv2d(conv2_out, conv3_filter)
conv3_out = tf.nn.relu(conv3 + conv3_bias)


flat = tf.reshape(conv3_out, (-1, 28*28*4))

fc_w1 = get_weight((28*28*4, 100))
fc_b1 = get_weight((1, 100))

fc_out1 = tf.nn.relu(tf.matmul(flat, fc_w1)+fc_b1)

fc_w2 = get_weight((100, 50))
fc_b2 = get_weight((1, 50))

fc_out2 = tf.nn.relu(tf.matmul(fc_out1, fc_w2)+fc_b2)

fc_w3 = get_weight((50, 6))
fc_b3 = get_weight((1, 6))

y_hat = tf.nn.softmax((tf.matmul(fc_out2, fc_w3)+fc_b3))

cross_entropy = - \
    tf.reduce_mean(y_hat*tf.log(tf.clip_by_value(y, 1e-11, 1.0)))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
# train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_hat, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float32"))


def cast_format(a):


with open('ml-2017fall-hw3/train.csv') as train, open('ml-2017fall-hw3/test.csv') as test:
    csv_train, csv_test = list(csv.reader(train))[
        1:], list(csv.reader(test))[1:]
    train_data = np.matrix(cast_format(csv_train))
    test_data = np.matrix(cast_format(csv_test))
    batch_size = 64

    with tf.Session() as sess:
        # init = tf.initialize_all_variables()
        # sess.run(init)
        # writer = tf.summary.FileWriter('./graphs', sess.graph)
        sess.run(tf.global_variables_initializer())
        for _ in range(10000):
            s = (_ % (train_data.shape[0]//batch_size))*batch_size
            l = (_ % (train_data.shape[0]//batch_size))*batch_size+batch_size
            sess.run(train_step, feed_dict={
                x_data: train_data[s: l, 1], y_data: train_data[s: l, 0]})
            if _ % 5 == 0:
                print(sess.run(accuracy, feed_dict={
                    x_data: test_data[:, 1], y_data: test_data[:, 0]}))
