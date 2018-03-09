import numpy as np
import tensorflow as tf
import csv


def get_weight(shape):
    return tf.Variable(tf.truncated_normal(shape, mean=1.0, stddev=0.5))


def conv2d(input, filter):
    return tf.nn.conv2d(input=input, filter=filter, strides=[1, 1, 1, 1], padding='SAME')


x_data = tf.placeholder('float32', (None))
y_data = tf.placeholder('int32', (None))

x = tf.reshape(x_data/255., (-1, 48, 48, 1))
y = tf.one_hot(y_data, 6)

conv1_filter = get_weight((5, 5, 1, 64))
conv1_bias = get_weight((1, 64))
conv1 = conv2d(x, conv1_filter)
conv1_out = tf.nn.relu(conv1+conv1_bias)

conv2_filter = get_weight((5, 5, 64, 32))
conv2_bias = get_weight((1, 32))
conv2 = conv2d(conv1_out, conv2_filter)
conv2_out = tf.nn.relu(conv2+conv2_bias)

conv3_filter = get_weight((5, 5, 32, 16))
conv3_bias = get_weight((1, 16))
conv3 = conv2d(conv2_out, conv3_filter)
conv3_out = tf.nn.relu(conv3 + conv3_bias)
flat_wide = int(conv3_out.shape[1]*conv3_out.shape[2]*conv3_out.shape[3])
print(conv3_out.shape)
flat = tf.reshape(conv3_out, (-1, flat_wide))

fc_w1 = get_weight((48*48*16, 100))
fc_b1 = get_weight((1, 100))

fc_out1 = tf.nn.relu(tf.matmul(flat, fc_w1)+fc_b1)

fc_w2 = get_weight((100, 50))
fc_b2 = get_weight((1, 50))

fc_out2 = tf.nn.relu(tf.matmul(fc_out1, fc_w2)+fc_b2)

fc_w3 = get_weight((50, 6))
fc_b3 = get_weight((1, 6))

y_hat = tf.nn.softmax((tf.matmul(fc_out2, fc_w3)+fc_b3))

cross_entropy = - \
    tf.reduce_mean(y*tf.log(tf.clip_by_value(y_hat, 1e-11, 1.0)))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
#train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
# train_step = tf.train.MomentumOptimizer(0.001).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_hat, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float32"))


def cast_format(a):
    def cast(t):
        return int(t[0]), list(map(int, t[1].split(' ')))
    lable, data = [], []
    for i in a:
        t1, t2 = cast(i)
        lable.append(t1)
        data.append(t2)
    return lable, data


with open('ml-2017fall-hw3/train.csv') as train, open('ml-2017fall-hw3/test.csv') as test:
    csv_train, csv_test = list(csv.reader(train))[
        1:], list(csv.reader(test))[1:]
    train_y, train_x = cast_format(csv_train)
    test_y, test_x = cast_format(csv_test)
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_y = np.array(test_y)
    test_x = np.array(test_x)
    batch_size = 32
    len_of_test = len(test_y)

    with tf.Session() as sess:
        # init = tf.initialize_all_variables()
        # sess.run(init)
        # writer = tf.summary.FileWriter('./graphs', sess.graph)
        sess.run(tf.global_variables_initializer())
        for _ in range(10000):
            s = (_ % (train_x.shape[0]//batch_size))*batch_size
            l = (_ % (train_x.shape[0]//batch_size))*batch_size+batch_size
            sess.run(train_step, feed_dict={
                x_data: train_x[s: l], y_data: train_y[s: l]})
            if _ % 5 == 0:
                total = 0.
                for i in range(10):
                    total += sess.run(accuracy, feed_dict={
                        x_data: test_x[i*512:i*512+512], y_data: test_y[i*512:i*512+512]})
                print(total/10)
