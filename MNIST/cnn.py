import tensorflow as tf
import numpy as np
import load_data

batch_size = 64


# def guiyi(a: list):
#     return list(map(lambda x: 0 if x == 0 else x/255, a))


def guiyi(a):
    return a


test_data = np.reshape(guiyi(load_data.test_imgs),
                       (-1, load_data.width, load_data.height, 1))
# test_data = guiyi(load_data.test_imgs)
# train_data = guiyi(load_data.train_lab)

train_data = np.reshape(guiyi(load_data.train_imgs),
                        (-1, load_data.width, load_data.height, 1))

train_lab = list(
    map(lambda x: [1. if j == x else 0. for j in range(10)], load_data.train_lab))

test_lab = list(
    map(lambda x: [1. if j == x else 0. for j in range(10)], load_data.test_lab))

x_data = tf.placeholder(
    'float32', (None, load_data.width, load_data.height, 1))
x_data_t = x_data/255.
y_hat = tf.placeholder('float32', shape=(None, 10))
keep_prob = tf.placeholder(tf.float32)
conv_filter_1 = tf.Variable(tf.truncated_normal([5, 5, 1, 16]))

print(conv_filter_1)

conv_1 = tf.nn.conv2d(x_data_t, filter=conv_filter_1,
                      strides=[1, 1, 1, 1], padding='SAME')
c1_b = tf.Variable(tf.constant(0.1, shape=[1, 16]))
conv_1_out = tf.nn.relu(conv_1+c1_b)
max_pool1 = tf.nn.max_pool(conv_1_out, ksize=[1, 2, 2, 1], strides=[
    1, 2, 2, 1], padding='SAME')
print(max_pool1)

conv_filter_2 = tf.Variable(tf.truncated_normal(
    [3, 3, 16, 8]))

conv_2 = tf.nn.conv2d(max_pool1, conv_filter_2, strides=[
    1, 1, 1, 1], padding='SAME')
c2_b = tf.Variable(tf.constant(0.1, shape=[1, 8]))
conv_2_out = tf.nn.relu(conv_2+c2_b)
max_pool2 = tf.nn.max_pool(conv_2_out, ksize=[1, 2, 2, 1], strides=[
    1, 2, 2, 1], padding='SAME')
# max_pool2 = conv_2_out
# print(max_pool2)
flat_width = int(max_pool2.shape[1]*max_pool2.shape[2]*max_pool2.shape[3])
flat_t = tf.reshape(max_pool2, [-1, flat_width])
# flat = flat_t/tf.reduce_max(flat_t, 1, keep_dims=True)
# flat = flat_t-tf.reduce_mean(flat_t, 1, keep_dims=True)
# print(flat)
flat = flat_t

fc_w1 = tf.Variable(tf.random_normal([flat_width, 100]))
fc_b1 = tf.Variable(tf.random_normal([1, 100]))
fc_out1_t = tf.nn.relu(tf.matmul(flat, fc_w1) + fc_b1)
fc_out1 = tf.nn.dropout(fc_out1_t, keep_prob=keep_prob)

out_w1 = tf.Variable(tf.random_normal([100, 10]))
out_b1 = tf.Variable(tf.random_normal([10]))
y = tf.nn.softmax(tf.matmul(fc_out1, out_w1)+out_b1)

cross_entropy = - \
    tf.reduce_mean(y_hat*tf.log(tf.clip_by_value(y, 1e-11, 1.0)))
# cross_entropy = tf.reduce_mean(
#     -tf.reduce_sum(y_hat*tf.log(y),
#                    reduction_indices=[1]))
# cross_entropy = -tf.reduce_mean(y_hat * tf.log(y))


train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
#train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_hat, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float32"))

with tf.Session() as sess:
    # init = tf.initialize_all_variables()
    # sess.run(init)
    # writer = tf.summary.FileWriter('./graphs', sess.graph)
    sess.run(tf.global_variables_initializer())
    for _ in range(10000):
        sess.run([train_step], feed_dict={keep_prob: 0.5,
                                          x_data: train_data[_ % (len(train_data)//batch_size)*batch_size:_ % (len(train_data)//batch_size)*batch_size+batch_size], y_hat: train_lab[_ % (len(train_data)//batch_size)*batch_size:_ % (len(train_data)//batch_size)*batch_size+batch_size]})
        if _ % 5 == 0:
            print(sess.run([accuracy], feed_dict={keep_prob: 1,
                                                  x_data: test_data, y_hat: test_lab}))
