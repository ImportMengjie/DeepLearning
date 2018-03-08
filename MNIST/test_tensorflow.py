import tensorflow as tf
import numpy as np
import load_data


def guiyi(a: list):
    return list(map(lambda x: 0 if x == 0 else x/255, a))
# def guiyi(a):
#     return a


# print(guiyi(load_data.test_imgs))
test_x_data = np.reshape(guiyi(load_data.test_imgs), (len(
    load_data.test_imgs)//(load_data.width*load_data.height), load_data.width*load_data.height))
# test_x_data = np.reshape(load_data.test_imgs, (len(
#     load_data.test_imgs)//(load_data.width*load_data.height), load_data.width*load_data.height))

# test_x_data = tf.constant(test_x_data, dtype='float32')

train_data = np.reshape(guiyi(load_data.train_imgs), (len(load_data.train_imgs) //
                                                      (load_data.width * load_data.height), load_data.width * load_data.height))
# train_data = np.reshape(load_data.train_imgs, (len(load_data.train_imgs) //
#                                                (load_data.width * load_data.height), load_data.width * load_data.height))

# train_data = tf.constant(train_data, dtype='float32')
x_data = tf.placeholder(
    'float32', (None, load_data.width*load_data.height))

train_lab = list(
    map(lambda x: [1. if j == x else 0. for j in range(10)], load_data.train_lab))
# train_lab = tf.constant(np.reshape(
#     train_lab, (len(load_data.train_lab), 10)), dtype='float32')

test_lab = list(
    map(lambda x: [1. if j == x else 0. for j in range(10)], load_data.test_lab))
# test_lab = tf.constant(np.reshape(
#     test_lab, (len(load_data.test_lab), 10)), dtype='float32')


y_hat = tf.placeholder('float32', shape=(None, 10))

# w1 = tf.Variable(
#     tf.ones((load_data.width * load_data.height, 300), name='layer1w', dtype='float32'))
#b1 = tf.Variable(tf.ones((1, 5), name='layer1b', dtype='float32'))
b1 = tf.get_variable(name='b1', shape=(1, 100), dtype='float32',
                     initializer=tf.zeros_initializer())
w1 = tf.get_variable(name='w1', shape=(load_data.width*load_data.height, 100),
                     dtype='float32', initializer=tf.truncated_normal_initializer())

# w2 = tf.Variable(
#     tf.ones((5, 10), name='layer2w'))
w2 = tf.get_variable(name='w2', shape=(100, 20),
                     dtype='float32', initializer=tf.truncated_normal_initializer())
#b2 = tf.Variable(tf.zeros((1, 10), name='layer2b'))
b2 = tf.get_variable(name='b2', shape=(1, 20), dtype='float32',
                     initializer=tf.zeros_initializer())

w3 = tf.get_variable(name='w3', shape=(20, 10),
                     dtype='float32', initializer=tf.truncated_normal_initializer())
b3 = tf.get_variable(name='b3', shape=(1, 10), dtype='float32',
                     initializer=tf.zeros_initializer())

a = tf.matmul(x_data, w1) + b1
# a1 = tf.nn.relu(a)
a1 = tf.nn.sigmoid(a)
a2 = tf.nn.sigmoid(tf.matmul(a1, w2) + b2)
#a2 = tf.matmul(a1, w2) + b2
a3 = tf.nn.sigmoid(tf.matmul(a2, w3)+b3)
y = tf.nn.softmax(a3)


cross_entropy = -tf.reduce_sum(y_hat * tf.log(y))
#cross_entropy = tf.reduce_sum(tf.square(y-y_hat))
# cross_entropy = tf.reduce_sum(
#     tf.nn.softmax_cross_entropy_with_logits(labels=y_hat, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_hat, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float32"))


with tf.Session() as sess:
    # init = tf.initialize_all_variables()
    # sess.run(init)
    # writer = tf.summary.FileWriter('./graphs', sess.graph)
    sess.run(tf.global_variables_initializer())
    for _ in range(10000):
        #sess.run(train_step, feed_dict={x_data: train_data, y_hat: train_lab})
        sess.run([train_step], feed_dict={
            x_data: train_data[_ % (len(train_data)//10)*10:_ % (len(train_data)//10)*10+10], y_hat: train_lab[_ % (len(train_data)//10)*10:_ % (len(train_data)//10)*10+10]})

        print(sess.run([accuracy], feed_dict={
              x_data: test_x_data, y_hat: test_lab}))

    # for i in range(len(load_data.train_imgs)):
    #     sess.run(train_step, feed_dict={
    #              x_data: train_data[i*784:(i+1)*784], y_hat: np.reshape(train_lab[i], (1, 10))})
    #     print(sess.run([accuracy], feed_dict={
    #         x_data: test_x_data, y_hat: test_lab}))
