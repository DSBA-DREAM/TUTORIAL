import os
import numpy as np
import tensorflow as tf
from scipy.misc import imread, imresize

def Train_generator(path):
    image = list()
    label = list()
    for folder in os.listdir(path):
        subpath = path + "/" + folder
        for file in os.listdir(subpath):
            subfile = subpath + "/" + file
            image.append(imresize(imread(subfile), [224, 224, 3]).astype(np.float) / 255.)
            if folder == "man":
                label.append([1, 0])
            else:
                label.append([0, 1])
    image = np.stack(image, axis=0)
    label = np.stack(label, axis=0)
    return (image, label)

def Test_generator(path):
    image = list()
    for folder in os.listdir(path):
        subfile = path + "/" + folder
        image.append(imresize(imread(subfile), [224, 224, 3]).astype(np.float) / 255.)
    image = np.stack(image, axis=0)
    return image

def feed_dict(train):
    batch_idx = np.random.choice(train_x.shape[0], 5, False)
    xs, ys = train_x[batch_idx], train_y[batch_idx]
    return {x: xs, y: ys}

train_path = os.getcwd() + "/CNN_Data/train"
test_path = os.getcwd() + "/CNN_Data/test"
train_x, train_y = Train_generator(train_path)
test_x = Test_generator(test_path)

x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
y = tf.placeholder(tf.float32, shape=[None, 2])


filt1 = tf.truncated_normal([3,3,3,50], mean=0, stddev=1)
# filt1 = tf.get_variable("xavier_1", [11,11,3,50], initializer=tf.contrib.layers.xavier_initializer())
bias1 = tf.Variable(tf.constant(0.1, shape=[50]))
conv1 = tf.nn.conv2d(x, filt1, [1,2,2,1], padding="SAME", name="conv1") + bias1
relu1 = tf.nn.relu(conv1, name="relu1")
pool1 = tf.nn.max_pool(relu1, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME", name="pool1")

filt2 = tf.truncated_normal([3,3,50,100], mean=0, stddev=1)
# filt2 = tf.get_variable("xavier_1", [11,11,50,100], initializer=tf.contrib.layers.xavier_initializer())
bias2 = tf.Variable(tf.constant(0.1, shape=[100]))
conv2 = tf.nn.conv2d(pool1, filt2, [1,2,2,1], padding="SAME", name="conv2") + bias2
relu2 = tf.nn.relu(conv2, name="relu2")
pool2 = tf.nn.max_pool(relu2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME", name="pool2")

shape = pool2.get_shape().as_list()[1] * pool2.get_shape().as_list()[2] * pool2.get_shape().as_list()[3]
flatted_pool2 = tf.reshape(pool2, [-1, shape])
filt3 = tf.truncated_normal([shape,2], mean=0, stddev=1)
# filt2 = tf.get_variable("xavier_1", [shape,2], initializer=tf.contrib.layers.xavier_initializer())
bias3 = tf.Variable(tf.constant(0.1, shape=[2]))
fc1 = tf.matmul(flatted_pool2, filt3) + bias3

probability = tf.nn.softmax(fc1)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=fc1))
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(fc1, 1), tf.argmax(y, 1)), dtype=tf.float32))
train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for i in range(2001):
    _ = sess.run(train_step, feed_dict(True))
    if i % 100 == 0:
        _, acc, loss = sess.run([train_step, accuracy, cross_entropy], feed_dict(True))
        print("step : %d, train accuracy : %g, train loss : %g"%(i, acc, loss))

save_model_path = "./"
tf.train.Saver().save(sess, save_model_path)




