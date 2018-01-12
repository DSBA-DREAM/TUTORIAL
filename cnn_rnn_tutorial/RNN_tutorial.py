import os
import numpy as np
import pandas as pd
import tensorflow as tf

#########################################################################
input_size = 5
rolled_size = 24
# input_size = len(wordset)

x = tf.placeholder(tf.float32, shape=[None, input_size])
rnn_cell = tf.contrib.rnn.BasicRNNCell(num_units=5, activation=tf.tanh)

hidden_state = tf.zeros([1, 5])
for i in range(rolled_size):
    a, hidden_state = rnn_cell(x, hidden_state)
#########################################################################
input_size = 5
rolled_size = 24
x = tf.placeholder(tf.float32, shape=[None, rolled_size, input_size])

rnn_cell = tf.contrib.rnn.BasicRNNCell(num_units=5, activation=tf.tanh)
hidden_state = tf.zeros([1, 5])
output, hidden_state = tf.nn.dynamic_rnn(rnn_cell, x, initial_state=hidden_state)
#########################################################################

def Train_generator(path):
    # train = pd.read_csv("C:/Users/user/Desktop/imple_seminar/RNN_Data/train.csv", encoding='cp949')
    train = pd.read_csv(path, encoding='cp949')
    label = list()

    for score in train['review_point']:
        if score == 1:
            label.append([0,1])
        else:
            label.append([1,0])

    max_sentence_length = 0
    wordset = []
    for review in train['review_text']:
        wordset = wordset + review.split()
        set(wordset)
        if len(review.split()) > max_sentence_length:
            max_sentence_length = len(review.split())

    text = list()
    for sentence in train['review_text']:
        sen = np.zeros((max_sentence_length, len(wordset)))
        i = 0
        for word in sentence.split():
            onehot = [0 for j in range(len(wordset))]
            idx = wordset.index(word)
            onehot[idx] = 1
            sen[i] = np.asarray(onehot)
            i += 1
        text.append(sen)

    text = np.stack(text, axis=0)
    label = np.asarray(label)
    return (text, label)

def feed_dict(train):
    batch_idx = np.random.choice(train_x.shape[0], batch_size, False)
    xs, ys = train_x[batch_idx], train_y[batch_idx]
    return {x: xs, y: ys}

train_x, train_y = Train_generator(os.getcwd() + "/RNN_Data/train.csv")

# rolled_size = 5
# input_size = 10
rolled_size = train_x.shape[1]
input_size = train_x.shape[2]
batch_size = 5

x = tf.placeholder(tf.float32, shape=[None, rolled_size, input_size])
y = tf.placeholder(tf.float32, shape=[None, 2])

rnn_cell = tf.contrib.rnn.BasicRNNCell(num_units=10, activation=tf.tanh)
lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=10, activation=tf.tanh)
gru_cell = tf.contrib.rnn.GRUCell(num_units=10, activation=tf.tanh)

hidden_state = tf.zeros([batch_size, 10])
output, hidden_state = tf.nn.dynamic_rnn(rnn_cell, x, initial_state=hidden_state)

last_output = tf.split(output, num_or_size_splits=rolled_size, axis=1)[rolled_size-1]
shape = last_output.get_shape().as_list()[1] * last_output.get_shape().as_list()[2]
flatted_last_output = tf.reshape(last_output, [-1, shape])
filt = tf.truncated_normal([shape, 2], mean=0, stddev=1)
bias = tf.Variable(tf.constant(0.1, shape=[2]))
fc = tf.matmul(flatted_last_output, filt) + bias
probability = tf.nn.softmax(fc)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=fc))
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(fc, 1), tf.argmax(y, 1)), dtype=tf.float32))
train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(fc, feed_dict(True))
for i in range(12001):
    _ = sess.run(train_step, feed_dict(True))
    if i % 100 == 0:
        _, acc, loss = sess.run([train_step, accuracy, cross_entropy], feed_dict(True))
        print("step : %d, train accuracy : %g, train loss : %g"%(i, acc, loss))

