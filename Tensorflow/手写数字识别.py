import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

batch_size = 100
num_batch = round(mnist.train.num_examples/batch_size)

x_data = tf.placeholder(tf.float32, [None, 784])
y_data = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

#prob = tf.constant(1.0)
w1 = tf.Variable(tf.truncated_normal([784, 2000], stddev=0.1))
b1 = tf.Variable(tf.zeros([2000]) + 0.1)########
wx_plus_bias1 = tf.matmul(x_data, w1) + b1
out1 = tf.nn.relu(wx_plus_bias1)
L1_drop = tf.nn.dropout(out1, keep_prob)

w2 = tf.Variable(tf.truncated_normal([2000, 2000], stddev=0.1))
b2 = tf.Variable(tf.zeros([2000]) + 0.1)########
wx_plus_bias2 = tf.matmul(L1_drop, w2) + b2
out2 = tf.nn.relu(wx_plus_bias2)
L2_drop = tf.nn.dropout(out2, keep_prob)

w3 = tf.Variable(tf.truncated_normal([2000, 1000], stddev=0.1))
b3 = tf.Variable(tf.zeros([1000]) + 0.1)########
wx_plus_bias3 = tf.matmul(L2_drop, w3) + b3
out3 = tf.nn.relu(wx_plus_bias3)
L3_drop = tf.nn.dropout(out3, keep_prob)

w4 = tf.Variable(tf.truncated_normal([1000, 10], stddev=0.1))
b4 = tf.Variable(tf.zeros([10]) + 0.1)########
wx_plus_bias4 = tf.matmul(L3_drop, w4) + b4
out4= tf.nn.softmax(wx_plus_bias4)






#loss = tf.reduce_mean(tf.square(y_data - out4))# + 0.005*tf.reduce_sum(tf.square(weights1))   #0.9138
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_data, logits=out4))
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

sess.run(tf.global_variables_initializer())

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_data, 1), tf.argmax(out4, 1)), tf.float32))

for i in range(21):
    for batch in range(num_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(train_step, feed_dict={x_data:batch_xs, y_data:batch_ys, keep_prob:1.})
        
    test_acc = sess.run(accuracy, feed_dict={keep_prob:1., x_data:mnist.test.images, y_data:mnist.test.labels})
    train_acc = sess.run(accuracy, feed_dict={keep_prob:1., x_data:mnist.train.images, y_data:mnist.train.labels})
    print('Test_acc %f, Train_acc %f'%(test_acc, train_acc))   
