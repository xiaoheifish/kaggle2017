import tensorflow as tf

IMG_SIZE_PX = 128
SLICE_COUNT = 128

n_classes = 2
keep_rate = 0.8

def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1,1,1,1,1], padding='SAME')

def maxpool3d(x,k,s):
    #                        size of window         movement of window as you slide about
    return tf.nn.max_pool3d(x, ksize=[1,k,k,k,1], strides=[1,s,s,s,1], padding='SAME')
def CNN(x):
    weights = {'W_conv1':tf.Variable(tf.random_normal([3,3,3,1,32])),
               'W_conv2':tf.Variable(tf.random_normal([3,3,3,32,32])),
               'W_conv3':tf.Variable(tf.random_normal([3,3,3,32,32])),
               'W_conv4':tf.Variable(tf.random_normal([3,3,3,32,32])),
               'W_conv5':tf.Variable(tf.random_normal([3,3,3,32,32])),
               'out':tf.Variable(tf.random_normal([32 , n_classes]))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
              'b_conv2':tf.Variable(tf.random_normal([32])),
              'b_conv3':tf.Variable(tf.random_normal([32])),
              'b_conv4':tf.Variable(tf.random_normal([32])),
              'b_conv5':tf.Variable(tf.random_normal([32])),
              'out':tf.Variable(tf.random_normal([n_classes]))}

    #                            image X      image Y        image Z
    x = tf.reshape(x, shape=[-1, IMG_SIZE_PX, IMG_SIZE_PX, SLICE_COUNT, 1])
    conv1 = tf.nn.relu(conv3d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool3d(conv1,k=3,s=2)
    conv2 = tf.nn.relu(conv3d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool3d(conv2,k=3,s=2)
    conv3 = tf.nn.relu(conv3d(conv2, weights['W_conv3']) + biases['b_conv3'])
    conv3 = maxpool3d(conv3,k=3,s=2)
    conv4 = tf.nn.relu(conv3d(conv3, weights['W_conv4']) + biases['b_conv4'])
    conv4 = maxpool3d(conv4,k=3,s=2)
    conv5 = tf.nn.relu(conv3d(conv4, weights['W_conv4']) + biases['b_conv5'])
    conv5 = maxpool3d(conv5,k=8,s=8)
	# [5,13,13,64]
    fc = tf.reshape(conv5,[-1, 32])
    fc = tf.nn.dropout(fc, keep_rate)
    output = tf.matmul(fc, weights['out'])+biases['out']
    return output
