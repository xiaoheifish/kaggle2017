import tensorflow as tf
import numpy as np
import time;
import models.cnn9 as model
IMG_SIZE_PX = 128
SLICE_COUNT = 128
batch_size=4 # batch_size <=4 , for memory constrain
display = 50
hm_epochs = 10

x = tf.placeholder('float')
y = tf.placeholder('float')

train_data = np.load('muchdata-128-128-128-train.npy')
validation_data = np.load('muchdata-128-128-128-validation.npy')
print 'train_data.shape:',train_data.shape
print 'validation_data.shape:',validation_data.shape

def get_normalized_data(iters,batch_size,data):
    X=[data[i,0].astype(float)/128-1 for i in range(iters*batch_size,(iters+1)*batch_size)]
    Y=[data[i,1] for i in range(iters*batch_size,(iters+1)*batch_size)]
    return X,Y


def train_neural_network(x):
    prediction = model.CNN(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction) )
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(cost)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
   
    with tf.Session(config=config) as sess:
        #sess.run(tf.initialize_all_variables())
        sess.run(tf.global_variables_initializer())
        successful_runs = 0
        total_runs = 0
        
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for iters in range(0,train_data.shape[0]/batch_size):
                total_runs += 1
                try:
                    X,Y=get_normalized_data(iters,batch_size,train_data)
                    _, c = sess.run([optimizer, cost], feed_dict={x: X, y: Y})
                    epoch_loss += c
                    successful_runs += 1
                    if (iters+1) % display == 0:
                        print '[',time.asctime( time.localtime(time.time()) ),']','Epoch',epoch+1,'Display', iters / display + 1,'Average loss:',epoch_loss / display,'(display iterations:',display,')'
                        epoch_loss=0
                except Exception as e:
                    # I am passing for the sake of notebook space, but we are getting 1 shaping issue from one 
                    # input tensor. Not sure why, will have to look into it. Guessing it's
                    # one of the depths that doesn't come to 20.
                    print 'Warning: Training Exception:',str(e)
                    pass
                #break 

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_sum(tf.cast(correct, 'float'))
			
            right_cnt=0
            for iters in range(0,validation_data.shape[0]/batch_size):
                try:
                    X,Y=get_normalized_data(iters,batch_size,train_data)
                    right_cnt+=accuracy.eval({x:X, y:Y})
                except Exception as e:
                    print('Warning: Test Exception:',str(e))
                    pass
            print('Accuracy:',float(right_cnt)/validation_data.shape[0])
        
        print('fitment percent:',successful_runs/total_runs)

# Run this locally:
train_neural_network(x)
