import tensorflow as tf
import numpy as np
import models.cnn9 as model
IMG_SIZE_PX = 128
SLICE_COUNT = 128

x = tf.placeholder('float')
y = tf.placeholder('float')

train_data = np.load('muchdata-128-128-128-train.npy')
validation_data = np.load('muchdata-128-128-128-validation.npy')
print train_data.shape

def train_neural_network(x):
    prediction = model.CNN(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction) )
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)
    
    hm_epochs = 10
    display = 100
    with tf.Session() as sess:
        #sess.run(tf.initialize_all_variables())
        sess.run(tf.global_variables_initializer())
        successful_runs = 0
        total_runs = 0
        
        for epoch in range(hm_epochs):
            epoch_loss = 0
            run_cnt = 0
            for data in train_data:
                total_runs += 1
                try:
                    X = data[0]
                    Y = data[1]
                    #X = tf.transpose(X, perm=[2,1,0])  
                    #X = tf.reshape(X, shape=[-1, IMG_SIZE_PX, IMG_SIZE_PX, SLICE_COUNT, 1])
                    _, c = sess.run([optimizer, cost], feed_dict={x: X, y: Y})
                    epoch_loss += c
                    successful_runs += 1
                    run_cnt +=1
                    if run_cnt % display == 0:
                        print('Display: ', run_cnt / display,'Average loss:',epoch_loss / run_cnt,'(display iterations:',display,')')
                except Exception as e:
                    # I am passing for the sake of notebook space, but we are getting 1 shaping issue from one 
                    # input tensor. Not sure why, will have to look into it. Guessing it's
                    # one of the depths that doesn't come to 20.
                    print('Warning: Training Exception:',str(e))
                    pass
            
            print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

            for data in validation_data:
                right_cnt=0
                try:
					right_cnt+=accuracy.eval({x:data[0], y:data[1]})
                except Exception as e:
                    print('Warning: Training Exception:',str(e))
                    pass
            print('Accuracy:',float(right_cnt)/validation_data.shape[0])
        
        print('fitment percent:',successful_runs/total_runs)

# Run this locally:
train_neural_network(x)
