import tensorflow as tf
import numpy as np
import net as net


def generator(x):
        
        l0 = tf.nn.softplus(net.feed_forward(x, 20, 'gen', 'l0'))
        l1 = net.feed_forward(l0, 1, 'gen', 'l1')
        
        return l1
'''


X = tf.placeholder(tf.float32, [1, 1], name="X")
Y = tf.placeholder(tf.float32, [1, 1], name="Y")

y_c = approximator(X)

cost = tf.nn.l2_loss(y_c - Y)
train_op = tf.train.AdamOptimizer().minimize(cost)

guesses = []
with tf.Session() as sess:
        
        sess.run(tf.initialize_all_variables())

        for i in range(n_epoch):
        
                inds = np.random.random_integers(0, 
                        len(train[0])-1, size=n_batch) 

                x_b = train[0].ravel()[inds]
                y_b = train[1].ravel()[inds]               
        
                for x, y in zip(x_b, y_b):
                        
                        sess.run(train_op, feed_dict={X: np.array([[x]]), 
                                                      Y: np.array([[y]])})

                if i%100 == 0 :
                        mse = 0.
                        for x, y in zip(test[0].ravel(), test[1].ravel()):

                                mse += sess.run(cost, feed_dict={X:np.array([[x]]),
                                                                 Y:np.array([[y]])})
                        

                        print("Epoch %i, MSE %g" %(i, mse))
        
        for x, y in zip(test[0].ravel(), test[1].ravel()):

                guesses.append(sess.run(y_c,  feed_dict={X:np.array([[x]])})[0][0])
                        
make_plot(train, guesses)
'''
