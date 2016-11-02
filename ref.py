import tensorflow as tf
import numpy as np
import net as net
import adv as adv
import gen as gen

fx = lambda x: 2*x**2 + x + 10
x_range = (-5,5)
n_train = 4000


def black_box(fx, x_range, white_box=None):

        one_hot = np.float32([[0, 0]])
        switch = white_box if white_box != None else np.random.randint(2)
        one_hot[0][switch] = 1.

        if switch == 1:
                x = np.random.uniform(x_range[0], x_range[1])
                y = fx(x)

        else:
                x = np.random.uniform(x_range[0], x_range[1])
                y = np.random.uniform(fx(x_range[0]), fx(x_range[1])) 

        pts = np.array([[np.float32(x), np.float32(y)]])

        return (pts, one_hot)



X = tf.placeholder(tf.float32, [1, 2], name="X")
Y = tf.placeholder(tf.float32, [1, 2], name="Y")

y_c = adv.adversary(X)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_c, Y))

train_op = tf.train.AdamOptimizer().minimize(cost)
test = tf.equal(tf.argmax(y_c, 1), tf.argmax(Y, 1))


with tf.Session() as sess:
        
        sess.run(tf.initialize_all_variables())
        
        pt, label = black_box(gen.generator, x_range)
        print(sess.run(cost, feed_dict={X:pt, Y:label}))

        for i in range(n_train):
                
                pt, label = black_box(fx, x_range)
                sess.run(train_op, feed_dict={X:pt, Y:label})

                if i%250 == 0:
                        err = 0
                        for j in range(50):
                                pt, lab = black_box(fx, x_range)
                                err += sess.run(test, feed_dict={X:pt, Y:lab})

                        print("Epoch: %d, Correct: %d/50"%(i, err))

