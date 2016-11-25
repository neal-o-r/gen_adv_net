import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import net as net
import adv as adv
import gen as gen

plt.style.use('fivethirtyeight')

fx = lambda x: x
x_range = (-5,5)
n_train = 3000
white_box = None


def black_box(white_box=None):

        one_hot = np.float32([[0, 0]])
        switch = white_box if white_box != None else np.random.randint(2)
        one_hot[0][switch] = 1.

        return (switch, one_hot)


def make_chart(fx, pts, labels, guess):

	flat_pts = np.ravel(pts).reshape(100,2)
	sym_inds = np.ravel(np.argmax(labels, axis=2))
	col_inds = np.ravel(guess)

	syms = ['s','o']  # s - random, o - function
	cols = ['r', 'g'] # green correct, red wrong

	for i, p in enumerate(flat_pts):
		plt.scatter(p[0], p[1], 
                   marker=syms[sym_inds[i]], color=cols[col_inds[i]])

	plt.show()

def step():

 	switch, label = black_box()

        x = np.random.uniform(x_range[0], x_range[1])
        if switch == 1:
		y = fx(x)
		pt = np.array([[x, y]])

	else:
                y = sess.run(gen_y, feed_dict={X_gen:np.array([[x]])})
        	pt = np.array([[x, y[0][0]]])
 
        return label, pt


def run_test(n):

        pts, ans, guess = [], [], []
	for i in range(n):

        	label, pt = step()

		pts.append(pt)
		ans.append(label)

                guess.append(sess.run(test_adv, feed_dict={X_adv:pt, Y_adv:label}))
        
        return pts, guess, ans


X_gen = tf.placeholder(tf.float32, [1,1], name="X_gen") 

X_adv = tf.placeholder(tf.float32, [1, 2], name="X_adv")
Y_adv = tf.placeholder(tf.float32, [1, 2], name="Y_adv")

gen_y = gen.generator(X_gen)
y_c = adv.adversary(X_adv)


cost_adv = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_c, Y_adv)) 
cost_gen = tf.nn.l2_loss(fx(X_gen) - gen_y)


train_adv = tf.train.AdamOptimizer().minimize(cost_adv)
train_gen = tf.train.AdamOptimizer().minimize(cost_gen)

test_adv = tf.equal(tf.argmax(y_c, 1), tf.argmax(Y_adv, 1))

with tf.Session() as sess:
        
        sess.run(tf.initialize_all_variables())
        
        for i in range(n_train):
        
                label, pt =  step()

                sess.run(train_adv, feed_dict={X_adv:pt, Y_adv:label})
                sess.run(train_gen, feed_dict={X_adv:pt, Y_adv:label, 
                                X_gen:np.array([[pt[0][0]]])})

        n = 100
	pts, guess, ans = run_test(n)
        print("Correct: %d/%d" %(sum(guess), n))
        
	make_chart(fx, pts, ans, guess)
