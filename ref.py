import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import net as net
import adv as adv
import gen as gen

fx = lambda x: x**2
x_range = (-5,5)
n_train = 10000
white_box = None

def black_box(x_range, white_box=None):

        one_hot = np.float32([[0, 0]])
        switch = white_box if white_box != None else np.random.randint(2)
        one_hot[0][switch] = 1.

        return (switch, one_hot)

def make_chart(fx, pts, labels, guess):

	flat_pts = np.ravel(pts).reshape(100,2)
	sym_inds = np.ravel(np.argmax(labels, axis=2))
	col_inds = np.ravel(guess)

	syms = ['s','o']
	cols = ['r', 'k']

	for i, p in enumerate(flat_pts):
		plt.scatter(p[0], p[1], marker=syms[sym_inds[i]], color=cols[col_inds[i]])

	plt.show()



X_gen = tf.placeholder(tf.float32, [1,1], name="X_gen") 

X_adv = tf.placeholder(tf.float32, [1, 2], name="X_adv")
Y_adv = tf.placeholder(tf.float32, [1, 2], name="Y_adv")

gen_x = gen.generator(X_gen)
y_c = adv.adversary(X_adv)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_c, Y_adv)) #\
#       + tf.nn.l2_loss(1.0 - y_c[0][1])

train_op = tf.train.AdamOptimizer().minimize(cost)
test = tf.equal(tf.argmax(y_c, 1), tf.argmax(Y_adv, 1))

with tf.Session() as sess:
        
        sess.run(tf.initialize_all_variables())
        
        for i in range(n_train):
        	switch, label = black_box(x_range)

        	if switch == 1:
                	x_rand = np.array([[np.random.uniform(x_range[0], x_range[1])]])
			y_gen = fx(x_rand)#sess.run(gen_x, feed_dict={X_gen:x_rand})
			pt = np.array([[x_rand[0][0], y_gen]])

		else:
                	x = np.random.uniform(x_range[0], x_range[1])
                	y = np.random.uniform(0, 25)
        		pt = np.array([[np.float32(x), np.float32(y)]])
 
                sess.run(train_op, feed_dict={X_adv:pt, Y_adv:label})

	pts, guess, ans = [], [], []
	for i in range(100):

        	switch, label = black_box(x_range)

        	if switch == 1:
                	x_rand = np.array([[np.random.uniform(x_range[0], x_range[1])]])
			y_gen = fx(x_rand)#sess.run(gen_x, feed_dict={X_gen:x_rand})
			pt = np.array([[x_rand[0][0], y_gen]])

		else:
                	x = np.random.uniform(x_range[0], x_range[1])
                	y = np.random.uniform(0, 25)
        		pt = np.array([[np.float32(x), np.float32(y)]])
		pts.append(pt)
		ans.append(label)

                guess.append(sess.run(test, feed_dict={X_adv:pt, Y_adv:label}))

	make_chart(fx, pts, ans, guess)
