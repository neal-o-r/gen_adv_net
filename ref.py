import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import net as net
import adv as adv
import gen as gen

plt.style.use('fivethirtyeight')

fx = lambda x: x
x_range = (-5,5)
n_train = 5000
white_box = None


def black_box(white_box=None):

        one_hot = np.float32([[0, 0]])
        switch = white_box if white_box != None else np.random.randint(2)
        one_hot[0][switch] = 1.

        return (switch, one_hot)


def make_chart(fx, pts, labels, guess):

	flat_pts = np.ravel(pts).reshape(100,2)
	col_inds = np.ravel(guess)

	col = ['#fc4f30', '#6d904f'] # green correct, red wrong
	
	cols = map(lambda x: col[0] if x==0 else col[1], col_inds)
	
	xs = np.linspace(x_range[0], x_range[1], 100)
        plt.plot(xs, fx(xs), label='Function', alpha=0.5)

	plt.scatter(flat_pts[:,0], flat_pts[:,1], s=20, color=cols)

        plt.legend(loc='best')
	plt.show()


def step():

 	switch, label = black_box()

        x = np.random.uniform(x_range[0], x_range[1])
        if switch == 1:
		y = fx(x)
		pt = np.array([[x, y]])

	else:
                pt = sess.run(gen_pt, feed_dict={X_gen:np.array([[x]])})
 
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
Y_rev = tf.abs(Y_adv - 1.0)

y_c = adv.adversary(X_adv)
gen_pt = gen.generator(X_gen)
gen_lab = adv.adversary(gen_pt)

adv_vars = [i for i in tf.trainable_variables() if i.name.startswith('adv')]
gen_vars = [i for i in tf.trainable_variables() if i.name.startswith('gen')]


cost_adv = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_c, Y_adv)) 
cost_gen = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(gen_lab, Y_rev))
 
train_adv = tf.train.AdamOptimizer().minimize(cost_adv, var_list=adv_vars)
train_gen = tf.train.AdamOptimizer().minimize(cost_gen, var_list=gen_vars)


test_adv = tf.equal(tf.argmax(y_c, 1), tf.argmax(Y_adv, 1))

with tf.Session() as sess:
        
        sess.run(tf.initialize_all_variables())
        
        for i in range(n_train):
        
                label, pt =  step()

                sess.run(train_adv, feed_dict={X_adv:pt, Y_adv:label})
          
                if label[0][0] == 1.:
                      sess.run(train_gen, feed_dict={X_adv:pt, Y_adv:label, 
                                X_gen:np.array([[pt[0][0]]])})

        n = 100
	pts, guess, ans = run_test(n)
        print("Correct: %d/%d" %(sum(guess), n))
        
	make_chart(fx, pts, ans, guess)
