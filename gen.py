import tensorflow as tf
import numpy as np
import net as net


def generator(x):
        
        l0 = tf.nn.softplus(net.feed_forward(x, 20, 'gen', 'l0'))
        l1 = net.feed_forward(l0, 1, 'gen', 'l1')     

        return tf.transpose(tf.concat(0, [x, l1]))
