import tensorflow as tf
import numpy as np
import net as net

def adversary(pt):

        l0 = tf.nn.tanh(net.feed_forward(pt, 20, 'adversary', 'l0'))
        l1 = tf.nn.tanh(net.feed_forward(l0, 20, 'adversary', 'l1'))
        l2 = tf.nn.tanh(net.feed_forward(l1, 10, 'adversary', 'l2'))
        l3 = net.feed_forward(l2, 2, 'adversary', 'l3')

        return l3
