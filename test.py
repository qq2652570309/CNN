import os
import time
import math
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

a = np.array(range(0,9)).reshape((3,3))
b = np.array(range(-9,0)).reshape((3,3))
c = np.stack((a,b))

a = np.ones(9).reshape((3,3))
b = np.zeros(9).reshape((3,3))
d = np.stack((a,b))

e = np.stack((c,d))

x = tf.get_variable(name="x",initializer=e)

max_pool_output = tf.nn.max_pool(value=x, ksize=[1,1,1,1], strides=[1,2,2,1],padding='VALID')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # print(sess.run(x))
    print(x.shape)

    print('\n#####################\n')
    
    print(sess.run(max_pool_output))
    print(max_pool_output.shape)


