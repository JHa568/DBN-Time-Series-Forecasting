import tensorflow as tf
import tensorflow.metrics as mtc
import numpy as np
'''
sub = tf.subtract(tf.constant(10), tf.constant(5))
sess = tf.InteractiveSession()
init_op = tf.global_variables_initializer()

def multiply(x, y):
    return tf.multiply(x,y)
with tf.Session() as sess:
     sess.run(init_op)
     print("Subtraction:", str(sess.run(sub)))
     print("Subtraction_NAN:", str(sess.run(multiply(5,4))))
'''
'''
DON'T DELETE THIS CODE:
# Trainning goes here
def nnTrain(inputData):
    with tf.device('/device:GPU:0'):
        iter = tf.Variable(0) # FIXME: What purpose does this serve
        numOfData = tf.Variable(len(inputData))
        try:
            # NOTE: NeuralNetwork algorithm here
            # NOTE: save the csv file WHEN FINISHING the algorithm
        except KeyboardInterrupt:
            # save memory to csv file
            d.packMemory(np.asarray(weights))

'''
'''
tuple = (tf.constant(0), tf.constant(0))
c = lambda i, x: tf.less(x, 10)
b = lambda i, x: (i + tf.constant(1), x + tf.constant(2))
q, r = tf.while_loop(c, b, tuple)
with tf.compat.v1.Session() as sess:
    print(sess.run(r))
    print(sess.run(q))
'''
