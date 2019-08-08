import tensorflow as tf
from tensorMath import tMath

class RBM (object):
    # NOTE: RBM has two layers; one visible and one hidden layer.
    def __init__(self, visibleLayer, hiddenLayer, learningRate  , bias_for=1, bias_back=1, weights=None):
        if weights != None:
            self.RBMWeight = tf.Variable(tf.random_normal([visibleLayer, hiddLayer]))
        else:
            self.RBMWeight = tf.convert_to_tensor(weights)
        self.bias_for = bias_for
        self.bias_back = bias_back

    def compute(self, inputData):
        # FIXME: Determine the dtype of the input data.
        data = tf.convert_to_tensor(inputData) # Initial visible layer.
        hiddenLayer = math.sigFunc(data, self.RBMWeight, self.bias_for)
        condition = tf.greater_equal(hiddenLayer, tf.constant(0.5))
        hiddenLayer = tf.where(condition, tf.zeros_like(hiddenLayer), tf.ones_like(hiddenLayer))
        # NOTE: This is the reconstruction of the 'new' inputs.
        # FIXME: find whether reconVisLayer goes through an activation function
        reconVisLayer = math.sigFunc(hiddenLayer, tf.transpose(self.RBMWeight), self.bias_back)
        #tf.add(tf.multiply(tf.transpose(self.RBMWeight), hiddenLayer), self.bias_back)
        condition = tf.greater_equal(reconVisLayer, tf.constant(0.5))
        reconVisLayer = tf.where(condition, tf.zeros_like(reconVisLayer), tf.ones_like(reconVisLayer))

        # NOTE: reconVisLayer is used to backpropagate the weights
        # NOTE: the hiddenLayer is the 'output'
        return reconVisLayer, hiddenLayer

    def getWeights(self):
        return self.RBMWeight

    def setWeights(self, w):
        # NOTE: weights may have tensor dtype
        self.RBMWeight = tf.convert_to_tensor(w)
