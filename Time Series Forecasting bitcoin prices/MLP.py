import tensorflow as tf
from tensorMath import tMath

class MLP(object):

    def __init__(self, inputLayer, hiddLayer, outputLayer, bias_1=1, bias_2=1, weights_1=None, weights_2=None):
        self.bias_1 = bias_1
        self.bias_2 = bias_2
        if (weights_1 != None) and (weights_2 != None):
            self.weight_1 = tf.Variable(tf.random_normal([inputLayer, hiddLayer]))
            self.weight_2 = tf.Variable(tf.random_normal([hiddLayer, outputLayer]))
            print("Created random sets of weights...")
        else:
            self.weight_1 = tf.convert_to_tensor(weights_1)
            self.weight_2 = tf.convert_to_tensor(weights_2)
            print("Preloaded weights...")

    def compute(self, inputData):
        # Simple feed forward mechanism
        hiddenLayer = math.sigFunc(inputData, self.weight_1, self.bias_1)
        outputLayer = math.sigFunc(hiddenLayer, self.weight_2, self.bias_2)
        # NOTE: outputLayer has tensor dtype
        return outputLayer

    def getWeights(self):
        # NOTE: weights have tensor dtype
        return self.weight_1, self.weight_2

    def setWeights(self, w1, w2):
        # NOTE: parameters may have a different dtype
        self.weight_1 = tf.convert_to_tensor(w1)
        self.weight_2 = tf.convert_to_tensor(w2)
