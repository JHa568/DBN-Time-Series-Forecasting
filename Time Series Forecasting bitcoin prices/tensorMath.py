import tensorflow as tf
import tensorflow.metrics as mtc
import math as m

# NOTE: Math that is needed is stored here
class TMath(object):

    def sigFunc(self, nodes, weights, bias=0):
        return tf.sigmoid(tf.add(bias, tf.multiply(weights, nodes)))

    def mean(self, meanWeights, bias, z):
        mean = tf.Variable(0)
        # FIXME: Make this into a tensorflow styled equation
        for i in range(0, len(z)):
            mean = tf.add(mean, tf.multiply(meanWeights, bias))
        return mean

    def variance(self, varWeights, bias, z):
        sum = tf.Variable(0)
        # FIXME: Make this into a tensorflow styled equation
        for i in range(0, len(z)):
            sum = tf.add(sum, tf.multiply(varWeights, bias))
        return tf.sigmoid(sum)

    # This function can be deleted
    def guassianDistriFunc(self, mean, variance, pastInput, instanceOfTime):
        # FIXME: make all math operators to tensorflow style
        # NOTE: pi is the population proportion
        two_x_pi = tf.multiply(2, tf.divide(instanceOfTIme, 3834))
        return tf.multiply(
                    tf.divide(tf.constant(1.0), tf.multiply(variance, tf.sqrt(two_x_pi))),
                    tf.exp(tf.negative(tf.divide(tf.square(tf.subtract(pastInput,mean)), tf.multiply(2, variance)))))

    def adpativeLearningRate(self, beta, truth, predicted):
        # NOTE: the after and before y-values are from the previous iteration
        return tf.multiply(beta, mtc.mean_squared_error(truth, predicted))

class EligibleMath(object):
    # This is the math that
    def meanWeight(self, yT, meanT, varT, node):
        # NOTE:
        #    - yT: The "current" y value in the series
        #    - meanT: The "current" mean value in the series
        #    - varT: The "current" variance value in the series
        #    - node: the hiddenLayer in the MLP at that time
        return tf.multiply(tf.divide(tf.subtract(yT, meanT), tf.squared(varT)), node)

    def varianceWeight(self, yT, meanT, varT, varNorm, node):
        # FIXME: what dType does varNorm hold?
        x = tf.divide(tf.subtract(tf.squared(tf.subtract(yT, meanT)), tf.squared(varNorm)), tf.squared(varT))
        y = tf.multiply(x, tf.subtract(tf.Constant(1.0), varT))
        return tf.multiply(y, node)

    def nodeWeight(self, eligMean, meanWeight, eligVar, varWeight, hidNode, inputNode):
        # FIXME: make varaible names more meaningful
        eligSum = tf.add(tf.multiply(eligMean, meanWeight), tf.multiply(eligVar, varWeight))
        return tf.multiply(
                tf.multiply(eligSum, tf.subtract(tf.constant(1.0), hidNode)),
                inputNode)

    def sumEligWeight(self, eligiWeight, weights, iter):
        # NOTE: eligiWeight and weights are tensors that are in the next layer of DBN 
        tuple = (tf.constant(0), tf.constant(0))
        c = lambda i, v: tf.less(i, iter)
        b = lambda i, v: (tf.add(v, tf.multiply(eligWeight, weights)), i+1)
        summation, iter = tf.while_loop(c, b, tuple)
        return summation # should be a matrix of values

    def rbmWeight(self, eligiWeight, weights, hidLayer, visLayer):
        # NOTE: sumEligWeight goes here
        # FIXME: a section needs to be iterated
        # NOTE: does this equation need to be rearranged?
        return tf.multiply(
                tf.multiply(sumEligWeight(eligiWeight, weights), tf.subtract(tf.constant(1), hidLayer)),
                visLayer)

    def rbmBias(self, eligiWeight, weights, hidLayer):
        # NOTE: sumEligWeight goes here
        # FIXME:a section needs to be iterated
        return tf.multiply(sumEligWeight(eligiWeight, weights), tf.subtract(tf.constant(1), hidLayer))
