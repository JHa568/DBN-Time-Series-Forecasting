import tensorflow as tf
import numpy as np
import csv
import math as m
from RBM import MLP
from MLP import MLP
from dataManip import Data
from tensorMath import TMath
from tensorMath import EligibleMath
'''
About Neural Network:
    Neural network recreation:
        - https://www.intechopen.com/online-first/training-deep-neural-networks-with-reinforcement-learning-for-time-series-forecasting
    Reinforcement learning for time-series prediction:
        - Neural Network architechture:
            - [] <-- Find out in the mean time
    Predictor -> Outcome:
        - Inputs:
            - What influences the drop or peak in price?
        - Outputs:
            - By how much (between -1 to 1)

Project Objective:
    - Predict the future price of bitcoin
        - A day or week or year in advance
'''
directory = 'C:\\Users\\Jamie\\Desktop\\NN_Project\\Tensorflow_projects\\timeSeriesPrediction\\ecoPrediction\\Bitcoin Mining\\DataNeededToBeFiltered\\'
memoryFile = 'memory.csv'
dumpFile = 'rate.csv'
epoch = tf.Variable(50) # How many iterations per unit data
iter = tf.Variable(300) # Iterating entire dataset

# NOTE: Need to extract and save the NN metaData & all variables must be global
bias = tf.Variable(1)
learning_rate = tf.Variable(0.01)
evalFactor = tf.Variable(90) # FIXME: change the evalFactor
reward = tf.Variable(0)
prevTrace = [tf.constant(0), tf.constant(0), tf.constant(0), tf.constant(0), tf.constant(0)]

# FIXME: change the class names into uppercase
tm = TMath()
em = EligibleMath()
d = Data()

RBM_1 = RBM(4, 3, learningRate)
RBM_2 = RBM(3, 2, learningRate)
MLP = MLP(2, 3, 2)
data = d.unpackData(directory, dumpFile)

# FIXME: Rearrange the functions
def createGraphs(x_Axis, y_Axis):
    with tf.device('/cpu:0'):
        # Create the graph here

def nnModel(input, trainning=False):
    bias = tf.Variable(1)
    # NOTE: NN that the weights have been optimised.
    with tf.device('/device:GPU:0'):
        # Here lies the neural networks
        rLayer_1, hLayer_1 = RBM_1.compute(input)
        rLayer_2, hLayer_2 = RBM_2.compute(hLayer_1)
        oLayer = MLP.compute(hLayer_2)
        m_weight, v_weight = MLP.getWeights()
        mean = tm.mean(m_weight, bias, oLayer)
        variance = tm.variance(v_weight, bias, oLayer)
    return mean, variance

def trainningNNModel(input):
    bias = tf.Variable(1)

def dataEnvCir(set):
    try:
        date = [data[0][set], data[0][set+1], data[0][set+2], data[0][set+3]]
        rateData = [data[1][set], data[1][set+1], data[1][set+2], data[1][set+3]]
        return rateData, date
    except:
        print("Error...") # Hwo could this trip the program???

def nextDate(endDate):
    # FIXME: generate the next sequence of dates
    return "Date..."

def grantReward(pastVal, futVal):
    # NOTE: the subtraction of future and past val is the state
    getReward = tf.less_equal(tf.square(tf.subtract(futVal, pastVal)), evalFactor)
    reward = tf.cond(getReward, tf.add(reward, tf.constant(1)), tf.subtract(reward, tf.constant(1)))

def updateWeights():
    # FIXME: identify the parameters needed to update the weights
    RBM_1_weights = RBM_1.getWeights()
    RBM_2_weights = RBM_2.getWeights()
    MLP_weights_1, MLP_weights_2 = MLP.getWeights()

    # FIXME: Have the update algorithm here
    RBM_1.setWeights(RBM_1_weights)
    RBM_2.setWeights(RBM_2_weights)
    MLP.setWeights(MLP_weights_1, MLP_weights_2)

def calEligibilityTrace(EligibleMath, previousTrace, discountFactor=0.1):
    return tf.add(EligibleMath, tf.multiply(discountFactor, previousTrace))

def calCharEligibility(yT, meanT, varT, varArr, nodeT, inputT, meanW, varW, e_Weight_RBM, rbmWeight,
                        hidLayer, visLayer):
    meanE = meanWeight(yT, meanT, varT, nodeT)
    varE = varianceWeight(yT, meanT, varT, varArr, nodeT)
    nodeE = nodeWeight(meanE, meanW, varE, varW, nodeT, inputT)
    # FIXME: iterate over this
    tupleW = (tf.constant(0), tf.constant(0)) # (weightedResult, iter)
    cW = lambda i, jW: tf.less(i, 0) # FIXME: choose an appropriate value to this condition
    bW = lambda i, jW: (tf.add(jW, rbmWeight(e_Weight_RBM, rbmWeight, hidLayer, visLayer)), i+1)
    weightedResultE, iter = tf.while_loop(cW, bW, tupleW) # FIXME: choose an appropriate number

    tupleB = (tf.constant(0), tf.constant(0)) # (biasResult, iter)
    cB = lambda i, kB : tf.less(i, 0) # FIXME:
    bB = lambda i, kB : (tf.add(kB, rbmBias(e_Weight_RBM, rbmWeight, hidLayer)), i+1)
    biasResultE, iter = tf.while_loop(cB, bB, tupleB)

    charE = [meanE, varE, nodeE, weightedResultE, biasResultE]
    trace = [calEligibilityTrace(charE[0], prevTrace[0]), calEligibilityTrace(charE[1], prevTrace[1]),
                calEligibilityTrace(charE[2], prevTrace[2]), calEligibilityTrace(charE[3], prevTrace[3]),
                calEligibilityTrace(charE[4], prevTrace[4])]
    prevTrace = trace
    return trace, characteristic

def deltaWeight(weights, eligTrace, reinBaseLine, reward):
    return tf.multiply(tf.subtract(reward, reinBaseLine), eligTrace)

def newWeight(weight, learningRate, deltaWeight):
    return tf.add(weight, tf.multiply(learning_rate, deltaWeight()))

# NOTE: This method grabs a random sample of data within the set and
# begins measuring the gradient. This is faster than backpropagation
def stochasticGradientAscent(numSet, fRate, iRate, iDate, fDate, predValue):
    # FIXME: Recontruct this function and have a loop
    reinBaseline = tf.constant(0) #FIXME: Choose an acceptable constant
    rates, dates = dataEnvCir(numSet+1)
    grantReward(iRate, fRate)
    # FIXME: Fill in the other parameters
    calCharEligibility(yT, meanT, varT, varArr, nodeT, inputT, meanW,
                        varW, e_Weight_RBM, rbmWeight, hidLayer, visLayer)
    updateWeights()
    # FIXME: have the eligibility math here


if __name__ == "__main__":
    # NOTE: This will be the environment setup for the neural network
    # Get the input data and store it in a tensor variable
    setToExtract = 0
    initialRates, initialDates = dataEnvCir(setToExtract)
    outputLayer = nnModel(initialData)
    predValue = tm.guassianDistriFunc(outputLayer[1], outputLayer[2], initialRates[3], setToExtract+1)
    stochasticGradientAscent(setToExtract, initialRates, initialDates, predValue)
    # NOTE: Trainning phase goes here
    if setToExtract+1 <= len(data[1]):
        # FIXME: loop over this 'x' amount of times
        stochasticGradientAscent(setToExtract, initialRates, initialDates)
    else:
        date = data[0]
        rate = data[1]
        rate.append(predValue)
        # FIXME: generate the date for next day
        date.append(nextDate(initialRates))
