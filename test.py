import os
import time
import math
import numpy as np
import tensorflow as tf
import ngraph_bridge
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load the CIFAR10 dataset
from keras.datasets import cifar10
baseDir = os.path.dirname(os.path.abspath('__file__')) + '/'
classesName = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
(xTrain, yTrain), (xTest, yTest) = cifar10.load_data()
xVal = xTrain[49000:, :].astype(np.float)
yVal = np.squeeze(yTrain[49000:, :])
xTrain = xTrain[:49000, :].astype(np.float)
yTrain = np.squeeze(yTrain[:49000, :])
yTest = np.squeeze(yTest)
xTest = xTest.astype(np.float)


# Show dimension for each variable
print ('Train image shape:    {0}'.format(xTrain.shape))
print ('Train label shape:    {0}'.format(yTrain.shape))
print ('Validate image shape: {0}'.format(xVal.shape))
print ('Validate label shape: {0}'.format(yVal.shape))
print ('Test image shape:     {0}'.format(xTest.shape))
print ('Test label shape:     {0}'.format(yTest.shape))

# Pre processing data
# Normalize the data by subtract the mean image
meanImage = np.mean(xTrain, axis=0)
xTrain -= meanImage
xVal -= meanImage
xTest -= meanImage

# Select device
deviceType = "/cpu:0"

def group_train():
    for intra in [2, 8, 32]:
        tfConfig = tf.ConfigProto(intra_op_parallelism_threads=intra, inter_op_parallelism_threads=2, allow_soft_placement=True, device_count = {'CPU': 10})
        tfConfig.gpu_options.allow_growth = True

        # Simple Model
        tf.reset_default_graph()
        with tf.device(deviceType):
            x = tf.placeholder(tf.float32, [None, 32, 32, 3])
            y = tf.placeholder(tf.int64, [None])
        def simpleModel():
            with tf.device(deviceType):
                wConv = tf.get_variable("wConv", shape=[7, 7, 3, 32])
                bConv = tf.get_variable("bConv", shape=[32])
                w = tf.get_variable("w", shape=[5408, 10]) # Stride = 2, ((32-7)/2)+1 = 13, 13*13*32=5408
                b = tf.get_variable("b", shape=[10])

                # Define Convolutional Neural Network
                a = tf.nn.conv2d(x, wConv, strides=[1, 2, 2, 1], padding='VALID') + bConv # Stride [batch, height, width, channels]
                h = tf.nn.relu(a)
                hFlat = tf.reshape(h, [-1, 5408]) # Flat the output to be size 5408 each row
                yOut = tf.matmul(hFlat, w) + b

                # Define Loss
                totalLoss = tf.losses.hinge_loss(tf.one_hot(y, 10), logits=yOut)
                meanLoss = tf.reduce_mean(totalLoss)

                # Define Optimizer
                optimizer = tf.train.AdamOptimizer(5e-4)
                trainStep = optimizer.minimize(meanLoss)

                # Define correct Prediction and accuracy
                correctPrediction = tf.equal(tf.argmax(yOut, 1), y)
                accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))

                return [meanLoss, accuracy, trainStep]

        def train(Model, xT, yT, xV, yV, xTe, yTe, tfConfig, batchSize=1000, epochs=10, printEvery=10):
            # Train Model
            trainIndex = np.arange(xTrain.shape[0])
            np.random.shuffle(trainIndex)
            with tf.Session(config=tfConfig) as sess:
                sess.run(tf.global_variables_initializer())
                st = time.time()
                for e in range(epochs):
                    # Mini-batch
                    losses = []
                    accs = []
                    # For each batch in training data
                    for i in range(int(math.ceil(xTrain.shape[0] / batchSize))):
                        # Get the batch data for training
                        startIndex = (i * batchSize) % xTrain.shape[0]
                        idX = trainIndex[startIndex:startIndex + batchSize]
                        currentBatchSize = yTrain[idX].shape[0]

                        # Train
                        loss, acc, _ = sess.run(Model, feed_dict={x: xT[idX, :], y: yT[idX]})

                        # Collect all mini-batch loss and accuracy
                        losses.append(loss * currentBatchSize)
                        accs.append(acc * currentBatchSize)

                    totalAcc = np.sum(accs) / float(xTrain.shape[0])
                    totalLoss = np.sum(losses) / xTrain.shape[0]
                    if e % printEvery == 0:
                        print('Iteration {0}: loss = {1:.3f} and training accuracy = {2:.2f}%,'.format(e, totalLoss, totalAcc * 100), end='')
                        loss, acc = sess.run(Model[:-1], feed_dict={x: xV, y: yV})
                        print(' Validate loss = {0:.3f} and validate accuracy = {1:.2f}%'.format(loss, acc * 100))

                loss, acc = sess.run(Model[:-1], feed_dict={x: xTe, y: yTe})
                print('Testing loss = {0:.3f} and testing accuracy = {1:.2f}%'.format(loss, acc * 100))
                print('Time = {0:.4f} seconds.'.format(time.time()-st))

        # Start training simple model
        print("\n################ Simple Model #########################")
        train(simpleModel(), xTrain, yTrain, xVal, yVal, xTest, yTest)

        # Complex Model
        tf.reset_default_graph()
        with tf.device(deviceType):
            x = tf.placeholder(tf.float32, [None, 32, 32, 3])
            y = tf.placeholder(tf.int64, [None])
        def complexModel():
            with tf.device(deviceType):
                #############################################################################
                # TODO: 40 points                                                           #
                # - Construct model follow below architecture                               #
                #       7x7 Convolution with stride = 2                                     #
                #       Relu Activation                                                     #
                #       2x2 Max Pooling                                                     #
                #       Fully connected layer with 1024 hidden neurons                      #
                #       Relu Activation                                                     #
                #       Fully connected layer to map to 10 outputs                          #
                # - Store last layer output in yOut                                         #
                #############################################################################

                Wconv1 = tf.get_variable("wConv", shape=[7, 7, 3, 64])
                bConv = tf.get_variable("bConv", shape=[64])

                # Convolutional Neural Network with stride = 2
                a = tf.nn.conv2d(x, Wconv1, strides=[1, 2, 2, 1], padding='VALID') + bConv # Stride [batch, height, width, channels]
                # Relu Activation
                h = tf.nn.relu(a)

                # 2x2 Max Pooling
                max_pooling = tf.nn.max_pool(h, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID') # 6*6*32
                hFlat = tf.reshape(max_pooling, [-1, 6*6*64]) # Flat the output to be size 6*6*32 each row

                # Fully connected layer with 1024 hidden neurons
                W1 = tf.get_variable("w1", shape=[6*6*64, 1024])
                b1 = tf.get_variable("b1", shape=[1024])
                Hin = tf.matmul(hFlat, W1) + b1

                # Relu Activation
                Hout = tf.nn.relu(Hin)
                HoutFlat = tf.reshape(Hout, [-1, 1024]) # Flat the output to be size 5408 each row

                # Fully connected layer to map to 10 output
                W2 = tf.get_variable("w2", shape=[1024, 10])
                b2 = tf.get_variable("b2", shape=[10])
                yOut = tf.matmul(HoutFlat, W2) + b2


                #########################################################################
                #                       END OF YOUR CODE                                #
                #########################################################################

                # Define Loss
                totalLoss = tf.losses.hinge_loss(tf.one_hot(y, 10), logits=yOut)
                meanLoss = tf.reduce_mean(totalLoss) + 5e-4*tf.nn.l2_loss(Wconv1) + 5e-4*tf.nn.l2_loss(W1) + 5e-4*tf.nn.l2_loss(W2)

                # Define Optimizer
                optimizer = tf.train.AdamOptimizer(5e-4)
                trainStep = optimizer.minimize(meanLoss)

                # Define correct Prediction and accuracy
                correctPrediction = tf.equal(tf.argmax(yOut, 1), y)
                accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))

                return [meanLoss, accuracy, trainStep]

        # Start training complex model
        print("\n################ Complex Model #########################")
        train(complexModel(), xTrain, yTrain, xVal, yVal, xTest, yTest)

group_train()