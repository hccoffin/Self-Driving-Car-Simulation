# Howard Coffin, hcoffin, Section B
# Neural Network

import numpy as np
import matplotlib.pyplot as plt
import random, math

class NeuralNetwork(object):

    def __init__(self, inputNodes, hiddenNodes, outputNodes):
        # number of nodes in each layer defines neural net's architecture
        self.weights = self.initWeights(inputNodes, hiddenNodes, outputNodes)
        self.numInputs = inputNodes
        self.numOutputs = outputNodes
        self.numHidden = hiddenNodes
        self.hasHiddenLayers = not(isinstance(hiddenNodes, int) 
                                  and hiddenNodes == 0)

    def makeRandom2DList(self, rows, cols):
        l = []
        for r in range(rows):
            row = []
            for c in range(cols):
                row.append(random.random()*2 - 1)
            l.append(row)
        return l

    def printWeights(self, weights):
        # used for testing
        print()
        for layer in weights:
            for row in layer:
                for weight in row:
                    w = "%05.2f" % weight
                    print(w, end = ", ")
                print()
            print()

    def getWeights(self):
        # returns weights
        return self.weights

    def setWeights(self, weights):
        # used for testing
        self.weights = weights


    def initWeightsNoHiddenLayers(self, inputNodes, outputNodes):
        # creates weights linking the input layer to the output layer
        inToOutWeights = self.makeRandom2DList(inputNodes+1, outputNodes)
        return [np.matrix(inToOutWeights)]

    def initWeightsOneHiddenLayer(self, inputNodes, hiddenNodes, outputNodes):
        # creates weights linking the input layer to the hidden layer and the 
        # hidden layer to the output layer
        weightsList = []
        inToHiddenWeights = self.makeRandom2DList(inputNodes+1, hiddenNodes)
        weightsList.append(np.matrix(inToHiddenWeights))
        hiddenToOutWeights = self.makeRandom2DList(hiddenNodes+1, outputNodes)
        weightsList.append(np.matrix(hiddenToOutWeights))
        return weightsList

    def initWeightsManyHiddenLayers(self, inputNodes, hiddenNodes, outputNodes):
        # creates weights linking the input layer to the first hidden layer,
        # each hidden layer to the next one and the last hidden layer to the
        # output layer
        weightsList = []
        inTo1stHiddenWeights = self.makeRandom2DList(inputNodes+1, 
                                                     hiddenNodes[0])
        weightsList.append(np.matrix(inTo1stHiddenWeights))

        for layer in range(len(hiddenNodes)-1):
            hiddenToHiddenWeights = self.makeRandom2DList(hiddenNodes[layer]+1, 
                                               hiddenNodes[layer+1])
            weightsList.append(np.matrix(hiddenToHiddenWeights))

        lastHiddenToOutWeights = self.makeRandom2DList(
            hiddenNodes[len(hiddenNodes) - 1] + 1, outputNodes)
        weightsList.append(np.matrix(lastHiddenToOutWeights))
        return weightsList

    def initWeights(self, inputNodes, hiddenNodes, outputNodes):
        # creates an matrix of weights to fit the specified architecture
        if isinstance(hiddenNodes, int): # one or zero hidden layers
            if(hiddenNodes == 0): # no hidden layer
                weights = self.initWeightsNoHiddenLayers(inputNodes, 
                    outputNodes)
            else: # one hidden layer
                weights = self.initWeightsOneHiddenLayer(inputNodes, 
                    hiddenNodes, outputNodes)
        else: # multiple hidden layers
            weights = self.initWeightsManyHiddenLayers(inputNodes, hiddenNodes, 
                outputNodes)
        return weights


    def sigmoid(self, a):
        # returns the sigmoid of an matrix: a (elementwise)
        return 1/(1+np.exp(-1*a))

    def sigmoidDerivative(self, a):
        # returns the derivative of the the sigmoid function of an matrix: a 
        # (elementwise)
        return np.multiply(self.sigmoid(a), (1-self.sigmoid(a)))

    def forwardPropagation(self, inputs):
        # calculates the outputs of the neural network for given inputs
        output = inputs
        for weightLayer in self.weights:
                # adds bias node
                output = np.matrix(np.append([1], output))
                # calculates output of next layer
                output = output * weightLayer
                output = self.sigmoid(output)
        return output

    def booleanOutput(self, inputs):
        outputs = self.forwardPropagation(inputs)
        return outputs >= .5

    def booleanOutputToNumericOutput(self, outputs):
        numericInputs = np.zeros(outputs.shape)
        trueNum = 1
        falseNum = 0
        for i in range(len(outputs)):
            if outputs[i] == True:
                numericInputs[i] = trueNum
            else:
                numericInputs[i] = falseNum
        return numericInputs 

    def getError(self, inputs, expectedOutputs):
        error = 0
        for example in range(len(expectedOutputs)):
            actualOutputs = self.booleanOutput(inputs[example])
            if actualOutputs != expectedOutputs[example]:
                error += 1
        return error

    def backPropagation(self, inputs, expectedOutputs, learningRate = .5):
        # attempts to correct error in the output of the neural network using
        # known inputs and outputs

        # finds the outputs and inputs at each level of the neural network
        output = inputs
        xList = []
        yList = []
        for weightLayer in self.weights:
            output = np.matrix(np.append([1], output))
            yList += [output] 
            output = output * weightLayer
            xList += [output]
            output = self.sigmoid(output)
        actualOutputs = output

        # finds out how much error is attributed to each weight and changes
        # weights accordingly
        changeInWeights = []
        error = (actualOutputs - expectedOutputs).transpose()
        # loops through neural network's layers backwards
        for weightLayer in range(len(self.weights) - 1, -1, -1):
            # E = .5*(total squared error)
            # W = weights, x = node input, y = node output
            # dE/dW = (dE/dy)*(dy/dx)*(dx/dW)   
            # dx/dW = y from previous layer; dy/dx = sigmoidDerivative(x)
            # dE/dy = actual-expected
            # (dE/dy)*(dy/dx) = error
            y = np.matrix(yList[weightLayer])
            x = np.matrix(xList[weightLayer])
            error = np.multiply(error.transpose(), self.sigmoidDerivative(x))
            deltaWeight = -1 * y.transpose() * error
            changeInWeights = [deltaWeight * learningRate] + changeInWeights
            # updates error for next layer
            error = self.weights[weightLayer] * error.transpose()
            # gets rid of error related to bias node's value
            error = error[1:]

        self.weights += np.array(changeInWeights)
        
    def learn(self, inputs, expectedOutputs, learningRate = .5):
        # uses back propagation to teach the neural network
        for example in range(len(inputs)):
            inputEx = inputs[example]
            outputEx = expectedOutputs[example]
            self.backPropagation(inputEx, outputEx, learningRate)

    @ staticmethod
    def drawNode(canvas, x, y, diameter, on):
        color = "white" if on else "black"
        radius = diameter/2
        points = (x-radius, y-radius, x+radius, y+radius)
        canvas.create_oval(points, fill = color)

    def drawSynapses(self, canvas, x, y, height, numLayers, nodeWidth, 
                     distBetweenLayers, nodes):
        # draws "synapses": the lines between nodes
        outputNodeX = x + nodeWidth/2
        inputNodeX = x + nodeWidth/2 + distBetweenLayers
        centerY = y + height/2
        for layer in range(numLayers-1):
            nodeLayer = nodes[layer]
            nextNodeLayer = nodes[layer+1]
            nodesInOutputLayer = nodeLayer.size
            nodesInInputLayer = nextNodeLayer.size

            outputStartY = centerY - 2*nodeWidth*((nodesInOutputLayer - 1)/2)
            outputEndY = centerY + 2*nodeWidth*((nodesInOutputLayer - 1)/2)
            outputNodeY = outputStartY

            inputStartY = centerY - 2*nodeWidth*((nodesInInputLayer - 1)/2)
            inputEndY = centerY + 2*nodeWidth*((nodesInInputLayer - 1)/2)

            for outputNode in np.nditer(nodeLayer):
                inputNodeY = inputStartY
                for inputNode in np.nditer(nextNodeLayer):
                    canvas.create_line(outputNodeX, outputNodeY, 
                                       inputNodeX, inputNodeY, fill = "black",
                                       stipple = "gray75")
                    inputNodeY += nodeWidth * 2
                outputNodeY += nodeWidth * 2
            outputNodeX += distBetweenLayers
            inputNodeX += distBetweenLayers

    def drawNodes(self, canvas, x, y, height, numLayers, nodeWidth, 
                  distBetweenLayers, nodes):
        # draws nodes
        nodeX = x + nodeWidth/2
        centerY = y + height/2
        for layer in range(numLayers):
            nodeLayer = nodes[layer]
            nodesInLayer = nodeLayer.size
            startY = centerY - 2 * nodeWidth * ((nodesInLayer - 1)/2)
            endY = centerY + 2 * nodeWidth * ((nodesInLayer - 1)/2)
            nodeY = startY
            for node in np.nditer(nodeLayer):
                NeuralNetwork.drawNode(canvas, nodeX, nodeY, nodeWidth, node)
                nodeY += nodeWidth * 2
            nodeX += distBetweenLayers

    def draw(self, canvas, inputs, x, y, width, height):
        # finds the outputs at each level of the neural network
        output = inputs
        sigmoidTrue = .5
        nodes = []
        for weightLayer in self.weights:
            output = np.matrix(np.append([1], output))
            nodes += [output >= sigmoidTrue] 
            output = output * weightLayer
            output = self.sigmoid(output)
        nodes += [output >= sigmoidTrue]

        # finds dimensions to use for drawing
        numLayers = len(nodes)
        maxNodes = 0
        for nodeLayer in nodes:
            if nodeLayer.size > maxNodes: maxNodes = nodeLayer.size
        nodeWidth = (height/maxNodes)/2
        distBetweenLayers = (width/(numLayers-1)) - nodeWidth/2

        self.drawSynapses(canvas, x, y, height, numLayers, nodeWidth, distBetweenLayers, nodes)
        self.drawNodes(canvas, x, y, height, numLayers, nodeWidth, distBetweenLayers, nodes)
        


            

def almostEqualMatrixes(a1, a2, epsilon=10**-4):
    maxError = np.amax(np.absolute(a1-a2))
    return maxError < epsilon

def testSigmoid():
    print("Testing sigmoid", end = " ")
    nn = NeuralNetwork(0,0,0)
    a = np.matrix([[-4, -3, 0, 3, 4], 
                   [-2, -1, 0, 1, 2]])
    sigAExpected = np.matrix([[.01799, .04743, .5, .95257, .98201],
                              [.1192 , .26894, .5, .73106, .8808]])
    assert(almostEqualMatrixes(nn.sigmoid(a), sigAExpected))
    print("Passed!")

def testforwardPropagation():
    print("Testing forwardPropagation", end = " ")
    inputArr = np.matrix([1,2,3,4])

    # testing network with no hidden layers
    nn = NeuralNetwork(4,0,3)
    weights = nn.getWeights()
    expectedOutput = nn.sigmoid(np.matrix(np.append([1], inputArr))*weights[0])
    assert(np.array_equal(nn.forwardPropagation(inputArr), expectedOutput))

    # testing neural netwrok with one hidden layer
    nn = NeuralNetwork(4,5,3)
    weights = nn.getWeights()
    expectedOutput = nn.sigmoid(np.matrix(np.append([1], inputArr))*weights[0])
    expectedOutput = (nn.sigmoid(np.matrix(np.append([1], expectedOutput)) 
                                          * weights[1]))
    assert(np.array_equal(nn.forwardPropagation(inputArr),expectedOutput))

    # testing neural netwrok with multiple hidden layers
    nn = NeuralNetwork(4,(3,4),3)
    weights = nn.getWeights()
    expectedOutput = nn.sigmoid(np.matrix(np.append([1], inputArr))*weights[0])
    expectedOutput = (nn.sigmoid(np.matrix(np.append([1], expectedOutput)) 
                                          * weights[1]))
    expectedOutput = (nn.sigmoid(np.matrix(np.append([1], expectedOutput)) 
                                          * weights[2]))
    assert(np.array_equal(nn.forwardPropagation(inputArr),expectedOutput))
    print("Passed!")

def testInitWeights():    
    print("Testing initWeights", end = " ")
    nn = NeuralNetwork(0, 0, 0)

    # tests making weights for a Neural Network with no hidden layers
    weights = nn.initWeights(1, 0, 3)
    assert(np.shape(weights[0]) == (2,3))

    # tests making weights for a Neural Network with one hidden layer
    weights = nn.initWeights(2, 3, 1)
    assert(np.shape(weights[0]) == (3,3))
    assert(np.shape(weights[1]) == (4,1))

    # tests making weights for a Neural Network with multiple hidden layers
    weights = nn.initWeights(1,(2,3),4)
    # nn.printWeights(weights)
    assert(np.shape(weights[0]) == (2,2))
    assert(np.shape(weights[1]) == (3,3))
    assert(np.shape(weights[2]) == (4,4))
    print("Passed!")

def testBackPropagation():
    print("Testing backPropagation...", end = "")
    nn = NeuralNetwork(2,2,2)
    w1 = np.matrix([[.35,.35],[.15,.25],[.20,.30]])
    w2 = np.matrix([[.60,.60],[.40,.50],[.45,.55]])
    weights = np.array([w1, w2])
    nn.setWeights(weights)
    inputs = np.array([.05, .10])
    expectedOutputs = np.array([.01, .99])
    learningRate = .5
    difference = expectedOutputs - nn.forwardPropagation(inputs)
    initialError = difference * difference.transpose()
    for i in range(1000):
        nn.backPropagation(inputs, expectedOutputs)
    difference = expectedOutputs - nn.forwardPropagation(inputs)
    finalError = difference * difference.transpose()
    assert(finalError < initialError/100)
    print("Passed!")


def generateLinearData(numExamples):
    # creates data in the form of inputs: (x, y); outputs: (true or false) 
    inputsList = []
    outputsList = []
    for n in range(numExamples):
        inputsList += [[random.random(), random.random()]]
        outputsList += [[1 if inputsList[n][1] > inputsList[n][0] else 0]]
    return (np.matrix(inputsList), np.matrix(outputsList))   

def generateExponentialData(numExamples):
    # creates data in the form of inputs: (x, y); outputs: (true or false) 
    inputsList = []
    outputsList = []
    for n in range(numExamples):
        inputsList += [[random.random(), random.random()]]
        outputsList += [[1 if inputsList[n][1] > inputsList[n][0]**2 else 0]]
    return (np.matrix(inputsList), np.matrix(outputsList))   

def testLearn():
    print("Testing learn...", end = "")
    # tests if the neural network can learn a linear function
    nn = NeuralNetwork(2, 0, 1)
    numCycles = 1000
    examplesPerCycle = 100
    time = np.arange(0, numCycles)
    error = np.zeros(numCycles)
    for cycle in range(numCycles):
        (inputs, expectedOutputs) = generateLinearData(examplesPerCycle)
        nn.learn(inputs, expectedOutputs, .01)
        error[cycle] = nn.getError(inputs, expectedOutputs)

    plt.plot(time, error)
    plt.show()

    # tests if the neural network will work on exponential data
    nn = NeuralNetwork(2, 3, 1)
    numCycles = 1000
    examplesPerCycle = 100
    time = np.arange(0, numCycles)
    error = np.zeros(numCycles)
    for cycle in range(numCycles):
        (inputs, expectedOutputs) = generateExponentialData(examplesPerCycle)
        nn.learn(inputs, expectedOutputs, .01)
        error[cycle] = nn.getError(inputs, expectedOutputs)

    plt.plot(time, error)
    plt.show()


def testAll():
    testInitWeights()
    testSigmoid()
    testforwardPropagation()
    testBackPropagation()
    testLearn()

#testAll()

