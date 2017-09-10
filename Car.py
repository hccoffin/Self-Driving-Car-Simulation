# Howard Coffin, hcoffin, Section B

from NeuralNetwork import NeuralNetwork
import math
import numpy as np

class Car(object):

    def __init__(self):
        self.numInputNodes = 15
        numHiddenNodes = 10
        numOutputNodes = 2
        self.neuralNet = NeuralNetwork(self.numInputNodes, numHiddenNodes, 
                                       numOutputNodes)
        self.direction = 0 # in radians
        self.vel = 60.0
        # car position is the middle of its back side
        self.screenX = 100
        self.x = 0.0
        self.y = 0.0
        self.width = 100
        self.length = 160

    def getInputs(self, path):
        inputsList = []
        for inputPosition in self.getInputPositions():
            inputsList += [1 if path.onPath(tuple(inputPosition)) else 0]
        return np.matrix(inputsList)

    def getInputPositions(self):
        tangentUnitVector = np.array([math.cos(self.direction), 
                                      -1 * math.sin(self.direction)])
        normalUnitVector = np.array([math.sin(self.direction), 
                                     math.cos(self.direction)])
        carFront = np.array([self.x, self.y]) + tangentUnitVector * self.length

        inputPositions = []
        for i in range(self.numInputNodes):
            ellipseR1 = self.length*2
            ellipseR2 = self.width*1.5
            angle = (i * (math.pi * 1.1 / (self.numInputNodes-1))) - math.pi*.55
            tangentialPortion = math.cos(angle) * ellipseR1
            normalPortion = math.sin(angle) * ellipseR2
            inputPositions += [carFront + normalUnitVector * normalPortion + 
                               tangentUnitVector * tangentialPortion]

        return inputPositions

    def move(self):
        self.x += math.cos(self.direction) * self.vel
        self.y -= math.sin(self.direction) * self.vel

    def getCornerPositions(self, x, y):
        bottomLeft = (x + self.width * math.sin(self.direction) / 2,
                      y + self.width * math.cos(self.direction) / 2)
        topLeft = (x - self.width * math.sin(self.direction) / 2,
                   y - self.width * math.cos(self.direction) / 2)
        bottomRight = (bottomLeft[0] + math.cos(self.direction) * self.length,
                       bottomLeft[1] - math.sin(self.direction) * self.length)
        topRight = (topLeft[0] + math.cos(self.direction) * self.length,
                    topLeft[1] - math.sin(self.direction) * self.length)
        return (bottomLeft, topLeft, topRight, bottomRight)

    def getPosition(self, x, y, infront, side):
        unitTangentDirection = (math.cos(self.direction), 
                                math.sin(self.direction))
        unitNormalDirection  = (-1 * math.sin(self.direction), 
                                math.cos(self.direction))
        posX = (x + unitTangentDirection[0] * infront + 
                unitNormalDirection[0] * side)
        posY = (y - unitTangentDirection[1] * infront - 
                unitNormalDirection[1] * side)
        return (posX, posY)

    def drawBackWheels(self, canvas, screenY, infront, side):
        leftWheels = []
        leftWheels += self.getPosition(self.screenX, screenY, 
                                             infront[0], side[0])
        leftWheels += self.getPosition(self.screenX, screenY, 
                                             infront[0], side[1])
        leftWheels += self.getPosition(self.screenX, screenY, 
                                             infront[1], side[1])
        leftWheels += self.getPosition(self.screenX, screenY, 
                                             infront[1], side[0])
        canvas.create_polygon(leftWheels, fill = "black")

        rightWheel = []
        rightWheel += self.getPosition(self.screenX, screenY, 
                                              infront[0], side[2])
        rightWheel += self.getPosition(self.screenX, screenY, 
                                              infront[0], side[3])
        rightWheel += self.getPosition(self.screenX, screenY, 
                                              infront[1], side[3])
        rightWheel += self.getPosition(self.screenX, screenY, 
                                              infront[1], side[2])
        canvas.create_polygon(rightWheel, fill = "black")

    def drawFrontWheels(self, canvas, screenY, infront, side):
        leftWheels = []
        leftWheels += self.getPosition(self.screenX, screenY, 
                                             infront[2], side[0])
        leftWheels += self.getPosition(self.screenX, screenY, 
                                             infront[2], side[1])
        leftWheels += self.getPosition(self.screenX, screenY, 
                                             infront[3], side[1])
        leftWheels += self.getPosition(self.screenX, screenY, 
                                             infront[3], side[0])
        canvas.create_polygon(leftWheels, fill = "black")

        rightWheel = []
        rightWheel += self.getPosition(self.screenX, screenY, 
                                              infront[2], side[2])
        rightWheel += self.getPosition(self.screenX, screenY, 
                                              infront[2], side[3])
        rightWheel += self.getPosition(self.screenX, screenY, 
                                              infront[3], side[3])
        rightWheel += self.getPosition(self.screenX, screenY, 
                                              infront[3], side[2])
        canvas.create_polygon(rightWheel, fill = "black")

    def drawWheels(self, canvas, screenY):
        infront = [.1 * self.length, .35 * self.length, 
                   .65 * self.length, .9 * self.length]
        side = [self.width * .48, self.width * .6,
                self.width * -.48, self.width *-.6]
        self.drawBackWheels(canvas, screenY, infront, side)
        self.drawFrontWheels(canvas, screenY, infront, side)

    @staticmethod
    def drawCenterCircle(canvas, x, y, radius, color):
        canvas.create_oval(x - radius, y - radius, x + radius, y + radius,
                           fill = color, width = 0)

    def drawRoundCorners(self, canvas, screenY, cornerRadius):
        bottomLeftCorner = self.getPosition(self.screenX, screenY,
                            cornerRadius, -self.width * .5 + cornerRadius + 1)
        self.drawCenterCircle(canvas, bottomLeftCorner[0], bottomLeftCorner[1], 
                              cornerRadius, "dark orange")

        topLeftCorner = self.getPosition(self.screenX, screenY,
                            cornerRadius, self.width * .5 - cornerRadius)
        self.drawCenterCircle(canvas, topLeftCorner[0], topLeftCorner[1], 
                              cornerRadius, "dark orange")

        topRightCorner = self.getPosition(self.screenX, screenY,
            self.length - cornerRadius - 1, self.width*.5 - cornerRadius)
        self.drawCenterCircle(canvas, topRightCorner[0], topRightCorner[1], 
                              cornerRadius, "dark orange")

        bottomRightCorner = self.getPosition(self.screenX, screenY,
            self.length - cornerRadius - 1, -self.width*.5 + cornerRadius + 1)
        self.drawCenterCircle(canvas, bottomRightCorner[0],bottomRightCorner[1], 
                              cornerRadius, "dark orange")

    def drawCar(self, canvas, screenY):
        cornerRadius = math.floor(.15 * self.length)
        # draws a rounded rectangle for the car
        firstRect = []
        firstRect += [self.getPosition(self.screenX, screenY,
                                      0, -self.width * .5 + cornerRadius)]
        firstRect += [self.getPosition(self.screenX, screenY,
                                      0, self.width * .5 - cornerRadius)]
        firstRect += [self.getPosition(self.screenX, screenY,
                                      self.length, self.width*.5-cornerRadius)]
        firstRect += [self.getPosition(self.screenX, screenY,
                                      self.length, -self.width*.5+cornerRadius)]
        canvas.create_polygon(firstRect, fill = "dark orange")

        secondRect = []
        secondRect += [self.getPosition(self.screenX, screenY,
                                      cornerRadius, -self.width/2)]
        secondRect += [self.getPosition(self.screenX, screenY,
                                      cornerRadius, self.width/2)]
        secondRect += [self.getPosition(self.screenX, screenY,
                                      self.length - cornerRadius, self.width/2)]
        secondRect += [self.getPosition(self.screenX, screenY,
                                    self.length - cornerRadius, -self.width/2)]
        canvas.create_polygon(secondRect, fill = "dark orange")

        self.drawRoundCorners(canvas, screenY, cornerRadius)
    
    def checkCrash(self, path):
        corners = self.getCornerPositions(self.x, self.y)
        for corner in corners:
            if not path.onPath(corner):
                return True
        return False

    def draw(self, canvas, screenHeight):
        # draws the car
        screenY = screenHeight/2
        self.drawCar(canvas, screenY)
        self.drawWheels(canvas, screenY)

    def drawNeuralNetwork(self, canvas, screenHeight, path):
        # draws the inputs and the neural network
        screenY = screenHeight/2
        for inputPosition in self.getInputPositions():
            (deltaX, deltaY) = tuple(inputPosition - np.array([self.x, self.y]))
            dotX = self.screenX + deltaX
            dotY = screenY + deltaY
            dotRadius = 3
            road = path.onPath(tuple(inputPosition))

            canvas.create_oval(dotX - dotRadius, dotY - dotRadius, 
                               dotX + dotRadius, dotY + dotRadius,
                               fill = "white" if road else "black")

        (x, y, width, height) = (20, 480, 300, 300)
        self.neuralNet.draw(canvas, self.getInputs(path), x, y, width, height)

    def drive(self, path):
        inputs = self.getInputs(path)
        controls = self.neuralNet.forwardPropagation(inputs)
        self.steer(controls)
        self.move()

    def steer(self, controls):
        sigmoidTrue = .5
        acceleration = math.pi/100

        if controls.item(0) >= sigmoidTrue:
            # turn left
            self.direction += acceleration
        if controls.item(1) >= sigmoidTrue:
            # turn right
            self.direction -= acceleration

    def learn(self, path, expectedControls):
        inputs = self.getInputs(path)
        learningRate = .5
        self.neuralNet.learn(inputs, expectedControls, learningRate)

    def getWeights(self):
        # returns neural net weights
        return self.neuralNet.getWeights()

    def setWeights(self, weights):
        self.neuralNet.setWeights(weights)