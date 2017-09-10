# Howard Coffin, hcoffin, Section B

import random
import math

class Path(object):

    def __init__(self, screenWidth, screenHeight):
        self.pathWidth = 200
        self.path = []
        self.margin = 200
        self.velocity = (1.0, 0.0)
        self.position = (-1 * self.margin, 0.0)
        self.maxVelocity = .5
        self.screenWidth = screenWidth
        self.screenHeight = screenHeight
        self.generateStraitSection(screenWidth + 2 * self.margin)

    def moveForward(self, strait = False):
        # changes position
        self.position =  (self.position[0] + self.velocity[0], 
                          self.position[1] + self.velocity[1])
        if strait:
            deltaVel = 0
        else:
            # randomly changes velocity in y direction over time
            mu = 0
            sigma = .01
            deltaVel = None
            while(deltaVel == None or 
                  math.fabs(self.velocity[1] + deltaVel) > self.maxVelocity):
                deltaVel = random.gauss(mu, sigma)
        self.velocity = (self.velocity[0], self.velocity[1] + deltaVel)

    def generateSection(self, sectionLength):
        for x in range(sectionLength):
            self.path.pop(0)
            normalVector = (-1 * self.velocity[1], self.velocity[0])
            self.path += [(self.position, normalVector)]
            self.moveForward()

    def generateStraitSection(self, sectionLength):
        for x in range(sectionLength):
            normalVector = (-1 * self.velocity[1], self.velocity[0])
            self.path += [(self.position, normalVector)]
            self.moveForward(True)

    def generatePath(self, car):
        carX = car.x
        carScreenX = car.screenX
        currentX = self.position[0] - self.margin - self.screenWidth
        sectionLength = math.floor(carX - currentX)
        self.generateSection(sectionLength)

    @staticmethod
    def distance(point1, point2):
        # distance between two points
        return (((point1[0]-point2[0])**2 + (point1[1] -point2[1])**2)**.5)

    def findXIndex(self, x):
        # returns a point on path for a given x position
        (p0, norm0) = self.path[0]
        (x0, y0) = p0
        index = int(x-x0)
        return index

    def onPath(self, point):
        # collision detection
        maxDistance = math.ceil(math.tan(self.maxVelocity)*self.pathWidth)
        (x, y) = point
        xIndex = self.findXIndex(x)
        for p in range(max(xIndex-maxDistance , 0), 
                       min(xIndex+maxDistance, len(self.path)-1)):
            if Path.distance(self.path[p][0], point) < self.pathWidth/2:
                return True
        return False        

    def getBoundaryPoints(self, point):
        # returns points which are on the edges of the road
        # returns in the form (higher point, lower point)
        ((x, y), (normX, normY)) = point
        normMagnitude = (normX**2 + normY**2)**.5
        unitNorm = (normX/normMagnitude, normY/normMagnitude)
        p1 = (x-self.pathWidth/2*unitNorm[0], y-self.pathWidth/2*unitNorm[1])
        p2 = (x+self.pathWidth/2*unitNorm[0], y+self.pathWidth/2*unitNorm[1])
        if p1[1] > p2[1]: return (p1, p2)
        else: return (p2, p1)

    def getOutOfBoundaryPoints(self, point):
        # returns points which are on the edges of the road
        # returns in the form (higher point, lower point)
        # for testing only
        ((x, y), (normX, normY)) = point
        normMagnitude = (normX**2 + normY**2)**.5 * .99
        unitNorm = (normX/normMagnitude, normY/normMagnitude)
        p1 = (x-self.pathWidth/2*unitNorm[0],y-self.pathWidth/2*unitNorm[1])
        p2 = (x+self.pathWidth/2*unitNorm[0],y+self.pathWidth/2*unitNorm[1])
        if p1[1] > p2[1]: return (p1, p2)
        else: return (p2, p1)

    def draw(self, canvas, xOffset, yOffset):
        lastSection = self.getBoundaryPoints(self.path[0])
        for p in range(1, len(self.path)):
            section = self.getBoundaryPoints(self.path[p])
            points = []
            points += (lastSection[0][0] - xOffset, 
                       lastSection[0][1] + self.screenHeight//2 - yOffset)
            points += (section[0][0] - xOffset, 
                       section[0][1] + self.screenHeight//2 - yOffset)
            points += (section[1][0] - xOffset, 
                       section[1][1] + self.screenHeight//2 - yOffset)
            points += (lastSection[1][0] - xOffset, 
                       lastSection[1][1] + self.screenHeight//2 - yOffset)
            if lastSection[0][0] == 0:
                canvas.create_polygon(points, fill = "red")
            else:
                canvas.create_polygon(points, fill = "dim gray")
            lastSection = section


def testOnPath():
    print("Testing onPath", end = "...")
    path = Path(1000, 1000)
    path.generateSection(1000)
    for p in range(path.margin, len(path.path)-path.margin):
        (point, norm) = path.path[p]
        assert(path.onPath(point))
        (p1, p2) = path.getBoundaryPoints(path.path[p])
        assert(path.onPath(p1))
        assert(path.onPath(p2))
        (p3, p4) = path.getOutOfBoundaryPoints(path.path[p])
        assert(not path.onPath(p3))
        assert(not path.onPath(p4))
    print("Passed!")

def testAll():
    testOnPath()

#testAll()


