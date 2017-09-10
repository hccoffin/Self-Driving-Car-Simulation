# Howard Coffin, hcoffin, Section B
# genetic algorithm

from Car import Car
from Path import Path
import random
import numpy as np

class Population(object):

    def __init__(self, numCars):
        self.generation = 0
        self.numCars = numCars
        self.cars = [Car() for c in range(numCars)]
        self.fitness = [0 for c in range(numCars)]
        self.runGeneration()

    @staticmethod
    def mutate(weights):
        fullMutationRate = .02
        partialMutationRate = .05
        mu = 0
        sigma = .05
        newWeights = []
        # loops through each weight and randomly changes them at certain rates
        numWeights = 0
        for weightLayer in weights:
            weightsList = weightLayer.tolist()
            for row in range(len(weightsList)):
                for col in range(len(weightsList[0])):
                    numWeights += 1
                    randomNumber = random.random()
                    if randomNumber < fullMutationRate:
                        weightsList[row][col] = random.random() * 2 - 1
                    randomNumber = random.random()
                    if randomNumber < partialMutationRate:
                        weightsList[row][col] += random.gauss(mu, sigma)
            newWeights += [np.matrix(weightsList)]
        return newWeights

    def runGeneration(self):
        time = 0
        crashed = [False for c in range(self.numCars)]
        numCrashed = 0
        path = Path(500, 500)
        self.generation += 1
        # drives cars until they crash
        while numCrashed < self.numCars:
            for c in range(self.numCars):
                if not crashed[c]: 
                    # car attempts to drive
                    car = self.cars[c]
                    car.drive(path)

                    # tests if the car crashed
                    maxTime = 5000
                    if car.checkCrash(path) or time >= maxTime:
                        print("Cars Remaining: %d Fitness: %d" % 
                              ((self.numCars-numCrashed), time))
                        crashed[c] = True
                        numCrashed += 1
                        self.fitness[c] = time
                    path.generatePath(car)
            time += 1

    def findSemirandomParents(self):
        # finds "random" parents from this generation, favoring cars which got
        # higher fitnesses
        fitnessPercentages = self.findRegularizedFitnesses()
        randPercent1 = random.random()
        randPercent2 = random.random()
        parent1 = None
        parent2 = None
        for c in range(self.numCars):
            if fitnessPercentages[c] > randPercent1 and parent1 == None:
                parent1 = self.cars[c]
            if fitnessPercentages[c] > randPercent2 and parent2 == None:
                parent2 = self.cars[c]
        return (parent1, parent2)

    def createChildred(self):
        # creates children from the current generation
        children = []
        for c in range(self.numCars):
            parents = self.findSemirandomParents()
            parentWeights = (parents[0].getWeights(), parents[1].getWeights())
            averageWeights = []
            for weightLayer in range(len(parentWeights[0])):
                averageWeights += [(parentWeights[0][weightLayer] + 
                                    parentWeights[1][weightLayer]) / 2]
            childWeights = Population.mutate(averageWeights)
            child = Car()
            child.setWeights(childWeights)
            children += [child]
        return children

    def findRegularizedFitnesses(self):
        # regularizes fitnesses 
        totalSquareFitness = sum(map(lambda x: x**2, self.fitness))
        squareFitnessSum = 0
        regularizedFitnesses = []
        for c in range(self.numCars):
            squareFitnessSum += self.fitness[c]**2
            regularizedFitnesses += [squareFitnessSum / totalSquareFitness]
        return regularizedFitnesses

    def makeNewGeneration(self):
        # creates next generation
        self.cars = self.createChildred()
        # runs the generation
        self.runGeneration()

    def getBestCar(self):
        # returns the best car from the generation
        bestFitness = -1
        bestCar = None
        for c in range(self.numCars):
            if self.fitness[c] > bestFitness:
                bestFitness = self.fitness[c]
                bestCar = self.cars[c]
        bestCar.x = 0
        bestCar.y = 0
        bestCar.direction = 0
        print("Generation %d, Best Fitness: %d" % (self.generation,bestFitness))
        return bestCar

    def getHighestFitness(self):
        bestFitness = -1
        for c in range(self.numCars):
            if self.fitness[c] > bestFitness:
                bestFitness = self.fitness[c]
        return bestFitness
