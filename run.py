# Howard Coffin, hcoffin, Section B
# run function and general layout taken from 15-112 course website

from tkinter import *
from Path import Path
from Car import Car
from Population import Population
from Button import Button
import math
import numpy as np

def init(data):
    data.car = Car()
    data.mode = "main menu"
    makeButtons(data)

    # backprop variables
    data.controls = np.matrix([0, 0])
    data.controlTimer = data.timerDelay * 2

    # genetic algorithm vairbales
    data.generation = 0
    data.highestFitness = 0
    #data.learning = False
    data.numCars = 50
    data.showNeuralNetwork = False


def resetRun(data):
    data.car.x = 0
    data.car.y = 0
    data.car.direction = 0
    data.path = Path(data.width, data.height)

def mousePressedSplashScreen(event, data):
    if data.trainButton.pressed(event):
        data.mode = "train"
    elif data.runButton.pressed(event):
        data.mode = "driving"
        resetRun(data)
        data.pause = False
        data.carInControl = True
        data.timerDelay = 0
    elif data.howItWorksButton.pressed(event):
        data.mode = "how it works"

def mousePressedHowItWorks(event, data):
    if data.howItWorksToMainMenuButton.pressed(event):
        data.mode = "main menu"

def mousePressedTrain(event, data):
    if data.trainToMainMenuButton.pressed(event):
        data.mode = "main menu"
    elif data.trainYourselfButton.pressed(event):
        data.mode = "train it yourself readme"
    elif data.learnAutoButton.pressed(event):
        data.mode = "learn auto"

def mousePressedTrainItYourselfReadme(event, data):
    data.mode = "driving"
    resetRun(data)
    data.pause = False
    data.carInControl = False
    data.timerDelay = 50

def mousePressedDriving(event, data):
    if data.drivingToMainMenuButton.pressed(event):
        data.mode = "main menu"
    elif data.showBrainButton.pressed(event):
        data.showNeuralNetwork = not data.showNeuralNetwork
    elif data.pauseButton.pressed(event):
        data.pause = not data.pause

def mousePressedLearnAuto(event, data):
    if data.nextGenerationButton.pressed(event):
        if data.generation == 0:
            data.population = Population(data.numCars)
        else:
            data.population.makeNewGeneration()
        data.generation += 1
        data.car = data.population.getBestCar()
        data.highestFitness = data.population.getHighestFitness()
    if data.learnAutoToMainMenuButton.pressed(event):
        data.mode = "main menu"

def mousePressed(event, data):
    if data.mode == "main menu":
        mousePressedSplashScreen(event, data)
    elif data.mode == "train":
        mousePressedTrain(event, data)
    elif data.mode == "train it yourself readme":
        mousePressedTrainItYourselfReadme(event, data)
    elif data.mode == "driving":
        mousePressedDriving(event, data)
    elif data.mode == "learn auto":
        mousePressedLearnAuto(event, data)
    elif data.mode == "how it works":
        mousePressedHowItWorks(event, data)


def keyPressedDriving(event, data):
    if event.keysym == "Left":
        data.controls = np.matrix([1, 0])
        data.controlTimer = data.timerDelay * 2
    elif event.keysym == "Right":
        data.controls = np.matrix([0, 1])
        data.controlTimer = data.timerDelay * 2

def keyPressed(event, data):
    if data.mode == "driving":
        keyPressedDriving(event, data)
    if data.mode == "learn auto":
        data.car.setWeights(data.weights)
        data.generation = data.gen
        data.highestFitness = data.fitness


def timerFiredDriving(data):
    data.path.generatePath(data.car)

    if data.carInControl:
        data.car.drive(data.path)
    else:
        # smooths out steering
        if data.controlTimer <= 0:
            data.controls = np.matrix([0, 0])
        if data.controlTimer >= -1 * data.timerDelay:
            data.car.learn(data.path, data.controls)
        data.controlTimer -= data.timerDelay
        # car drives using user input
        data.car.steer(data.controls)
        data.car.move()

def timerFired(data):
    if data.mode == "driving":
        if not data.pause:
            timerFiredDriving(data)


def drawSplashScreen(canvas, data):
    text = "Self Driving\nCar Simulation"
    font = "Arial 60"
    position = (data.width/2, 300)
    canvas.create_text(position, text = text, font = font, 
                       fill = "white", justify = "center")
    data.trainButton.draw(canvas)
    data.runButton.draw(canvas)
    data.howItWorksButton.draw(canvas)

def drawHowItWorksScreen(canvas, data):
    text = "How it Works"
    font = "Arial 60"
    position = (data.width/2, 80)
    canvas.create_text(position, text = text, font = font, 
                       fill = "white", justify = "center")
    text = """
    Driving: 
        The car is steered using a neural network which acts as its "brain". The 
     first layer of the network are the inputs--what the car can see--and the 
     last layer is the outputs--telling the car to turn left or right.The middle 
     layer, called the hidden layer, is somewhat more complicated. It does not 
     directly control anything, but does some processing of the input data to 
     help the car drive. For this reason, it is hard to understand what the 
     values of the hidden layer really mean. The inputs of the neural network 
     for the car is "visual" information about the road.

    Learning: 
        The brain of the car can be trained using two different methods.
        The first of these methods is called backpropagation and is used in 
     "Train it Yourself". In this method, the car takes in visual input and 
     tries to come up with the steering output. At the same time, the user puts 
     in their own output by steering the car. The car then finds the difference 
     between how it steered the car and how the the user steered and uses this 
     information to correct itself.
        The second method is known as a genetic algorithm, and is used in 
     "Learn Automatically". In this method, a generation of %d cars is made
     randomly to start out with. These cars are run down a path until they 
     crash. The next generation is composed of cars which have similar 
     characteristics to the best cars of the last generation as well as some 
     random mutation to keep some variation. The car which went the farthest
     before crashing from any generation can be run from the main menu.
    """ % (data.numCars)
    font = "Arial 14"
    position = (data.width/2, data.height/2)
    canvas.create_text(position, text = text, font = font, 
                       fill = "white")


    data.howItWorksToMainMenuButton.draw(canvas)

def drawTrainScreen(canvas, data):
    text = "Train the Car"
    font = "Arial 60"
    position = (data.width/2, 150)
    canvas.create_text(position, text = text, font = font, 
                       fill = "white", justify = "center")
    data.trainYourselfButton.draw(canvas)
    data.learnAutoButton.draw(canvas)
    data.trainToMainMenuButton.draw(canvas)

def drawTrainItYouselfScreen(canvas, data):
    text = """    Use the left and right
    arrow keys to drive. 
    The longer you drive, 
    the better the car 
    will drive.


    Click to Continue"""
    font = "Arial 50"
    position = (data.width/2 - 50, data.height/2)
    canvas.create_text(position, text = text, font = font, 
                       fill = "white", justify = "center")

def drawDrivingScreen(canvas, data):
    data.path.draw(canvas, data.car.x - data.car.screenX, data.car.y)
    data.car.draw(canvas, data.height)
    if data.showNeuralNetwork:
        data.car.drawNeuralNetwork(canvas, data.height, data.path)
    # draws a collision warning if car off road
    if data.car.checkCrash(data.path):
        boxWidth, boxHeight = 100, 40
        collisionBoxCorners = [(data.width-boxWidth, 0), 
                              (data.width, boxHeight)]
        textLocation = (data.width - boxWidth/2, boxHeight/2)
        canvas.create_rectangle(collisionBoxCorners, fill = "red", 
                                outline = "red")
        canvas.create_text(textLocation, text = "Off Road!", 
                           fill = "white", font = "Arial 15")

    if data.pause:
        canvas.create_rectangle(0, 0, data.width, data.height, 
                                fill = "gray", stipple = "gray25")

    data.drivingToMainMenuButton.draw(canvas)
    data.showBrainButton.draw(canvas)
    data.pauseButton.draw(canvas)

def drawAutoLearningScreen(canvas, data):
    # title
    text = "Learn Automatically"
    font = "Arial 60"
    position = (data.width/2, 150)
    canvas.create_text(position, text = text, font = font, 
                       fill = "white", justify = "center")
    # information
    fitness = str(data.highestFitness) if data.highestFitness<5000 else "5000+" 
    text = ("Generation: %d     Best Distance: %s" % 
            (data.generation, fitness))
    font = "Arial 30"
    position = (data.width/2, 250)
    canvas.create_text(position, text = text, font = font, 
                       fill = "white", justify = "center")
    data.nextGenerationButton.draw(canvas)
    data.learnAutoToMainMenuButton.draw(canvas)

def redrawAll(canvas, data):
    if data.mode == "main menu":
        drawSplashScreen(canvas, data)
    elif data.mode == "how it works":
        drawHowItWorksScreen(canvas, data)
    elif data.mode == "train":
        drawTrainScreen(canvas, data)
    elif data.mode == "train it yourself readme":
        drawTrainItYouselfScreen(canvas, data)
    elif data.mode == "driving":   
        drawDrivingScreen(canvas, data)
    elif data.mode == "learn auto":
        drawAutoLearningScreen(canvas, data)


def makeMainMenuButtons(data):
    (x, y, width, height) = (100, 600, 150, 100)
    data.trainButton = Button(x, y, width, height, "Train the\nCar")
    (x, y, width, height) = (750, 600, 150, 100)
    data.runButton = Button(x, y, width, height, "Run the\nCar")
    (x, y, width, height) = (data.width/2 - 150, 700, 300, 80)
    data.howItWorksButton = Button(x, y, width, height, "How it Works")

def makeHowItWorksButtons(data):
    (x, y, width, height) = (data.width/2 - 150, 700, 300, 80)
    data.howItWorksToMainMenuButton = Button(x, y, width, height, "Main Menu")

def makeTrainButtons(data):
    (x, y, width, height) = (data.width/2 - 150, 300, 300, 80)
    data.trainYourselfButton = Button(x, y, width, height, "Train it Yourself")
    (x, y, width, height) = (data.width/2 - 150, 450, 300, 80)
    data.learnAutoButton = Button(x, y, width, height, "Learn Automatically")
    (x, y, width, height) = (data.width/2 - 150, 600, 300, 80)
    data.trainToMainMenuButton = Button(x, y, width, height, "Main Menu")

def makeDrivingButtons(data):
    (x, y, width, height) = (10, 10, 150, 70)
    data.drivingToMainMenuButton = Button(x, y, width, height, "Main Menu")
    (x, y, width, height) = (170, 10, 150, 70)
    data.showBrainButton = Button(x, y, width, height, "Show Brain")
    (x, y, width, height) = (330, 10, 150, 70)
    data.pauseButton = Button(x, y, width, height, "Pause")

def makeGeneticAlgorithmButtons(data):
    (x, y, width, height) = (data.width/2 - 150, 450, 300, 80)
    data.nextGenerationButton = Button(x, y, width, height, "Next Generation")
    (x, y, width, height) = (data.width/2 - 150, 600, 300, 80)
    data.learnAutoToMainMenuButton = Button(x, y, width, height, "Main Menu")

def makeButtons(data):
    makeMainMenuButtons(data)
    makeHowItWorksButtons(data)
    makeTrainButtons(data)
    makeDrivingButtons(data)
    makeGeneticAlgorithmButtons(data)
    


# run function taken from 15-112 course website
def run(width=300, height=300):
    def redrawAllWrapper(canvas, data):
        canvas.delete(ALL)
        canvas.create_rectangle(0, 0, data.width, data.height,
                                fill='SpringGreen3', width=0)
        redrawAll(canvas, data)
        canvas.update()    

    def mousePressedWrapper(event, canvas, data):
        mousePressed(event, data)
        redrawAllWrapper(canvas, data)

    def keyPressedWrapper(event, canvas, data):
        keyPressed(event, data)
        redrawAllWrapper(canvas, data)

    def timerFiredWrapper(canvas, data):
        timerFired(data)
        redrawAllWrapper(canvas, data)
        # pause, then call timerFired again
        canvas.after(data.timerDelay, timerFiredWrapper, canvas, data)
    # Set up data and call init
    class Struct(object): pass
    data = Struct()
    data.width = width
    data.height = height
    data.timerDelay = 50 # milliseconds
    init(data)
    # create the root and the canvas
    root = Tk()
    canvas = Canvas(root, width=data.width, height=data.height)
    canvas.pack()
    # set up events
    root.bind("<Button-1>", lambda event:
                            mousePressedWrapper(event, canvas, data))
    root.bind("<Key>", lambda event:
                            keyPressedWrapper(event, canvas, data))
    timerFiredWrapper(canvas, data)
    # and launch the app
    root.mainloop()  # blocks until window is closed
    print("bye!")

run(1000, 800)