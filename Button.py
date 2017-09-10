# Howard Coffin, hcoffin, Section B
# pressable button

class Button(object):

    def __init__(self, x, y, width, height, text, 
                 color = "orange red", font = "Arial 20"):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.text = text
        self.color = color
        self.font = font

    @staticmethod
    def drawCenteredCircle(canvas, x, y, radius, color):
        topLeft = (x - radius, y - radius)
        bottomRight = (x + radius, y + radius)
        canvas.create_oval(topLeft, bottomRight, fill = color)

    def drawCornerCircles(self, canvas, cornerRadius):
        Button.drawCenteredCircle(canvas, self.x + cornerRadius, 
                                  self.y + cornerRadius, 
                                  cornerRadius, self.color)
        Button.drawCenteredCircle(canvas, self.x + self.width - cornerRadius, 
                                  self.y + cornerRadius, 
                                  cornerRadius, self.color)
        Button.drawCenteredCircle(canvas, self.x + cornerRadius, 
                                  self.y + self.height - cornerRadius, 
                                  cornerRadius, self.color)
        Button.drawCenteredCircle(canvas, self.x + self.width - cornerRadius, 
                                  self.y + self.height - cornerRadius, 
                                  cornerRadius, self.color)

    def draw(self, canvas):
        # draws a rounded rectangle with text in it
        cornerRadius = 10
        # draws corner circles
        self.drawCornerCircles(canvas, cornerRadius)
        # draws center rectangles
        firstRect = (self.x + cornerRadius, self.y, 
                     self.x + self.width - cornerRadius, self.y + self.height)
        secondRect = (self.x, self.y + cornerRadius,
                      self.x + self.width, self.y + self.height - cornerRadius)
        canvas.create_rectangle(firstRect, fill = self.color, width = 0)
        canvas.create_rectangle(secondRect, fill = self.color, width = 0)
        # creates outline
        canvas.create_line(firstRect[0], firstRect[1], 
                           firstRect[2], firstRect[1])
        canvas.create_line(firstRect[0], firstRect[3], 
                           firstRect[2], firstRect[3])
        canvas.create_line(secondRect[0], secondRect[1], 
                           secondRect[0], secondRect[3])
        canvas.create_line(secondRect[2], secondRect[1], 
                           secondRect[2], secondRect[3])
        # draws text
        canvas.create_text(self.x+self.width/2,self.y+self.height/2,
                         text = self.text, fill = "white", font = self.font,
                         justify = "center")

    def pressed(self, event):
        (x, y) = (event.x, event.y)
        return (x > self.x and y > self.y 
                and x < self.x + self.width and y < self.y + self.height)
