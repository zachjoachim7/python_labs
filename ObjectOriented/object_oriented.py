# object_oriented.py
"""Python Essentials: Object Oriented Programming.
<Zach Joachim>
<Math 345>
<September 13, 2022>
"""

import math

class Backpack:
    """A Backpack object class. Has a name and a list of contents.

    Attributes:
        name (str): the name of the backpack's owner.
        contents (list): the contents of the backpack.
        color (str): the color of the backpack.
        max_size (int): the maximum number of contents in the backpack.
    """

    # Problem 1: Modify __init__() and put(), and write dump().
    def __init__(self, name, color, max_size=5):
        """Set the name and initialize an empty list of contents.

        Parameters:
            name (str): the name of the backpack's owner.
            color (str): the color of the backpack.
            max_size (int): the maximum number of contents in the backpack.
        """
        self.name = name
        self.contents = []
        self.color = color
        self.max_size = max_size                        # Adding max_size as an attribute

    def put(self, item):
        """Add an item to the backpack's list of contents if the backpack is not full already."""
        if (len(self.contents) < self.max_size):        # Must be < sign or it will overload the backpack
            self.contents.append(item)
        else:
            print("No Room!")

    def take(self, item):
        """Remove an item from the backpack's list of contents."""
        self.contents.remove(item)

    def dump(self):
        """Clear the contents of the backpack and return empty list of contents"""
        while self.contents:
            self.contents.pop()
        return self.contents                            # Popping off the last item until the list is empty


    # Magic Methods -----------------------------------------------------------

    # Problem 3: Write __eq__() and __str__().
    def __add__(self, other):
        """Add the number of contents of each Backpack."""
        return len(self.contents) + len(other.contents)

    def __lt__(self, other):
        """Compare two backpacks. If 'self' has fewer contents
        than 'other', return True. Otherwise, return False.
        """
        return len(self.contents) < len(other.contents)

    def __eq__(self, other):
        """Compare two backpacks. If 'self' and 'other' have the same 
        number of contents, same name, and same color, return True, otherwise False
        """
        return len(self.contents) == len(other.contents) and (self.name == other.name) and (self.color == other.color)

    def __str__(self):
        """Prints the object as a string, this function manipulates
        how the information is presented making it easier to read
        """
        ObjectString = "Owner:\t\t" + self.name
        ObjectString += "\nColor:\t\t" + self.color
        ObjectString += "\nSize:\t\t" + str(len(self.contents))
        ObjectString += "\nMax Size:\t" + str(self.max_size)
        ObjectString += "\nContents:\t" + ' '.join(self.contents)
        return ObjectString


# An example of inheritance. You are not required to modify this class.
class Knapsack(Backpack):
    """A Knapsack object class. Inherits from the Backpack class.
    A knapsack is smaller than a backpack and can be tied closed.

    Attributes:
        name (str): the name of the knapsack's owner.
        color (str): the color of the knapsack.
        max_size (int): the maximum number of items that can fit inside.
        contents (list): the contents of the backpack.
        closed (bool): whether or not the knapsack is tied shut.
    """
    def __init__(self, name, color):
        """Use the Backpack constructor to initialize the name, color,
        and max_size attributes. A knapsack only holds 3 item by default.

        Parameters:
            name (str): the name of the knapsack's owner.
            color (str): the color of the knapsack.
            max_size (int): the maximum number of items that can fit inside.
        """
        Backpack.__init__(self, name, color, max_size=3)
        self.closed = True

    def put(self, item):
        """If the knapsack is untied, use the Backpack.put() method."""
        if self.closed:
            print("I'm closed!")
        else:
            Backpack.put(self, item)

    def take(self, item):
        """If the knapsack is untied, use the Backpack.take() method."""
        if self.closed:
            print("I'm closed!")
        else:
            Backpack.take(self, item)

    def weight(self):
        """Calculate the weight of the knapsack by counting the length of the
        string representations of each item in the contents list.
        """
        return sum(len(str(item)) for item in self.contents)


# Problem 2: Write a 'Jetpack' class that inherits from the 'Backpack' class.
class Jetpack(Backpack):
    """Jetpack object class. Inherits from the Backpack class.
    A Jetpack is a backpack that flies.

    Attributes: 
    name (str): the name of the Jetpack's owner.
    color (str): the color of the Jetpack.
    contents (list): list of the contents in the Jetpack.
    max_size (int): the max amount of contents in Jetpack.
    fuel_amount (int): the max fuel amount allowed in Jetpack.
    """
    def __init__(self, name, color, fuel_amount=10):
        """Using the Backpack constructor to initialize the name, color,
        and max_size attributes. A Jetpack only holds 2 items by default
        and has defualt fuel value of 10.
            
            Parameters:
                name (str): the name of the knapsack's owner.
                color (str): the color of the knapsack.
                max_size (int): the maximum number of items that can fit inside.
                fuel_amount (int): the fuel value.
        """
        Backpack.__init__(name, color, max_size=2)              # Overriding the JetPack contructor to only allow
        self.fuel_amount = fuel_amount                          # a max of 2 items.

    def fly(self, fuel_burned):
        """Taking in fuel to be burned and subtracting it from fuel_amount.
        If the user tries to burn more fuel than is available, we print
        "Not enough fuel".
        """
        if (fuel_burned > self.fuel_amount):                # Must be > since you would get a negative fuel_amount
            print("Not enough fuel!")                       # from subtraction.
        else:                                                    
            self.fuel_amount -= fuel_burned

    def dump(self):
        """Override the dump() function from backpack to clear fuel_amount"""
        self.fuel_amount = 0
        return self.fuel_amount

# Problem 4: Write a 'ComplexNumber' class.
class ComplexNumber:
    """Complex numbers are denoted 'a+bi' where a and b are real and 
    i = sqrt(-1). It has a real part and an imaginary part.
    
    Attributes:
        real (int): real part of the complex number (a)
        imag (int): imaginary part of the complex number (b)
    """
    def __init__(self, real, imag):
        """Constructor call for the complex number.

            Parameters:
                real (int): real part of the complex number (a)
                imag (int): imaginary part of the complex number (b)
        """
        self.real = real
        self.imag = imag

    def conjugate(self):
        """Creates a new ComplexNumber object that is the conjugate the original."""
        newComplexNumber = ComplexNumber(self.real, (-1)*self.imag)
        return newComplexNumber

    def __str__(self):
        """Prints the string representation of a ComplexNumber object."""
        if (self.imag >= 0):
            ObjectString = "(" + str(self.real) + "+" + str(self.imag) + "j)"
        else:
            ObjectString = "(" + str(self.real) + str(self.imag) + "j)"     # If imaginary part is less than 0, you do not
        return ObjectString                                                 # need to print an additional "-" sign.

    def __abs__(self):
        """Returns the absolute value of the ComplexNumber"""
        return math.sqrt((self.real)**2 + (self.imag)**2)       # Follows equation of sqrt(a^2 + b^2)

    def __eq__(self, other):
        """Checks to see if the real components are the same and the imaginary componets. Returns True or False."""
        return (self.real == other.real) and (self.imag == other.imag)

    def __add__(self, other):
        """Takes 2 ComplexNumber objects and performs addition of the complex numbers. Returns new ComplexNuber."""
        totalReal = self.real + other.real                  # Simply adding like terms
        totalImag = self.imag + other.imag
        sumComplex = ComplexNumber(totalReal, totalImag)

        return sumComplex

    def __sub__(self, other):
        """Takes 2 ComplexNumber objects and performs subtraction of the complex numbers. Returns new ComplexNuber."""
        diffReal = self.real - other.real                   # Simply subtracting like terms
        diffImag = self.imag - other.imag
        diffComplex = ComplexNumber(diffReal, diffImag)

        return diffComplex

    def __mul__(self, other):
        """Takes 2 ComplexNumber objects and performs multiplication of complex numbers. Returns new ComplexNuber."""
        productReal = self.real * other.real                # From the equations of the complex numbers, a+bi and c+di,
        productImag = self.imag * other.imag                # I am returning (ac-bd)/(c^2 + d^2)
        productComplex = ComplexNumber((productReal - productImag), ((self.real * other.imag) + (self.imag * other.real)))

        return productComplex

    def __truediv__(self, other):
        """Takes 2 ComplexNumber objects and performs division of complex numbers. Returns new ComplexNuber."""
        realNumerator = (self.real * other.real) + (self.imag * other.imag)
        imagNumerator = (self.imag * other.real) - (self.real * other.imag)
        denominator = ((other.real)**2) + ((other.imag)**2)
        return ComplexNumber(realNumerator/denominator, imagNumerator/denominator)

