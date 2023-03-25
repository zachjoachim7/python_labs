# exceptions_fileIO.py
"""Python Essentials: Exceptions and File Input/Output.
<Zach Joachim>
<Math 321>
<September 20, 2022>
"""

from random import choice


# Problem 1
def arithmagic():
    """
    Takes in user input to perform a magic trick and prints the result.
    Verifies the user's input at each step and raises a
    ValueError with an informative error message if any of the following occur:

    The first number step_1 is not a 3-digit number.
    The first number's first and last digits differ by less than $2$.
    The second number step_2 is not the reverse of the first number.
    The third number step_3 is not the positive difference of the first two numbers.
    The fourth number step_4 is not the reverse of the third number.
    """

    step_1 = input("Enter a 3-digit number where the first and last digits differ by 2 or more: ")
    print(type(step_1))
    if (len(step_1) != 3):
        raise ValueError("Not a 3 digit number")
    if (abs(int(step_1[2]) - int(step_1[0]))) < 2:
        raise ValueError("The first and last digit must differ by more than 2")
    step_2 = input("Enter the reverse of the first number, obtained by reading it backwards: ")
    if ((step_2[0] != step_1[2]) or (step_2[1] != step_1[1]) or (step_2[2] != step_1[0])):
        raise ValueError("Not the reverse of the number")
    step_3 = input("Enter the positive difference of these numbers: ")
    if (int(step_3) != abs(int(step_1) - int(step_2))):
        raise ValueError("Not the positive difference")
    step_4 = input("Enter the reverse of the previous result: ")
    if ((step_4[0] != step_3[2]) or (step_4[1] != step_3[1]) or (step_4[2] != step_3[0])):
        raise ValueError("Not the reverse of the positive difference")
    print(str(step_3), "+", str(step_4), "= 1089 (ta-da!)")




# Problem 2
def random_walk(max_iters=1e12):
    """
    If the user raises a KeyboardInterrupt by pressing ctrl+c while the
    program is running, the function should catch the exception and
    print "Process interrupted at iteration $i$".
    If no KeyboardInterrupt is raised, print "Process completed".

    Return walk.
    """
    try:
        walk = 0
        directions = [1, -1]
        for i in range(int(max_iters)):
            walk += choice(directions)
    except KeyboardInterrupt:
        print("Process interupted at iteration " + str(i))
    else:
        print("Process completed")
    finally:
        return walk


# Problems 3 and 4: Write a 'ContentFilter' class.
class ContentFilter(object):
    """Class for reading in file

    Attributes:
        filename (str): The name of the file
        contents (str): the contents of the file

    """
    # Problem 3
    def __init__(self, filename):
        """ Read from the specified file. If the filename is invalid, prompt
        the user until a valid filename is given.
        """
        valid = False                                     # Boolean for validity of filename
        while not valid:
            try:
                valid = True                                            
                with open(str(filename), 'r') as file:
                    file_string = file.read()   
            except FileNotFoundError:                      # exception case 1
                filename = input("Please enter a valid file name: ")
                valid = False                              # must keep boolean false in exception cases
            except TypeError:                              # exception case 2
                filename = input("Please enter a valid file name: ")
                valid = False
            except OSError:                                # exception case 3
                filename = input("Please enter a valid file name: ")
                valid = False
            else:
                valid = True

        self.filename = filename
        self.contents = file_string
        

 # Problem 4 ---------------------------------------------------------------
    def check_mode(self, mode):
        """ Raise a ValueError if the mode is invalid. """
        if (mode != "w" and mode != "a" and mode != "x"):
            raise ValueError("The file access must be 'w', 'a', or 'x'")

    def uniform(self, outfile, mode='w', case='upper'):
        """ Write the data to the outfile with uniform case. Include an additional
        keyword argument case that defaults to "upper". If case="upper", write
        the data in upper case. If case="lower", write the data in lower case.
        If case is not one of these two values, raise a ValueError. """
        if (case == 'upper'): 
            with open(str(outfile), mode) as newFile:               # Making alias for output file
                for line in self.contents:                          # Iterate over every line in the input file contents
                    newFile.write(line.upper())                     # Write to new file
        elif (case == 'lower'):
            with open(str(outfile), mode) as newFile:               # Same process, just lowercase
                for line in self.contents:
                    newFile.write(line.lower())                                                      
        else:                                                       # If the case is mispelled or does not exist, raise an error
            raise ValueError("The case must be either 'upper' or 'lower'")


    def reverse(self, outfile, mode='w', unit='word'):
        """ Write the data to the outfile in reverse order. Include an additional
        keyword argument unit that defaults to "line". If unit="word", reverse
        the ordering of the words in each line, but write the lines in the same
        order as the original file. If units="line", reverse the ordering of the
        lines, but do not change the ordering of the words on each individual
        line. If unit is not one of these two values, raise a ValueError. """
        if (unit == 'word'):
            with open(str(outfile), mode) as newFile:
                data = self.contents.split("\n")                    # split by new lines
                words = [line.split(" ") for line in data]          # split by spaces
                reversewordlist = [word[::-1] for word in words]    # reverse the strings
                joinlist = [' '.join(revword) for revword in reversewordlist]   # rejoin by space first
                final = '\n'.join(joinlist)                         # rejoing by new lines
                newFile.writelines(final.strip())
        elif (unit == 'line'):
            with open(str(outfile), mode) as newFile:
                splitlines = self.contents.split("\n")              # splitting by new lines
                reversedlines = splitlines[::-1]                    # reverse order of lines
                finalstring = '\n'.join(reversedlines)              # rejoin them by new lines
                newFile.writelines(finalstring)
        else:
            raise ValueError("You can only reverse words and lines, check spelling")

    def transpose(self, outfile, mode='w'):
        """ Write a transposed version of the data to the outfile. That is, write
        the first word of each line of the data to the first line of the new file,
        the second word of each line of the data to the second line of the new
        file, and so on. Viewed as a matrix of words, the rows of the input file
        then become the columns of the output file, and viceversa. You may assume
        that there are an equal number of words on each line of the input file. """
        self.check_mode(mode)
        with open(outfile, mode) as outfile:
            lines = self.contents.split("\n")               # split string first
            words = [line.split(' ') for line in lines]
            newTrans = ''
            for i in range(len(words[0])):
                for j in range(len(lines) - 1):
                    if j == len(words) - 2:
                        newTrans += words[j][i]
                    else:
                        newTrans += words[j][i] + " "
                newTrans += "\n"

            outfile.write(newTrans.strip())

    def __str__(self):
        """ Printing a ContentFilter object yields the following output:

        Source file:            <filename>
        Total characters:       <The total number of characters in file>
        Alphabetic characters:  <The number of letters>
        Numerical characters:   <The number of digits>
        Whitespace characters:  <The number of spaces, tabs, and newlines>
        Number of lines:        <The number of lines>
        """
        objectString = "Source file:\t\t\t" + self.filename + "\n"
        characterString = ''.join(self.contents[::-1])      # creating string from self.contents
        objectString += "Total characters:\t\t" + str(len(characterString)) + "\n"
        alphacount = 0                          # count for alpha
        for character in characterString:
            if character.isalpha():
                alphacount += 1
        objectString += "Alphabetical characters:\t" + str(alphacount) + "\n"
        numericalcount = 0                     # count for numbers
        for number in characterString:
            if number.isdigit():
                numericalcount += 1
        objectString += "Numerical characters:\t\t" + str(numericalcount) + "\n"
        whitespacecount = 0                    # count of whitespace
        for char in characterString:
            if char.isspace():
                whitespacecount += 1
        objectString += "Whitespace characters:\t\t" + str(whitespacecount) + "\n"
        objectString += "Number of lines:\t\t" + str(characterString.count("\n")) + "\n"

        return objectString
