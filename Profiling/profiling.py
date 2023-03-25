# profiling.py
"""Python Essentials: Profiling.
<Zach Joachim>
<Math 347>
<March 20, 2023>
"""

# Note: for problems 1-4, you need only implement the second function listed.
# For example, you need to write max_path_fast(), but keep max_path() unchanged
# so you can do a before-and-after comparison.

import numpy as np
import time
from matplotlib import pyplot as plt
from numba import jit

# Problem 1
def max_path(filename="triangle.txt"):
    """Find the maximum vertical path in a triangle of values."""
    with open(filename, 'r') as infile:
        data = [[int(n) for n in line.split()]
                        for line in infile.readlines()]
    def path_sum(r, c, total):
        """Recursively compute the max sum of the path starting in row r
        and column c, given the current total.
        """
        total += data[r][c]
        if r == len(data) - 1:          # Base case.
            return total
        else:                           # Recursive case.
            return max(path_sum(r+1, c,   total),   # Next row, same column
                       path_sum(r+1, c+1, total))   # Next row, next column

    return path_sum(0, 0, 0)            # Start the recursion from the top.

def max_path_fast(filename="triangle_large.txt"):
    """Find the maximum vertical path in a triangle of values."""
    # Opening the file and creating list of data
    with open(filename, 'r') as infile:
        data = [[int(n) for n in line.split()]
                        for line in infile.readlines()]

    # Setting r to be the size of the data minus 1   
    r = len(data) - 1
    while r > 0:
        for c in range(r):
            data[r-1][c] = max(data[r-1][c] + data[r][c], data[r-1][c] + data[r][c+1])  
        r -= 1
    
    return data[0][0]

# print(max_path_fast())

# Problem 2
def primes(N):
    """Compute the first N primes."""
    primes_list = []
    current = 2
    while len(primes_list) < N:
        isprime = True
        for i in range(2, current):     # Check for nontrivial divisors.
            if current % i == 0:
                isprime = False
        if isprime:
            primes_list.append(current)
        current += 1
    return primes_list

def primes_fast(N):
    """Compute the first N primes."""
    primes_list = [2]
    current = 3
    length = 1
    # Iterate as long as the length is less than N
    while length < N:
        isprime = True
        x = int(current**.5)
        # Iterate through the primes
        for i in primes_list:
            if i > x:
                break
            # If divisible by another number, it's not prime
            if current % i == 0:
                isprime = False
                break
        # Append all primes to list
        if isprime:
            primes_list.append(current)
            length += 1
        current += 2

    return primes_list

# primes_fast(10000)

# Problem 3
def nearest_column(A, x):
    """Find the index of the column of A that is closest to x.

    Parameters:
        A ((m,n) ndarray)
        x ((m, ) ndarray)

    Returns:
        (int): The index of the column of A that is closest in norm to x.
    """
    distances = []
    for j in range(A.shape[1]):
        distances.append(np.linalg.norm(A[:,j] - x))
    return np.argmin(distances)

def nearest_column_fast(A, x):
    """Find the index of the column of A that is closest in norm to x.
    Refrain from using any loops or list comprehensions.

    Parameters:
        A ((m,n) ndarray)
        x ((m, ) ndarray)

    Returns:
        (int): The index of the column of A that is closest in norm to x.
    """
    return np.argmin(np.linalg.norm(A - x[:,np.newaxis],axis = 0))

"""
A = np.array([[1,2,3],[4,5,6],[7,8,9]])
x = np.array([8,4,7])
print(nearest_column(A, x))
print(nearest_column_fast(A, x))
"""

# Problem 4
def name_scores(filename="names.txt"):
    """Find the total of the name scores in the given file."""
    with open(filename, 'r') as infile:
        names = sorted(infile.read().replace('"', '').split(','))
    total = 0
    for i in range(len(names)):
        name_value = 0
        for j in range(len(names[i])):
            alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            for k in range(len(alphabet)):
                if names[i][j] == alphabet[k]:
                    letter_value = k + 1
            name_value += letter_value
        total += (names.index(names[i]) + 1) * name_value
    return total

def name_scores_fast(filename='names.txt'):
    """Find the total of the name scores in the given file."""
    with open(filename, 'r') as infile:
        names = sorted(infile.read().replace('"', '').split(','))
    
    # Initialize data
    total = 0
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    numbers = [i for i in range(1,27)]
    name_index = 1
    
    # Zip the alphabete to the indicies
    dictionary = dict(zip(alphabet, numbers))
    
    #Eliminate one of the for loops
    for name in names:
        name_value = 0
        for letter in name:
            letter_value = dictionary[letter]
            name_value += letter_value
        total += name_index * name_value
        name_index += 1
    return total
    
# print(name_scores())
# print(name_scores_fast())

# Problem 5
def fibonacci():
    """Yield the terms of the Fibonacci sequence with F_1 = F_2 = 1."""
    Fn_minus1 = 1
    Fn = 1
    yield Fn_minus1
    yield Fn
    
    #Recursively yield the fibonacci numbers
    while True:
        Fib = Fn_minus1 + Fn
        yield Fib
        Fn_minus1 = Fn
        Fn = Fib

def fibonacci_digits(N=1000):
    """Return the index of the first term in the Fibonacci sequence with
    N digits.

    Returns:
        (int): The index.
    """
    x = 10**(N-1)
    
    # Get indices and fib numbers
    for i, fib in enumerate(fibonacci()):
        # Check if not enough digits yet
        if fib < x:
            pass
        else:
            return i+1
        
# print(fibonacci_digits(3))

# Problem 6
def prime_sieve(N):
    """Yield all primes that are less than N."""

    integers = np.arange(2, N+1)
    # While the array is not empty
    while True:
        first = integers[0]
        # Create a mask where it keeps only numbers not divisble by the first
        mask = tuple([(integers % first) != 0])
        integers = integers[mask]
        yield first
        if len(integers) == 0:
            break


# print(prime_sieve(100000))

# Problem 7
def matrix_power(A, n):
    """Compute A^n, the n-th power of the matrix A."""
    product = A.copy()
    temporary_array = np.empty_like(A[0])
    m = A.shape[0]
    for power in range(1, n):
        for i in range(m):
            for j in range(m):
                total = 0
                for k in range(m):
                    total += product[i,k] * A[k,j]
                temporary_array[j] = total
            product[i] = temporary_array
    return product

@jit
def matrix_power_numba(A, n):
    """Compute A^n, the n-th power of the matrix A, with Numba optimization."""
    product = A.copy()
    temporary_array = np.empty_like(A[0])
    m = A.shape[0]
    for power in range(1, n):
        for i in range(m):
            for j in range(m):
                total = 0
                for k in range(m):
                    total += product[i,k] * A[k,j]
                temporary_array[j] = total
            product[i] = temporary_array

    return product

def prob7(n=10):
    """Time matrix_power(), matrix_power_numba(), and np.linalg.matrix_power()
    on square matrices of increasing size. Plot the times versus the size.
    """
    mvals = [2**i for i in range(2,8)]
    regular, numba, numpy = [],[],[]
    call = matrix_power_numba(np.array([[1,1],[2,3]]), 2)
    
    for m in mvals:
        A = np.random.random((m,m))
        
        # Time regular
        start = time.time()
        comp1 = matrix_power(A,n)
        end1 = time.time() - start
        regular.append(end1)
        
        # Time numba
        start = time.time()
        comp1 = matrix_power_numba(A,n)
        end1 = time.time() - start
        numba.append(end1)
        
        # Time numpy
        start = time.time()
        comp1 = np.linalg.matrix_power(A,n)
        end1 = time.time() - start
        numpy.append(end1)
    
    # Plotting everything
    plt.loglog(mvals, regular, label='Regular')
    plt.loglog(mvals,numba,label='numba')
    plt.loglog(mvals,numpy,label='numpy')
    plt.legend(loc='upper left')
    plt.title("Matrix Power Methods")
    plt.tight_layout()
    plt.show()
    
    return

prob7()
