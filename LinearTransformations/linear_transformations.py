# linear_transformations.py
"""Volume 1: Linear Transformations.
<Zach Joachim>
<Math 345>
<September 27, 2022>
"""

from random import random
import numpy as np
from matplotlib import pyplot as plt
import time

data = np.load("horse.npy")
horse_matrix = data[:2]                                                     # Taking just the first 2 columns of data

# Problem 1
def stretch(A, a, b):
    """Scale the points in A by a in the x direction and b in the
    y direction.

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): scaling factor in the x direction.
        b (float): scaling factor in the y direction.
    Return:
        ((2,n) ndarray): Transformed matrix
    """
    stretch_matrix_representation = np.array([[a, 0], [0, b]])
    stretch_horse = np.matmul(stretch_matrix_representation, horse_matrix)      # The stretch matrix must be first so the dimensions line up
    
    return stretch_horse
    

def shear(A, a, b):
    """Slant the points in A by a in the x direction and b in the
    y direction.

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): scaling factor in the x direction.
        b (float): scaling factor in the y direction.
    Return:
        ((2,n) ndarray): Transformed matrix
    """
    shear_matrix_representation = np.array([[1, a], [b, 1]])
    shear_horse = np.matmul(shear_matrix_representation, horse_matrix)

    return shear_horse

def reflect(A, a, b):
    """Reflect the points in A about the line that passes through the origin
    and the point (a,b).

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): x-coordinate of a point on the reflecting line.
        b (float): y-coordinate of the same point on the reflecting line.
    Return:
        ((2,n) ndarray): Transformed matrix
    """
    reflect_matrix_representation = (1/((a**2)+(b**2))) * np.array([[(a**2)-(b**2), 2*a*b], [2*a*b, (b**2)-(a**2)]])
    reflect_horse = np.matmul(reflect_matrix_representation, horse_matrix)

    return reflect_horse

def rotate(A, theta):
    """Rotate the points in A about the origin by theta radians.

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        theta (float): The rotation angle in radians.
    Return:
        ((2,n) ndarray): Transformed matrix
    """
    scale = np.array([[np.cos(theta),-1*np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    
    return np.matmul(scale,A)


# Problem 2
def solar_system(T, x_e, x_m, omega_e, omega_m):
    """Plot the trajectories of the earth and moon over the time interval [0,T]
    assuming the initial position of the earth is (x_e,0) and the initial
    position of the moon is (x_m,0).

    Parameters:
        T (float): The final time.
        x_e (float): The earth's initial x coordinate.
        x_m (float): The moon's initial x coordinate.
        omega_e (float): The earth's angular velocity.
        omega_m (float): The moon's angular velocity.
    """
    # Initial Conditions
    Pe0 = np.array([x_e,0])
    Pm0 = np.array([x_m,0])
    timer = np.linspace(0,T,500)
    
    #Rotations
    PeT = np.array([rotate(Pe0,t*omega_e) for t in timer]).T
    PmT = np.array([rotate(Pm0 - Pe0,t*omega_m) for t in timer] + PeT.T).T
    
    # Plotting Orbits
    plt.plot(PeT[0],PeT[1], 'b-', label="Earth")
    plt.plot(PmT[0],PmT[1], 'tab:orange',label="Moon")
    plt.legend(loc = "lower right")
    plt.show()

#solar_system((3*np.pi)/2,10,11,1,13)

def random_vector(n):
    """Generate a random vector of length n as a list."""
    return [random() for i in range(n)]

def random_matrix(n):
    """Generate a random nxn matrix as a list of lists."""
    return [[random() for j in range(n)] for i in range(n)]

def matrix_vector_product(A, x):
    """Compute the matrix-vector product Ax as a list."""
    m, n = len(A), len(x)
    return [sum([A[i][k] * x[k] for k in range(n)]) for i in range(m)]

def matrix_matrix_product(A, B):
    """Compute the matrix-matrix product AB as a list of lists."""
    m, n, p = len(A), len(B), len(B[0])
    return [[sum([A[i][k] * B[k][j] for k in range(n)])
                                    for j in range(p) ]
                                    for i in range(m) ]

# Problem 3
def prob3():
    """Use time.time(), timeit.timeit(), or %timeit to time
    matrix_vector_product() and matrix-matrix-mult() with increasingly large
    inputs. Generate the inputs A, x, and B with random_matrix() and
    random_vector() (so each input will be nxn or nx1).
    Only time the multiplication functions, not the generating functions.

    Report your findings in a single figure with two subplots: one with matrix-
    vector times, and one with matrix-matrix times. Choose a domain for n so
    that your figure accurately describes the growth, but avoid values of n
    that lead to execution times of more than 1 minute.
    """
    # Creating lists
    list_of_sizes = 2**np.arange(1,9)
    matrix_multiplication_times = []
    matrix_vector_times = []
    # Appending times for different sizes
    for size in list_of_sizes:
        matrix_1 = random_matrix(size)
        matrix_2 = random_matrix(size)
        start = time.perf_counter()
        matrix_matrix_product(matrix_1, matrix_2)
        matrix_multiplication_times.append(time.perf_counter() - start)

    for size in list_of_sizes:
        matrix_3 = random_matrix(size)
        vector = random_vector(size)
        start = time.perf_counter()
        matrix_vector_product(matrix_3, vector)
        matrix_vector_times.append(time.perf_counter() - start)

    # Plotting Matrix-Vector
    plot_1 = plt.subplot(121)
    plot_1.plot(list_of_sizes, matrix_vector_times, 'b.-', linewidth=2, markersize=15)
    plot_1.set_title("Matrix-Vector Multiplication")
    plot_1.set_xlabel("n")
    plot_1.set_ylabel("Seconds")
    # Plotting Matrix-Matrix
    plot_2 = plt.subplot(122)
    plot_2.plot(list_of_sizes, matrix_multiplication_times, 'g.-', linewidth=2, markersize=15)
    plot_2.set_title("Matrix-Matrix Multiplication")
    plot_2.set_xlabel("n")
    plt.tight_layout()
    plt.show()


# Problem 4
def prob4():
    """Time matrix_vector_product(), matrix_matrix_product(), and np.dot().

    Report your findings in a single figure with two subplots: one with all
    four sets of execution times on a regular linear scale, and one with all
    four sets of exections times on a log-log scale.
    """
    # Creating lists
    list_of_sizes = 2**np.arange(1,9)
    matrix_multiplication_times = []
    matrix_vector_times = []
    matrix_vector_dot_product_times = []
    matrix_matrix_dot_product_times = []
    # Appending times for different sizes
    for size in list_of_sizes:
        matrix_1 = random_matrix(size)
        matrix_2 = random_matrix(size)
        start = time.perf_counter()
        matrix_matrix_product(matrix_1, matrix_2)
        matrix_multiplication_times.append(time.perf_counter() - start)

    for size in list_of_sizes:
        matrix_3 = random_matrix(size)
        vector_1 = random_vector(size)
        start = time.perf_counter()
        matrix_vector_product(matrix_3, vector_1)
        matrix_vector_times.append(time.perf_counter() - start)

    for size in list_of_sizes:
        matrix_4 = random_matrix(size)
        vector_2 = random_vector(size)
        start = time.perf_counter()
        np.dot(matrix_4, vector_2)
        matrix_vector_dot_product_times.append(time.perf_counter() - start)

    for size in list_of_sizes:
        matrix_5 = random_matrix(size)
        matrix_6 = random_matrix(size)
        start = time.perf_counter()
        np.dot(matrix_5, matrix_6)
        matrix_matrix_dot_product_times.append(time.perf_counter() - start)

    # Plotting Matrix-Vector vs Matrix-Matrix in linear time
    plot1 = plt.subplot(121)
    plot1.plot(list_of_sizes, matrix_multiplication_times, 'b.-',  lw=2, ms=15, label="Matrix-Vector")
    plot1.plot(list_of_sizes, matrix_vector_times, 'g.-',  lw=2, ms=15, label="Matrix-Matrix")
    plot1.plot(list_of_sizes, matrix_vector_dot_product_times, 'r.-', lw=2, ms=15, label="Matrix Vector Dot Product")
    plot1.plot(list_of_sizes, matrix_matrix_dot_product_times,'m.-', lw=2, ms=15, label="Matrix Matrix Dot Product")
    plot1.set_title("Linear Time")
    plot1.legend(loc="upper left")

    # Plotting in log time
    plot2 = plt.subplot(122)
    plot2.set_title("Logarithmic Time")
    plot2.loglog(list_of_sizes, matrix_multiplication_times, 'b.-', base=2, lw=2)
    plot2.loglog(list_of_sizes, matrix_vector_times, 'g.-', base=2, lw=2)
    plot2.loglog(list_of_sizes, matrix_vector_dot_product_times, 'r.-', base=2, lw=2)
    plot2.loglog(list_of_sizes, matrix_matrix_dot_product_times, 'm.-', base=2, lw=2)

    plt.tight_layout()
    plt.show()
