# lstsq_eigs.py
"""Volume 1: Least Squares and Computing Eigenvalues.
<Zach Joachim>
<Math 345>
<October 25, 2022>
"""

# (Optional) Import functions from your QR Decomposition lab.
# import sys
# sys.path.insert(1, "../QR_Decomposition")
# from qr_decomposition import qr_gram_schmidt, qr_householder, hessenberg

import cmath
import numpy as np
from matplotlib import pyplot as plt
import scipy.linalg as la
from scipy.stats import linregress

data = np.load("housing.npy")

# Problem 1
def least_squares(A, b):
    """Calculate the least squares solutions to Ax = b by using the QR
    decomposition.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n <= m.
        b ((m, ) ndarray): A vector of length m.

    Returns:
        x ((n, ) ndarray): The solution to the normal equations.
    """
    # Getting reduced QR Decomposition
    Q, R = la.qr(A, mode="economic")
    y = np.matmul(np.transpose(Q), b)
    n = len(y)
    x_hat = np.zeros(n).astype("float64")
    x_hat[n-1] = (1/R[n-1,n-1])*y[n-1]
    # Solving by back-substitution
    for k in range(n-2, -1, -1):
        current_sum = 0
        for j in range(k+1, n):
            current_sum += R[k,j] * x_hat[j]
        x_hat[k] = (1/R[k,k])*(y[k] - current_sum)

    return x_hat

# Problem 2
def line_fit():
    """Find the least squares line that relates the year to the housing price
    index for the data in housing.npy. Plot both the data points and the least
    squares line.
    """
    # Getting data
    year = data[:,0]
    index = data[:,1]
    ones_vector = np.ones(len(index))
    A = np.column_stack((year, ones_vector))
    b = index
    # Getting least squares solution
    least_squares_solution = least_squares(A, b)
    y_vals = []
    for i in year:
        # Append to y_vals
        y_vals.append(i*least_squares_solution[0] + least_squares_solution[1])

    # Plotting function
    plt.scatter(year, index, c="blue")
    plt.plot(year, y_vals, "r")
    plt.show()

# Problem 3
def polynomial_fit():
    """Find the least squares polynomials of degree 3, 6, 9, and 12 that relate
    the year to the housing price index for the data in housing.npy. Plot both
    the data points and the least squares polynomials in individual subplots.
    """
    # degrees = [3,6,9,12]
    year = data[:,0]
    index = data[:,1]
    b = index
    f3_yvals = []
    f6_yvals = []
    f9_yvals = []
    f12_yvals = []
    # Calculate Vadermonde Matrix of degree 3
    vander_3 = np.vander(year, 3)
    least_sqaures_solution_1 = least_squares(vander_3, b)
    f_3 = np.poly1d(least_sqaures_solution_1)
    for i in year:
        f3_yvals.append(f_3(i))
    # Calculate Vadermonde Matrix of degree 6
    vander_6 = np.vander(year, 6)
    least_sqaures_solution_2 = least_squares(vander_6, b)
    f_6 = np.poly1d(least_sqaures_solution_2)
    for i in year:
        f6_yvals.append(f_6(i))
    # Calculate Vadermonde Matrix of degree 9
    vander_9 = np.vander(year, 9)
    least_sqaures_solution_3 = least_squares(vander_9, b)
    f_9 = np.poly1d(least_sqaures_solution_3)
    for i in year:
        f9_yvals.append(f_9(i))
    # Calculate Vadermonde Matrix of degree 12
    vander_12 = np.vander(year, 12)
    least_sqaures_solution_4 = least_squares(vander_12, b)
    f_12 = np.poly1d(least_sqaures_solution_4)
    for i in year:
        f12_yvals.append(f_12(i))

    plt.subplot(221) 
    plt.title("Degree 3")
    plt.scatter(year, index, c="blue")
    plt.plot(year, f3_yvals, 'r-')
    plt.subplot(222)
    plt.title("Degree 6")
    plt.scatter(year, index, c="blue")
    plt.plot(year, f6_yvals, 'g-')
    plt.subplot(223)
    plt.title("Degree 9")
    plt.scatter(year, index, c="blue")
    plt.plot(year, f9_yvals, 'r-')
    plt.subplot(224)
    plt.title("Degree 12")
    plt.scatter(year, index, c="blue")
    plt.plot(year, f12_yvals, 'g-')
    plt.tight_layout()
    plt.show()


def plot_ellipse(a, b, c, d, e):
    """Plot an ellipse of the form ax^2 + bx + cxy + dy + ey^2 = 1."""
    theta = np.linspace(0, 2*np.pi, 200)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    A = a*(cos_t**2) + c*cos_t*sin_t + e*(sin_t**2)
    B = b*cos_t + d*sin_t
    r = (-B + np.sqrt(B**2 + 4*A)) / (2*A)

    plt.plot(r*cos_t, r*sin_t)
    plt.gca().set_aspect("equal", "datalim")

# Problem 4
def ellipse_fit():
    """Calculate the parameters for the ellipse that best fits the data in
    ellipse.npy. Plot the original data points and the ellipse together, using
    plot_ellipse() to plot the ellipse.
    """
    # Loading data
    x, y = np.load("ellipse.npy").T
    # Forming my A matrix and b vector
    A = np.column_stack((x**2, x, x*y, y, y**2))
    b = np.ones(A.shape[0])
    # Getting the least squares solution
    a, b, c, d, e = la.lstsq(A, b)[0]
    plot_ellipse(a, b, c, d, e)
    # Setting title, axes, and plotting original points
    plt.title("Ellipse Fit Function")
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.scatter(x,y, c="orange")
    plt.show()

# Problem 5
def power_method(A, N=20, tol=1e-12):
    """Compute the dominant eigenvalue of A and a corresponding eigenvector
    via the power method.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        N (int): The maximum number of iterations.
        tol (float): The stopping tolerance.

    Returns:
        (float): The dominant eigenvalue of A.
        ((n,) ndarray): An eigenvector corresponding to the dominant
            eigenvalue of A.
    """
    # Getting shape of A
    m, n = len(A), len(A[0])
    # Making random matrix
    x = np.random.random(n)
    # Normalize
    x = x / la.norm(x)
    for k in range(N-1):
        x = A @ x
        x = x / la.norm(x)
        
    return np.transpose(x) @ A @ x, x

# Problem 6
def qr_algorithm(A, N=50, tol=1e-12):
    """Compute the eigenvalues of A via the QR algorithm.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        N (int): The number of iterations to run the QR algorithm.
        tol (float): The threshold value for determining if a diagonal S_i
            block is 1x1 or 2x2.

    Returns:
        ((n,) ndarray): The eigenvalues of A.
    """
    # Getting shape of A
    m, n = len(A), len(A[0])
    # Turning into hessenberg form
    S = la.hessenberg(A)
    for k in range(N-1):
        # Getting QR Decomposition
        Q, R = la.qr(S)
        S = np.matmul(R,Q)
    # Initialiing empty list
    eigs = []
    i = 0
    while i < n:
        # If Si is 1x1
        if (i == n-1) or (np.absolute(S[i+1,i]) < tol):
            eigs.append(S[i,i])
        # If Si is 2x2
        else:
            # Computing eigenvalues using quadratic formula
            lambda_1 = ((S[i,i]+S[i+1,i+1]) + cmath.sqrt((S[i,i]+S[i+1,i+1])**2 - 4*(S[i,i]*S[i+1,i+1] - S[i,i+1]*S[i+1,i]))) / 2
            lambda_2 = ((S[i,i]+S[i+1,i+1]) - cmath.sqrt((S[i,i]+S[i+1,i+1])**2 - 4*(S[i,i]*S[i+1,i+1] - S[i,i+1]*S[i+1,i]))) / 2
            # Appending to eigenvalue list
            eigs.append(lambda_1)
            eigs.append(lambda_2)
            i = i + 1
        i = i + 1
    return eigs
