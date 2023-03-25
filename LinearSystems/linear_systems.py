# linear_systems.py
"""Volume 1: Linear Systems.
<Zach Joachim>
<Math 345>
<October 11, 2022>
"""

import numpy as np
import time as time
import scipy.linalg as la
from scipy import sparse
from matplotlib import pyplot as plt
from scipy.sparse import linalg as spla

# Problem 1
def ref(A):
    """Reduce the square matrix A to REF. You may assume that A is invertible
    and that a 0 will never appear on the main diagonal. Avoid operating on
    entries that you know will be 0 before and after a row operation.

    Parameters:
        A ((n,n) ndarray): The square invertible matrix to be reduced.

    Returns:
        ((n,n) ndarray): The REF of A.
    """
    for j in range(0, len(A[0]) - 1):
        for i in range(j+1, len(A[0])):
            A[i,j:] = A[i,j:] - ((A[i,j] / A[j,j]) * A[j,j:])

    return A

# Problem 2
def lu(A):
    """Compute the LU decomposition of the square matrix A. You may
    assume that the decomposition exists and requires no row swaps.

    Parameters:
        A ((n,n) ndarray): The matrix to decompose.

    Returns:
        L ((n,n) ndarray): The lower-triangular part of the decomposition.
        U ((n,n) ndarray): The upper-triangular part of the decomposition.
    """
    m = len(A)                                   # number of rows
    n = len(A[0])                                # number of columns
    U = np.copy(A)
    L = np.eye(m)                                # initialize to an identity matrix

    for j in range(n-1):
        for i in range(j+1, m):
            L[i,j] = U[i,j] / U[j,j]
            U[i,j:] = U[i,j:] - (L[i,j]*U[j,j:])

    return L, U

# Problem 3
def solve(A, b):
    """Use the LU decomposition and back substitution to solve the linear
    system Ax = b. You may again assume that no row swaps are required.

    Parameters:
        A ((n,n) ndarray)
        b ((n,) ndarray)

    Returns:
        x ((m,) ndarray): The solution to the linear system.
    """
    A = np.copy(A)
    b = np.copy(b)

    L = np.eye(A.shape[0])
    U = np.zeros(A.shape)

    for j in range(A.shape[1]):
        for i in range(j+1, A.shape[0]):
            mult = A[i,j] / A[j,j]
            A[i,:] -= mult * A[j,:]
            L[i,j] = mult
    
    U = A
    y = la.solve_triangular(L, b, lower=True)
    x = la.solve_triangular(U, y, lower=True)

    return x

# Problem 4
def prob4():
    """Time different scipy.linalg functions for solving square linear systems.

    For various values of n, generate a random nxn matrix A and a random
    n-vector b using np.random.random(). Time how long it takes to solve the
    system Ax = b with each of the following approaches:

        1. Invert A with la.inv() and left-multiply the inverse to b.
        2. Use la.solve().
        3. Use la.lu_factor() and la.lu_solve() to solve the system with the
            LU decomposition.
        4. Use la.lu_factor() and la.lu_solve(), but only time la.lu_solve()
            (not the time it takes to do the factorization).

    Plot the system size n versus the execution times. Use log scales if
    needed.
    """
    domain = 2**np.arange(1,13)                       # logarithmic domain
    inverse_A_times = []
    solve_times = []
    lu_times = []
    lu_solve_times = []                               # 4 empty lists for the approaches

    for n in domain:
        A = np.random.random((n, n))
        b = np.random.random(n)
        start_time = time.time()
        solution_1 = np.matmul(la.inv(A), b)          # matrix multiplication
        inverse_A_times.append(time.time() - start_time)

        start_time_2 = time.time()
        solution_2 = la.solve(A, b)                   # linalg solve function
        solve_times.append(time.time() - start_time_2)

        start_time_3 = time.time()
        L, U = la.lu_factor(A)
        solution_3 = la.lu_solve((L, U), b)           # lu factoring and then solving
        lu_times.append(time.time() - start_time_3)

        L, U = la.lu_factor(A)
        start_time_4 = time.time()
        solution_4 = la.lu_solve((L, U), b)           # just lu_solve
        lu_solve_times.append(time.time() - start_time_4)
    
    plt.loglog(domain, inverse_A_times, base=2)
    plt.loglog(domain, solve_times, base=2)
    plt.loglog(domain, lu_times, base=2)
    plt.loglog(domain, lu_solve_times, base=2)
    plt.xlabel("n")
    plt.ylabel("Time for execution")
    plt.legend(("inv()", "solve()", "lu_factor", "lu_solve"))
    plt.tight_layout()
    plt.show()

# Problem 5
def prob5(n):
    """Let I be the n × n identity matrix, and define
                    [B I        ]        [-4  1            ]
                    [I B I      ]        [ 1 -4  1         ]
                A = [  I . .    ]    B = [    1  .  .      ],
                    [      . . I]        [          .  .  1]
                    [        I B]        [             1 -4]
    where A is (n**2,n**2) and each block B is (n,n).
    Construct and returns A as a sparse matrix.

    Parameters:
        n (int): Dimensions of the sparse matrix B.

    Returns:
        A ((n**2,n**2) SciPy sparse matrix)
    """
    B = sparse.diags([1, -4, 1], [-1, 0, 1], shape=(n,n))   # setting diagonals of B
    A = sparse.block_diag([B] * n)  
    A.setdiag(1, -n)                                        # setting identity matrix
    A.setdiag(1, n)
    
    return A

# Problem 6
def prob6():
    """Time regular and sparse linear system solvers.

    For various values of n, generate the (n**2,n**2) matrix A described of
    prob5() and vector b of length n**2. Time how long it takes to solve the
    system Ax = b with each of the following approaches:

        1. Convert A to CSR format and use scipy.sparse.linalg.spsolve()
        2. Convert A to a NumPy array and use scipy.linalg.solve().

    In each experiment, only time how long it takes to solve the system (not
    how long it takes to convert A to the appropriate format). Plot the system
    size n**2 versus the execution times. As always, use log scales where
    appropriate and use a legend to label each line.
    """

    domain = 2**np.arange(1,7)                      # setting domain
    CSR_times = []                              
    numpy_times = []                                # 2 empty lists for both approaches

    for n in domain:
        A = prob5(n)
        b = np.random.random(n**2)

        Acsr = A.tocsr()                            # converting to csr matrix
        start_1 = time.time()
        solution_1 = spla.spsolve(Acsr, b)          
        CSR_times.append(time.time() - start_1)

        A_numpy = A.toarray()                       # converting to numpy matrix
        start_2 = time.time()
        solution_2 = la.solve(A_numpy, b)
        numpy_times.append(time.time() - start_2)

    plt.loglog(domain, CSR_times, base=2)
    plt.loglog(domain, numpy_times, base=2)
    plt.xlabel("n")
    plt.ylabel("Times")
    plt.legend(("CSR Times", "Numpy Times"))
    plt.tight_layout()
    plt.show()