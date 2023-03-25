# qr_decomposition.py
"""Volume 1: The QR Decomposition.
<Name>
<Math 345>
<October 18, 2022>
"""

import numpy as np
import scipy.linalg as la


# Problem 1
def qr_gram_schmidt(A):
    """Compute the reduced QR decomposition of A via Modified Gram-Schmidt.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n.

    Returns:
        Q ((m,n) ndarray): An orthonormal matrix.
        R ((n,n) ndarray): An upper triangular matrix.
    """
    # Getting shape of A
    m, n = len(A), len(A[0])
    # Initializing Q and R
    Q = np.copy(A).astype("float64")
    R = np.zeros((n,n)).astype("float64")

    # Following algorithm for QR Decomposition
    for i in range(n):
        R[i,i] = la.norm(Q[:,i])
        Q[:,i] = Q[:,i] / R[i,i]
        for j in range(i+1, n):
            R[i,j] = np.matmul(np.transpose(Q[:,j]), Q[:,i])
            Q[:,j] = Q[:,j] - (R[i,j]*Q[:,i])

    return Q, R         # Returning Q and R

# Problem 2
def abs_det(A):
    """Use the QR decomposition to efficiently compute the absolute value of
    the determinant of A.

    Parameters:
        A ((n,n) ndarray): A square matrix.

    Returns:
        (float) the absolute value of the determinant of A.
    """
    # Getting Q and R
    Q, R = qr_gram_schmidt(A)
    # Returning determinant
    return np.abs(np.product(np.diag(R)))


# Problem 3
def solve(A, b):
    """Use the QR decomposition to efficiently solve the system Ax = b.

    Parameters:
        A ((n,n) ndarray): An invertible matrix.
        b ((n, ) ndarray): A vector of length n.

    Returns:
        x ((n, ) ndarray): The solution to the system Ax = b.
    """
    # Get Q and R
    Q, R = qr_gram_schmidt(A)
    # Set y = Q^T*b
    y = np.matmul(np.transpose(Q), b)
    n = len(y)
    # Initialize solution to 0 vector
    x = np.zeros(n).astype("float64")
    # Set last element
    x[n-1] = (1/R[n-1,n-1])*y[n-1]
    # Fill in from the back
    for k in range(n-2, -1, -1):
        # Initialize a sum of R[k,j] * x[j] to 0 for every k
        current_sum = 0
        # Get new sum
        for j in range(k+1, n):
            current_sum += R[k,j] * x[j]
        # Set kth element of the solution x
        x[k] = (1/R[k,k])*(y[k] - current_sum)

    return x

# Problem 4
def qr_householder(A):
    """Compute the full QR decomposition of A via Householder reflections.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n.

    Returns:
        Q ((m,m) ndarray): An orthonormal matrix.
        R ((m,n) ndarray): An upper triangular matrix.
    """
    # Get shape of A
    m, n = len(A), len(A[0])
    # Initialize Q and R
    R = np.copy(A).astype("float64")
    Q = np.eye(m).astype("float64")
    # Following algorithm
    for k in range(n-3):
        u = np.copy(R[k:,k])
        # More optimal sign function
        sign = lambda x: 1 if x >= 0 else -1
        u[0] = u[0] + sign(u[0])*la.norm(u)
        # Normalize u
        u = u / la.norm(u)
        R[k:,k:] = R[k:,k:] - np.outer(2*u, np.matmul(np.transpose(u), R[k:,k:]))
        Q[k:,:] = Q[k:,:] - np.outer(2*u, np.matmul(np.transpose(u), Q[k:,:])) 

    return np.transpose(Q), R

# Problem 5
def hessenberg(A):
    """Compute the Hessenberg form H of A, along with the orthonormal matrix Q
    such that A = QHQ^T.

    Parameters:
        A ((n,n) ndarray): An invertible matrix.

    Returns:
        H ((n,n) ndarray): The upper Hessenberg form of A.
        Q ((n,n) ndarray): An orthonormal matrix.
    """
    # Get shape of A
    m, n = len(A), len(A[0])
    # Initialize H and Q
    H = np.copy(A)
    Q = np.eye(m)
    # Start algorithm
    for k in range(n-1):
        u = np.copy(H[k+1:,k])
        # More optimal sign function
        sign = lambda x: 1 if x >= 0 else -1
        u[0] = u[0] + sign(u[0])*la.norm(u)
        # Normalize u
        u = u / la.norm(u)
        H[k+1:,k:] = H[k+1:,k:] - np.outer(2*u, np.matmul(np.transpose(u), H[k+1:, k:]))
        H[:,k+1:] = H[:,k+1:] - np.outer(2*np.matmul(H[:,k+1:], u), np.transpose(u))
        Q[k+1:,:] = Q[k+1:,:] - np.outer(2*u, np.matmul(np.transpose(u), Q[k+1:,:]))

    return H, np.transpose(Q)
