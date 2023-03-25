# image_segmentation.py
"""Volume 1: Image Segmentation.
<Zach Joachim>
<Math 345>
<November 1, 2022>
"""

import numpy as np
import scipy.linalg as la
from scipy.sparse import lil_matrix, csc_matrix, csgraph, diags, linalg
from imageio.v2 import imread
from matplotlib import pyplot as plt


# Problem 1
def laplacian(A):
    """Compute the Laplacian matrix of the graph G that has adjacency matrix A.

    Parameters:
        A ((N,N) ndarray): The adjacency matrix of an undirected graph G.

    Returns:
        L ((N,N) ndarray): The Laplacian matrix of G.
    """
    # Initialize a degree matrix D and Laplacian matrix L
    D = np.zeros_like(A)
    L = np.zeros_like(A)
    # Compute the digsonal entries of D by summing the rows of A
    for i in range(len(A)):
        D[i,i] = np.sum(A[i])
    
    # Compute L by subtracting A from D
    L = D - A

    return L

# Problem 2
def connectivity(A, tol=1e-8):
    """Compute the number of connected components in the graph G and its
    algebraic connectivity, given the adjacency matrix A of G.

    Parameters:
        A ((N,N) ndarray): The adjacency matrix of an undirected graph G.
        tol (float): Eigenvalues that are less than this tolerance are
            considered zero.

    Returns:
        (int): The number of connected components in G.
        (float): the algebraic connectivity of G.
    """
    # Get the Laplacian matrix of A
    L = laplacian(A)
    # Compute its eigenvalues and eigenvectors
    eigs, vecs = la.eig(L)
    algebraic_connectivity = 0
    # Counting the number of eigenvalues that are 0, which will give us the number of connected graphs
    for i in range(len(eigs)):
        if i == 0:
            algebraic_connectivity += 1

    if eigs[1] == 0:
        return algebraic_connectivity, 0
    else:
        return algebraic_connectivity, eigs[1]

# Helper function for problem 4.
def get_neighbors(index, radius, height, width):
    """Calculate the flattened indices of the pixels that are within the given
    distance of a central pixel, and their distances from the central pixel.

    Parameters:
        index (int): The index of a central pixel in a flattened image array
            with original shape (radius, height).
        radius (float): Radius of the neighborhood around the central pixel.
        height (int): The height of the original image in pixels.
        width (int): The width of the original image in pixels.

    Returns:
        (1-D ndarray): the indices of the pixels that are within the specified
            radius of the central pixel, with respect to the flattened image.
        (1-D ndarray): the euclidean distances from the neighborhood pixels to
            the central pixel.
    """
    # Calculate the original 2-D coordinates of the central pixel.
    row, col = index // width, index % width

    # Get a grid of possible candidates that are close to the central pixel.
    r = int(radius)
    x = np.arange(max(col - r, 0), min(col + r + 1, width))
    y = np.arange(max(row - r, 0), min(row + r + 1, height))
    X, Y = np.meshgrid(x, y)

    # Determine which candidates are within the given radius of the pixel.
    R = np.sqrt(((X - col)**2 + (Y - row)**2))
    mask = R < radius
    return (X[mask] + Y[mask]*width).astype(np.int), R[mask]


# Problems 3-6
class ImageSegmenter:
    """Class for storing and segmenting images."""

    # Problem 3
    def __init__(self, filename):
        """Read the image file. Store its brightness values as a flat array."""
        # Read in the image and scale it, store as attribute
        picture = imread(str(filename))
        scaled = picture / 255.
        self.image = scaled
        # Creating an attribute isgray to help get dimensions of the image in Problems 4 and 5
        self.isgray = False
        # If the image is RGB
        if len(picture.shape) == 3:
            brightness = scaled.mean(axis=2)
            flatten_brightness = np.ravel(brightness)
        # Else the image is already in greyscale
        else:
            self.isgray = True
            brightness = scaled
            flatten_brightness = np.ravel(brightness)

        self.brightness = flatten_brightness

    # Problem 3
    def show_original(self):
        """Display the original image."""

        if len(self.image.shape) == 3:
            plt.imshow(self.image)
            plt.show()
        else:
            plt.imshow(self.image, cmap="gray")
            plt.show()

    # Problem 4
    def adjacency(self, r=5., sigma_B2=.02, sigma_X2=3.):
        """Compute the Adjacency and Degree matrices for the image graph."""
        # Getting dimensions of the image
        if self.isgray:
            m,n = self.image.shape
        else:
            m,n,z = self.image.shape
        
        # Initializing A and D
        A = lil_matrix((m*n, m*n), dtype="float")
        D = np.zeros(m*n)
        
        # Getting neigbors within radius and the distances from each
        for i in range(m*n):
            verticies, distances = get_neighbors(i, r, m, n)
            weights = []
            # Getting the weight
            for k in range(len(verticies)):
                if distances[k] < r:
                    weight = np.exp(0-((np.abs(self.brightness[i] - self.brightness[verticies[k]]))/(sigma_B2))-(distances[k]/sigma_X2))
                else:
                    weight = 0
                weights.append(weight)

            D[i] = np.sum(weights)
            A[i, verticies] = weights

        # Turning A into and CSC matrix
        A = csc_matrix(A)

        return A, D


    # Problem 5
    def cut(self, A, D):
        """Compute the boolean mask that segments the image."""
        # Getting dimnesions of the image
        if self.isgray:
            m,n = self.image.shape
        else:
            m,n,z = self.image.shape
        
        L = csgraph.laplacian(A)
        # Computing D^(-1/2)
        new_D = diags(D**(-.5))

        new_matrix = new_D @ L @ new_D
        # Getting eigenvalues and eigenvectors of the new matrix
        eigs, vecs = linalg.eigsh(new_matrix, which="SM", k=2)
        vecs = vecs[:,1]

        mask = np.reshape(vecs, (m,n)) > 0

        return mask

    # Problem 6
    def segment(self, r=5., sigma_B=.02, sigma_X=3.):
        """Display the original image and its segments."""
        # Getting A, D, and mask
        A, D = self.adjacency(r, sigma_B, sigma_X)
        mask = self.cut(A,D)

        # If the image is in grayscale
        if self.isgray:
            positive_segment = np.multiply(self.image, mask)
            negative_segment = np.multiply(self.image, mask)

            axis_1 = plt.subplot(131)
            axis_1.imshow(self.image)
            axis_1.set_title("Original")
            axis_2 = plt.subplot(132)
            axis_2.imshow(positive_segment)
            axis_2.set_title("Positive Segment")
            axis_3 = plt.subplot(133)
            axis_3.imshow(negative_segment)
            axis_3.set_title("Negative Segment")

        # If the image is in RGB
        else:
            mask = np.dstack((mask,mask,mask))
            positive_segment = self.image * mask
            negative_segment = self.image * ~mask

            axis_1 = plt.subplot(131)
            axis_1.imshow(self.image, cmap="gray")
            axis_1.set_title("Original")
            axis_2 = plt.subplot(132)
            axis_2.imshow(positive_segment, cmap="gray")
            axis_2.set_title("Positive Segment")
            axis_3 = plt.subplot(133)
            axis_3.imshow(negative_segment, cmap="gray")
            axis_3.set_title("Negative Segment")

        plt.tight_layout()
        plt.show()

        
#if __name__ == '__main__':
#     ImageSegmenter("dream_gray.png").segment()
#     ImageSegmenter("dream.png").segment()
#     ImageSegmenter("monument_gray.png").segment()
#     ImageSegmenter("monument.png").segment()
