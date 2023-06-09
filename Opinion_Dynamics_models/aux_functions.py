import random
import numpy as np

def initialize_random_matrix(N, M):
    matrix = np.array([[random.choice([-1, 1])
                        for _ in range(N)] for _ in range(M)])
    return matrix


def initialize_circle(n, m, perc):
    """
    Initialize a rectangular matrix which has binary 1/-1 values,
    being the ones forming a circle around the center of radius
    a given % of the smaller direction
    """
    radius = min(n, m)*perc
    matrix = np.where(np.array(
                [[(i-n/2)**2 + (j-m/2)**2 < radius**2
                  for i in range(n)] for j in range(m)]),
                1, -1)
    return matrix


def random_element(matrix):
    """
    Select the indexes of a random element within a matrix
    """
    n = len(matrix)
    m = len(matrix[0])
    random_row = random.randint(0, n - 1)
    random_col = random.randint(0, m - 1)
    return random_row, random_col


def random_neighbor(i, j, N, M):
    """
    Select a random neighbor of the element i,j within an array of size N, M.
    The array is considered cyclical, that is, the neighbors of element (0,0)
    are (0,1), (1,0), (0, N-1) and (N-1, 0)
    """
    neighbors = [(i-1, j), ((i+1)//N, j), (i, j-1), (i, (j+1)//M)]
    rnd_neighbor = random.choice(neighbors)
    return rnd_neighbor
