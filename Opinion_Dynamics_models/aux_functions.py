import random
import networkx as nx
import numpy as np

random.seed(11859)


def initialize_random_scalar_network(N, M, bias=0.5):
    choices = [-1, 1]
    probabilities = [1 - bias, bias]
    network = np.random.choice(choices, size=(N, M), p=probabilities)
    return network


def initialize_circular_scalar_network(n, m, perc):
    """
    Initialize a rectangular matrix which has binary 1/-1 values,
    being the ones forming a circle around the center of radius
    a given % of the smaller direction
    """
    radius = min(n, m)*perc
    network = np.where(np.array(
                [[(i-n/2)**2 + (j-m/2)**2 < radius**2
                  for j in range(m)] for i in range(n)]),
                1, -1)
    return network


def initialize_random_vector_network(N, M, F, q):
    network = np.random.randint(0, q, size=(N, M, F))
    categories = np.arange(q)
    network = np.take(categories, network)
    return network


def random_element(matrix):
    """
    Select the indexes of a random element within a matrix
    """
    n = len(matrix)
    m = len(matrix[0])
    random_row = random.randint(0, n - 1)
    random_col = random.randint(0, m - 1)
    return random_row, random_col


def random_neighbor(coords, N, M):
    """
    Select a random neighbor of the element i,j within an array of size N, M.
    The array is considered cyclical, that is, the neighbors of element (0,0)
    are (0,1), (1,0), (0, N-1) and (N-1, 0)
    """
    i, j = coords
    neighbors = [(i-1, j), ((i+1) % N, j), (i, j-1), (i, (j+1) % M)]
    rnd_neighbor = random.choice(neighbors)
    return rnd_neighbor


def sznajd_neighbors(elem, N, M):
    """
    Returns:
        - partner neighbor indexes
        - a list with indexes of the neighbors to be updated when using
          Sznajd method in 2D. Elements are ordered in such a way that
          neighbors [0:3] take the value of the partner while neighbors
          [3:6] take the value of element (i,j)
    """
    i, j = elem
    partner_neighbor = (i, (j+1) % M)
    neighbors_to_update = [(i-1,        j),
                           (i,          j-1),
                           ((i+1) % N,  j),
                           (i-1,       (j+1) % M),
                           (i,         (j+2) % M),
                           ((i+1) % N, (j+1) % M)]
    return partner_neighbor, neighbors_to_update


def create_small_world_network(N, k, p, bias=0.5):
    """
    Initializes Small World Network of N agents with k nerarest
    neighbors and k probability of rewiring.
    Each agent is initialized with an opinion sigma which is
    randomly set to either -1 or 1 (being biased towards 1 as
    given by bias param)
    """
    # Create a regular ring lattice
    ring_lattice = nx.watts_strogatz_graph(N, k, p)

    # Create a small-world network by rewiring edges
    network = nx.connected_watts_strogatz_graph(N, k, p)

    # Combine the attributes of the original ring lattice
    for node in network.nodes:
        network.nodes[node].update(ring_lattice.nodes[node])

    # Assign a random sigma value to each node
    for node in network.nodes:
        sigma = -1 if random.uniform(0, 1) > bias else 1
        network.nodes[node]['sigma'] = sigma

    return network


def proportion_different_sigma_connections(network):
    """
    Gets the proportion of connections between agents
    with different opinion in the network.
    """
    different_sigma_connections = 0
    total_connections = 0

    for edge in network.edges:
        node1 = edge[0]
        node2 = edge[1]
        sigma1 = network.nodes[node1]['sigma']
        sigma2 = network.nodes[node2]['sigma']

        total_connections += 1
        if sigma1 != sigma2:
            different_sigma_connections += 1

    return different_sigma_connections/total_connections


def initialize_schelling_network(N, p, red_fraction):
    """
    Initialize the network for implementing Schelling segregation
     model with two types of agents: red and blue.
    """
    network = nx.grid_2d_graph(N, N)
    num_agents = int((1-p) * N * N)
    agent_nodes = random.sample(list(network.nodes), num_agents)

    red_nodes = random.sample(agent_nodes, int(red_fraction * num_agents))
    blue_nodes = list(set(agent_nodes) - set(red_nodes))

    nx.set_node_attributes(network,
                           {node: '' for node in network.nodes},
                           name='color')
    nx.set_node_attributes(network,
                           {node: 'red' for node in red_nodes},
                           name='color')
    nx.set_node_attributes(network,
                           {node: 'blue' for node in blue_nodes},
                           name='color')

    return network


def compute_similarity(network, node):
    """
    Compute the similarity between a given node and its neighbors.
    Needed for Schelling segregation model
    """
    color = network.nodes[node]['color']
    neighbors = list(network.neighbors(node))  # Convert neighbors iterator to a list
    num_neighbors = sum(network.nodes[neighbor]['color'] != '' for neighbor in neighbors)
    similar_neighbors = sum(network.nodes[neighbor]['color'] == color for neighbor in neighbors)
    similarity = similar_neighbors / num_neighbors if num_neighbors > 0 else 1  # Handle division by zero
    return similarity
