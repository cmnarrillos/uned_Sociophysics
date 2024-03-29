
import numpy as np
import networkx as nx
import scipy.sparse.linalg as sp_linalg


def load_graph_edges(file):
    edges = []
    with open(file, 'r') as f:
        for line in f:
            if '#' in line:
                continue
            node1, node2 = line.strip().split()
            edges.append((int(node1), int(node2)))

    # Create a graph and add edges
    graph = nx.Graph()
    graph.add_edges_from(edges)

    return graph


def load_node_categories(file):
    categories = {}
    with open(file, 'r') as f:
        for line in f:
            category = line.strip().split(';')[0].replace('Category:', '')
            nodes = list(map(int, line.strip().split(';')[1].split()))
            categories[category] = nodes

    return categories


def power_iteration(adj_matrix, num_iterations):
    n = adj_matrix.shape[0]
    v = np.random.rand(n)
    for _ in range(num_iterations):
        v = adj_matrix @ v
        v = v / np.linalg.norm(v)

    return v


def estimate_spectral_density(graph, num_eigenvalues=100):
    # Convert the graph to a sparse adjacency matrix
    adj_matrix = nx.adjacency_matrix(graph)

    # Calculate the spectral density using Lanczos algorithm
    eigenvalues, _ = sp_linalg.eigsh(adj_matrix, k=num_eigenvalues)

    return eigenvalues


def dirac_delta(x, x0, sigma=1e-5):
    return np.exp(-0.5 * ((x - x0) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def compute_rho_array(eigenvalues, lambda_array):
    N = len(eigenvalues)
    rho_array = []
    threshold = lambda_array[1]-lambda_array[0]
    for lambda_val in lambda_array:
        delta_vals = np.array([dirac_delta(lambda_val, eig_val, threshold)
                               for eig_val in eigenvalues])
        rho_val = np.sum(delta_vals) / N
        rho_array.append(rho_val)
    return np.array(rho_array)


def compute_rho_array_random(N, p, lambda_array):
    rho_array = []
    for lambda_val in lambda_array:
        if abs(lambda_val) < 2*np.sqrt(N * p * (1-p)):
            rho_val = np.sqrt(4 * N * p * (1-p) - lambda_val**2) / \
                      (2 * np.pi * N * p * (1 - p))
        else:
            rho_val = 0
        rho_array.append(rho_val)
    return np.array(rho_array)


def find_subgraphs_recursive(graph):
    subgraphs = []  # List to store the identified subgraphs
    visited = set()  # Set to keep track of visited nodes

    def dfs(node, subgraph):
        visited.add(node)
        subgraph.add_node(node)

        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                dfs(neighbor, subgraph)

    for node in graph.nodes:
        if node not in visited:
            subgraph = nx.Graph()
            dfs(node, subgraph)
            subgraphs.append(subgraph)

    return subgraphs
