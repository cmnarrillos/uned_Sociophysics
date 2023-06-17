
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
