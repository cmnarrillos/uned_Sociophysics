import time
import datetime
import os
import matplotlib.pyplot as plt
from scipy.stats import poisson
from aux_network import *

# Identify the test (for saving results)
current_time = datetime.datetime.now()
id_test = 'net_' + current_time.strftime("%Y-%m-%d_%H-%M-%S")
# Create folder for results
if not os.path.exists('./tests/' + id_test):
    os.makedirs('./tests/' + id_test)

t0 = time.time()

# Load the edges from the file
edges_file = './data/CA-GrQc.txt'
graph = load_graph_edges(edges_file)


# Print basic information about the network
print('Number of nodes:', graph.number_of_nodes())
print('Number of edges:', graph.number_of_edges())
p = 0
# print('Number of categories:', len(categories))
print('Elapsed time for loading: ', time.time()-t0)
t0 = time.time()


# Get the cumulative distribution of node degrees
if True:
    degree = dict(graph.degree())
    degree_histogram = nx.degree_histogram(graph)
    # Calculate the cumulative distribution
    degree_cumulative = np.cumsum(degree_histogram)

    print('Average degree of the network: <k>=', np.mean(list(degree.values())))
    p = np.mean(list(degree.values())) / graph.number_of_nodes()
    print('Probability of 2 random nodes connected: p=', p)
    N = graph.number_of_nodes()

    # Baseline: random network with theoretical poisson distribution
    x = np.arange(0, N)  # Range of values to calculate CDF
    poisson_cdf = poisson.cdf(x, mu=np.mean(list(degree.values())))

    # Plotting the cumulative distribution
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(degree_cumulative)),
             degree_cumulative/max(degree_cumulative),
             'r', label='Arxiv network')
    plt.plot(x, poisson_cdf, 'k--', label='random network')
    plt.xlim([1, len(degree_cumulative)])
    plt.ylim([0, 1])
    plt.xlabel('Degree')
    plt.ylabel('Cumulative [% of nodes]')
    plt.grid('minor')
    plt.legend()
    plt.title('Degree Cumulative Distribution')
    plt.savefig(f'./tests/{id_test}/degree_cumulative.png')

    plt.figure(figsize=(8, 6))
    plt.semilogx(range(len(degree_cumulative)),
                 degree_cumulative/max(degree_cumulative),
                 'r', label='Arxiv network')
    plt.semilogx(x, poisson_cdf, 'k--', label='random network')
    plt.xlim([1, len(degree_cumulative)])
    plt.ylim([0, 1])
    plt.xlabel('Degree')
    plt.ylabel('Cumulative [% of nodes]')
    plt.grid('minor')
    plt.legend()
    plt.title('Degree Cumulative Distribution')
    plt.savefig(f'./tests/{id_test}/degree_cumulative_log.png')


# Calculate clustering coefficient for each node
if True:
    clustering_coefficients = nx.clustering(graph)
    # Get the sorted clustering coefficients
    sorted_coefficients = np.sort(list(clustering_coefficients.values()))
    # Calculate cumulative distribution
    num_nodes = len(sorted_coefficients)
    clustering_vals = np.linspace(0, 1, 1001)
    cumulative_distribution = np.searchsorted(sorted_coefficients,
                                              clustering_vals, side='right')

    # Plotting the cumulative distribution
    plt.figure(figsize=(8, 6))
    plt.plot(clustering_vals, cumulative_distribution/num_nodes, 'r')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('Clustering Coefficient')
    plt.ylabel('Cumulative [% of nodes]')
    plt.title('Clustering Coefficient Cumulative Distribution')
    plt.savefig(f'./tests/{id_test}/clustering_cumulative.png')

    print('Average clustering coefficient of the network: ', np.mean(sorted_coefficients))


# Compute the shortest paths for all pairs
if True:
    shortest_paths = dict(nx.all_pairs_shortest_path_length(graph))
    path_lengths = [length for paths in shortest_paths.values() for length in paths.values()]
    hist, bins = np.histogram(path_lengths, bins=range(max(path_lengths)+2))

    # Plotting the histogram
    plt.figure(figsize=(8, 6))
    plt.bar(bins[:-1], hist, width=1)
    plt.xlim([0, max(path_lengths) + 1])
    plt.xlabel('Shortest Path Length')
    plt.ylabel('Frequency')
    plt.title('Distribution of Shortest Path Lengths')
    plt.savefig(f'./tests/{id_test}/shortest_path_hist.png')

    print('Average shortest path between 2 points of the network: ', np.mean(path_lengths))
    print('Diameter of the network: ', np.max(path_lengths))

# Compute the Spectral density
if True:
    if not p:
        degree = dict(graph.degree())
        p = np.mean(list(degree.values())) / graph.number_of_nodes()
        N = graph.number_of_nodes()
    factor = np.sqrt(N * p * (1-p))

    # Get the adjacency matrix of the graph
    adj_matrix = nx.adjacency_matrix(graph).toarray()

    # Compute the eigenvalues
    eigenvalues = np.real(np.linalg.eigvals(adj_matrix))

    print('Max eigenvalue of adjacent matrix of the network: ', np.max(eigenvalues))
    print('Median eigenvalue: ', np.median(eigenvalues))

    lambda_array = np.linspace(-3*factor, max(eigenvalues), 1000)
    rho = compute_rho_array(eigenvalues, lambda_array)

    rho_random = compute_rho_array_random(N, p, lambda_array)

    # Plot the spectral density
    plt.figure(figsize=(8, 6))
    plt.plot(lambda_array / factor, rho * factor, 'r', label='Arxiv network')
    plt.plot(lambda_array / factor, rho_random * factor, 'k--', label='random network')
    plt.xlabel('$ \\lambda/\\sqrt{Np(1-p)}$')
    plt.ylabel('$ \\rho \\sqrt{Np(1-p)}$')
    plt.xlim([min(lambda_array)/factor, max(lambda_array)/factor])
    plt.ylim([0, 1.1*max(rho)*factor])
    plt.legend()
    plt.grid()
    plt.title('Spectral Density')
    plt.savefig(f'./tests/{id_test}/spectral_density.png')


print('Elapsed time for computing: ', time.time()-t0)


if True:
    # Plot the network
    plt.figure(figsize=(8, 8))
    # Create a layout for the nodes
    layout = nx.spring_layout(
        graph,
        k=0.3,  # Adjust the optimal distance between nodes
        iterations=100,  # Increase the number of iterations
        scale=2,  # Adjust the scaling factor
        center=(0, 0),  # Specify the center coordinates
        seed=42,  # Set a specific random seed
    )
    # Draw nodes
    nx.draw_networkx_nodes(
        graph,
        layout,
        node_color=list(degree.values()),  # Adjust the node color to colormap
        cmap='coolwarm',
        vmin=1,
        vmax=max(list(degree.values())),
        node_size=4,  # Adjust the node size as desired
    )
    # Draw edges
    nx.draw_networkx_edges(
        graph,
        layout,
        edge_color='k',  # Adjust the edge color as desired
        width=0.15,  # Adjust the edge width as desired
        alpha=0.5,  # Adjust the edge transparency as desired
    )
    sm = plt.cm.ScalarMappable(cmap='coolwarm',
                               norm=plt.Normalize(vmin=1,
                                                  vmax=max(list(degree.values()))))
    sm.set_array([])
    cbar = plt.colorbar(sm, label='Degree')
    plt.title('General Relativity Arxiv (1993-2003)')
    plt.axis('off')
    plt.savefig(f'./tests/{id_test}/network_schema.png')

if True:
    # Plot a large copy of the network
    plt.figure(figsize=(20, 20))
    # Draw nodes
    nx.draw_networkx_nodes(
        graph,
        layout,
        node_color=list(degree.values()),  # Adjust the node color to colormap
        cmap='coolwarm',
        vmin=1,
        vmax=max(list(degree.values())),
        node_size=10,  # Adjust the node size as desired
    )
    # Draw edges
    nx.draw_networkx_edges(
        graph,
        layout,
        edge_color='k',  # Adjust the edge color as desired
        width=0.15,  # Adjust the edge width as desired
        alpha=0.5,  # Adjust the edge transparency as desired
    )
    sm = plt.cm.ScalarMappable(cmap='coolwarm',
                               norm=plt.Normalize(vmin=1,
                                                  vmax=max(list(degree.values()))))
    sm.set_array([])
    cbar = plt.colorbar(sm, label='Degree')
    plt.title('General Relativity Arxiv (1993-2003)')
    plt.axis('off')
    plt.savefig(f'./tests/{id_test}/network_schema.pdf')

# Show plots
plt.show()
