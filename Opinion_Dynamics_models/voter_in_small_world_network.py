import datetime
import os
import time
import matplotlib.pyplot as plt
from aux_functions import *


# Identify the test (for saving results)
current_time = datetime.datetime.now()
id_test = 'voter_SWN_' + current_time.strftime("%Y-%m-%d_%H-%M-%S")
# Create folder for results
if not os.path.exists('./tests/' + id_test):
    os.makedirs('./tests/' + id_test)


# PARAMS of the test
max_iter = 1000000
num_max_stuck = max(200, max_iter//100)
bias = 0.5
# Parameters of the Small World Network
n = 500  # Number of agents
k = 8   # Number of nearest neighbors to connect
p_max = 0.2  # Probability of rewiring (used as pmax if next param is false)
just_1_p = False
n_p_tries = 10

# Try different p values or not depending on param
rho_multi = []
dt = 0
for ii in range(n_p_tries+1):
    if just_1_p:
        p = p_max
    else:
        p = ii/n_p_tries*p_max

    # Initialize Small World Network
    small_world_network = create_small_world_network(n, k, p, bias)

    # Plotting the network
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(small_world_network)  # Layout for visualization
    # Draw nodes
    nx.draw_networkx_nodes(
        small_world_network,
        pos,
        node_color='b',  # Adjust the node color as desired
        node_size=20,  # Adjust the node size as desired
    )
    # Draw edges
    nx.draw_networkx_edges(
        small_world_network,
        pos,
        edge_color='k',  # Adjust the edge color as desired
        width=0.25,  # Adjust the edge width as desired
        alpha=1,  # Adjust the edge transparency as desired
    )
    plt.title(f'Network schema (p={round(p,3)})')
    plt.axis('off')  # Disable axis display
    plt.savefig(f'./tests/{id_test}/network_{ii}.png')
    plt.close()


    rho = [proportion_different_sigma_connections(small_world_network)]
    no_changes_since = 0
    t0 = time.time()


    # Voter model
    for iteration in range(max_iter):
        # Select a random node from the network
        random_node = random.choice(list(small_world_network.nodes))

        # Get the neighbors of the selected node
        neighbors = list(small_world_network.neighbors(random_node))

        # Select a random neighbor from the list of neighbors
        random_neighbor = random.choice(neighbors)

        # Update the opinion of agent ii according to Voter model
        if small_world_network.nodes[random_node]['sigma'] == \
                small_world_network.nodes[random_neighbor]['sigma']:
            no_changes_since += 1
        else:
            small_world_network.nodes[random_node]['sigma'] = \
                small_world_network.nodes[random_neighbor]['sigma']
            no_changes_since = 0

        # Store rho value
        rho.append(proportion_different_sigma_connections(
            small_world_network))

        # Exit the loop if there are no updates
        if no_changes_since == num_max_stuck:
            print(f'There have been {num_max_stuck} steps without changes.'
                  f'Process terminated.')
            break

    if just_1_p:
        break
    else:
        rho_multi.append(rho)


    dt = dt + time.time() - t0


# Plot order parameter during simulation
plt.figure(figsize=(8, 6))
if just_1_p:
    plt.plot(rho)
else:
    for ii in range(n_p_tries+1):
        plt.plot(rho_multi[ii], label=f'p={round(ii/n_p_tries*p_max,3)}')
plt.xlabel('iterations (t)')
plt.ylabel('$\\rho$')
if just_1_p:
    plt.title(f'Order parameter (p={round(p,3)})')
    plt.xlim([0, len(rho)])
    plt.ylim([min(rho), 1])
else:
    plt.legend()
    plt.title('Order parameter')
    plt.xlim([0, max([len(rho_) for rho_ in rho_multi])])
    plt.ylim([min([min(rho_) for rho_ in rho_multi]), 1])
plt.tight_layout()
plt.savefig(f'./tests/{id_test}/order_evolution.png')
plt.close()

plt.figure(figsize=(8, 6))
if just_1_p:
    plt.loglog(rho)
else:
    for ii in range(n_p_tries+1):
        plt.loglog(rho_multi[ii], label=f'p={round(ii/n_p_tries*p_max,3)}')
plt.xlabel('iterations (t)')
plt.ylabel('$\\rho$')
if just_1_p:
    plt.title(f'Order parameter (p={round(p,3)})')
    plt.xlim([0, len(rho)])
    plt.ylim([min(rho), 1])
else:
    plt.legend()
    plt.title('Order parameter')
    plt.xlim([0, max([len(rho_) for rho_ in rho_multi])])
    plt.ylim([min([min(rho_) for rho_ in rho_multi]), 1])
plt.tight_layout()
plt.savefig(f'./tests/{id_test}/order_evolution_log.png')
plt.close()


# Document the test
with open(f'./tests/{id_test}/doc_test.txt', 'w') as f:
    f.write(f'Voter test with population belonging to Small'
            f' World Network of size {n}\n')
    if just_1_p:
        f.write(f'Number of nearest neighbors: {k}, rewiring prob: {round(p,3)}\n\n')
    else:
        f.write(f'Number of nearest neighbors: {k}, rewiring prob '
                f'takes {n_p_tries} equispaced values between 0 and {round(p_max,3)}\n\n')
    f.write(f'Initial random distribution of 2 opinions biased with '
            f'{round(100*bias,2)}% supporting [1]\n\n')
    f.write(f'Max # of iterations allowed: {max_iter}\n')
    f.write(f'Stop criteria: no evolution since {num_max_stuck} steps ago\n\n')
    if iteration < max_iter-1:
        f.write(f'Process finished at iter {iteration}\n\n')
    else:
        f.write(f'Process stopped due to max iter criteria\n\n')
    f.write(f'Time employed for running and plotting intermediate '
            f'steps: {dt} s')


