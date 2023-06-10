import datetime
import os
import time
import matplotlib.pyplot as plt
from aux_functions import *


# Identify the test (for saving results)
current_time = datetime.datetime.now()
id_test = 'schelling_' + current_time.strftime("%Y-%m-%d_%H-%M-%S")
# Create folder for results
if not os.path.exists('./tests/' + id_test):
    os.makedirs('./tests/' + id_test)


# PARAMS of the test
max_iter = 100

# Schelling model parameters
N = 100  # Grid size (N x N)
p = 0.08  # Voids density
red_fraction = 0.45  # Fraction of red agents
threshold = 0.55  # Similarity threshold for agent movement

# Initialize the network
network = initialize_schelling_network(N, p, red_fraction)

# Plot initial state
pos = {(x, y): (x, y) for x, y in network.nodes}
node_colors = ['w' if network.nodes[node]['color'] == ''
               else network.nodes[node]['color']
               for node in network.nodes]
plt.figure(figsize=(8, 8))
nx.draw(network, pos, node_size=250000/N**2,
        node_color=node_colors, with_labels=False)
plt.title(f'Schelling Segregation Model initial state', loc='left')
plt.axis('off')  # Disable axis display
plt.savefig(f'./tests/{id_test}/segregation_init.png')
plt.close()


t0 = time.time()
# Simulate Schelling segregation model
node_list = list(network.nodes)
for iteration in range(max_iter):
    move_occurred = False
    unsatisfied_agents = 0

    random.shuffle(node_list)
    for node in node_list:
        # Skip empty nodes
        if network.nodes[node]['color'] == '':
            continue

        # Get satisfaction of the agent
        similarity = compute_similarity(network, node)
        # If the agent is not satisfied, moves to an empty node
        if similarity < threshold:
            unsatisfied_agents += 1
            vacant_nodes = list([n for n in network.nodes
                                 if network.nodes[n]['color'] == ''])
            random.shuffle(vacant_nodes)
            if vacant_nodes:
                for new_location in vacant_nodes:
                    # Try moving to vacant node
                    network.nodes[new_location]['color'] = \
                        network.nodes[node]['color']
                    # Check satisfaction
                    satisfaction_new = compute_similarity(network, new_location)

                    if not satisfaction_new < threshold:
                        # Moved to vacant location (old location is now empty)
                        network.nodes[node]['color'] = ''
                        move_occurred = True
                        unsatisfied_agents -= 1
                        break
                    else:
                        # Try again (leave vacant location empty)
                        network.nodes[new_location]['color'] = ''

    # The simulation ends if all agents are satisfied or
    # there's no available space
    if not move_occurred:
        break

    # Plot intermediate steps through the process
    if (iteration+1) % (max_iter//20) == 0:
        pos = {(x, y): (x, y) for x, y in network.nodes}
        node_colors = ['w' if network.nodes[node]['color'] == ''
                       else network.nodes[node]['color']
                       for node in network.nodes]
        plt.figure(figsize=(8, 8))
        nx.draw(network, pos, node_size=250000/N**2,
                node_color=node_colors, with_labels=False)
        plt.title(f'Schelling Segregation Model '
                  f'after {iteration+1} steps')
        plt.savefig(f'./tests/{id_test}/'
                    f'segregation_iter{iteration+1}.png')
        plt.close()

dt = time.time() - t0

# Plot the final state
pos = {(x, y): (x, y) for x, y in network.nodes}
node_colors = ['w' if network.nodes[node]['color'] == ''
               else network.nodes[node]['color']
               for node in network.nodes]
plt.figure(figsize=(8, 8))
nx.draw(network, pos, node_size=250000/N**2,
        node_color=node_colors, with_labels=False)
plt.title(f'Schelling Segregation Model after {iteration+1} steps')
plt.savefig(f'./tests/{id_test}/segregation_end.png')
plt.close()


# Document the test
with open(f'./tests/{id_test}/doc_test.txt', 'w') as fw:
    fw.write(f'Schelling test with population shape [{N}, {N}]\n\n')
    fw.write(f'Initial random distribution of {int((1-p)*N*N)} agents with '
             f'{int(p*N*N)} vacant nodes, being {int((1-p)*N*N*red_fraction)} '
             f'agents red and {int((1-p)*N*N*(1-red_fraction))}, blue\n')
    fw.write(f'Threshold for being satisfied: {100*threshold}% of '
             f'neighbors sharing the same group\n\n')
    fw.write(f'Max # of iterations allowed: {max_iter}\n')
    fw.write(f'Stop criteria: no movement of any agent in last step\n\n')
    if iteration < max_iter-1:
        fw.write(f'Process finished at iter {iteration}\n\n')
    else:
        fw.write(f'Process stopped due to max iter criteria\n\n')
    fw.write(f'There are {unsatisfied_agents} agents which are still '
             f'unsatisfied but could not find a suitable node to move into \n')
    fw.write(f'Time employed for running and plotting intermediate '
             f'steps: {dt} s')
