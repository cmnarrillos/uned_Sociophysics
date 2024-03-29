import datetime
import os
import time
import matplotlib.pyplot as plt
from aux_functions import *


# Identify the test (for saving results)
current_time = datetime.datetime.now()
id_test = 'voter_' + current_time.strftime("%Y-%m-%d_%H-%M-%S")
# Create folder for results
if not os.path.exists('./tests/' + id_test):
    os.makedirs('./tests/' + id_test)


# PARAMS of the test
max_iter = 10000000
num_max_stuck = max(200, max_iter//100)
circle = False
radius = 0.48
bias = 0.5

# Shape of the population (N, M)
n = 40
m = 50


# Initialize population opinion
if circle:
    population_opinion = initialize_circular_scalar_network(n, m, radius)
else:
    population_opinion = initialize_random_scalar_network(n, m, bias)

# Plot the initial state
plt.figure(figsize=(8, 6))
plt.imshow(population_opinion, cmap='gray')
plt.title(f'Initial state of Population Opinion')
plt.colorbar(ticks=[-1, 1])
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.savefig(f'./tests/{id_test}/population_init.png')
plt.close()


no_changes_since = 0
num_1s = [np.count_nonzero(population_opinion == 1)]
rho = [proportion_different_sigma_lattice(population_opinion)]
t0 = time.time()

# Voter model
for iteration in range(max_iter):
    # Select a random element of the matrix: ii
    elem = random_element(population_opinion)

    # Select a random neighbor of this element: jj
    neighbor = random_neighbor(elem, n, m)

    # Update the opinion of agent ii according to Voter model
    if population_opinion[elem] == population_opinion[neighbor]:
        no_changes_since += 1
    else:
        population_opinion[elem] = population_opinion[neighbor]
        no_changes_since = 0

    # Track population support of idea [1]
    num_1s.append(np.count_nonzero(population_opinion == 1))
    # Store order parameter
    rho.append(proportion_different_sigma_lattice(population_opinion))

    # Exit the loop if there are no updates
    if no_changes_since == num_max_stuck:
        print(f'There have been {num_max_stuck} steps without changes.'
              f'Process terminated.')
        break

    # Plot intermediate steps through the process
    if (iteration+1) % (max_iter//100) == 0:
        plt.figure(figsize=(8, 6))
        plt.imshow(population_opinion, cmap='gray')
        plt.title(f'Population Opinion after {iteration+1} iterations')
        plt.colorbar(ticks=[-1, 1])
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(f'./tests/{id_test}/population_iter{iteration+1}.png')
        plt.close()

dt = time.time() - t0


# Plot the population at the end of the process
plt.figure(figsize=(8, 6))
plt.imshow(population_opinion, cmap='gray')
if iteration < max_iter-1:
    plt.title(f'Population Opinion after {iteration+1} iterations')
else:
    plt.title(f'Population Opinion after {max_iter} iterations')
plt.colorbar(ticks=[-1, 1])
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.savefig(f'./tests/{id_test}/population_end.png')
plt.close()


# Plot Support evolution during simulation
pop_size = np.size(population_opinion)
plt.figure(figsize=(8, 6))
plt.plot([supporters*100/pop_size for supporters in num_1s])
plt.title(f'Population sharing opinion [1]')
plt.xlabel(f'iterations')
plt.ylabel(f'% supporters')
plt.xlim([0, len(num_1s)])
plt.tight_layout()
plt.savefig(f'./tests/{id_test}/support_evolution_1.png')
plt.ylim([0, 100])
plt.savefig(f'./tests/{id_test}/support_evolution.png')
plt.close()


# Plot order parameter during simulation
plt.figure(figsize=(8, 6))
plt.plot(rho)
plt.xlabel('iterations (t)')
plt.ylabel('$\\rho$')
plt.title(f'Order parameter')
plt.xlim([0, len(rho)])
plt.ylim([min(rho), 1])
plt.tight_layout()
plt.grid()
plt.savefig(f'./tests/{id_test}/order_evolution.png')
plt.close()

plt.figure(figsize=(8, 6))
plt.loglog(rho)
plt.xlabel('iterations (t)')
plt.ylabel('$\\rho$')
plt.title(f'Order parameter')
plt.xlim([0, len(rho)])
plt.ylim([min(rho), 1])
plt.tight_layout()
plt.grid()
plt.savefig(f'./tests/{id_test}/order_evolution_log.png')
plt.close()


# Document the test
with open(f'./tests/{id_test}/doc_test.txt', 'w') as f:
    f.write(f'Voter test with population shape [{n}, {m}]\n\n')
    if circle:
        f.write(f'Initial opinions shape: circle of radius '
                f'{int(min(n,m)*radius)} centered at ({n/2}, {m/2})\n\n')
    else:
        f.write(f'Initial random distribution of 2 opinions biased with '
                f'{100*bias}% supporting [1]\n\n')
    f.write(f'Max # of iterations allowed: {max_iter}\n')
    f.write(f'Stop criteria: no evolution since {num_max_stuck} steps ago\n\n')
    if iteration < max_iter-1:
        f.write(f'Process finished at iter {iteration}\n\n')
    else:
        f.write(f'Process stopped due to max iter criteria\n\n')
    f.write(f'Time employed for running and plotting intermediate '
            f'steps: {dt} s')
