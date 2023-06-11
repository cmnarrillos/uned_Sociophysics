import datetime
import os
import time
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from aux_functions import *


# Identify the test (for saving results)
current_time = datetime.datetime.now()
id_test = 'axelrod_' + current_time.strftime("%Y-%m-%d_%H-%M-%S")
# Create folder for results
if not os.path.exists('./tests/' + id_test):
    os.makedirs('./tests/' + id_test)


# PARAMS of the test
max_iter = 2000000
num_max_stuck = max(200, max_iter//100)
# Shape of the network
n = 20
m = 25
# Number of attributes
f = 8
# Number of possible values for each attribute
q = 4


# Initialize population profiles
population_culture = initialize_random_vector_network(n, m, f, q)

# Get the colormap
if q <= 10:
    catcmap = ListedColormap(plt.get_cmap(f"tab10").colors[:q])
else:
    catcmap = ListedColormap(plt.get_cmap(f"tab20").colors[:q])

# Plot the initial state
for ii in range(f):
    plt.figure(figsize=(8, 6))
    im = plt.imshow(population_culture[:, :, ii], cmap=catcmap)
    im.set_clim(0, q-1)
    plt.title(f'Initial state of Population cultural profile '
              f'(feature {ii+1})')
    plt.colorbar(ticks=[jj for jj in range(q)])
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(f'./tests/{id_test}/population_dim{ii+1}_init.png')
    plt.close()


no_changes_since = 0
rho = [[proportion_different_sigma_lattice(
        population_culture[:, :, ii]) for ii in range(f)]]
t0 = time.time()

# Axelrod model
for iteration in range(max_iter):
    # Select a random element of the matrix: ii
    elem = random_element(population_culture)

    # Select a random neighbor of this element: jj
    neighbor = random_neighbor(elem, n, m)

    # Update the profile of agent ii according to Axelrod model
    p = np.sum(population_culture[elem] ==
               population_culture[neighbor])
    not_equal_indices = np.where(population_culture[elem] !=
                                 population_culture[neighbor])[0]
    if (p < f) & (random.uniform(0, 1) < p/f):
        kk = random.choice(not_equal_indices)
        population_culture[elem][kk] = population_culture[neighbor][kk]
        no_changes_since = 0
    else:
        no_changes_since += 1

    # Store order parameter
    rho.append([proportion_different_sigma_lattice(
                population_culture[:, :, ii]) for ii in range(f)])

    # Exit the loop if there are no updates
    if no_changes_since == num_max_stuck:
        print(f'There have been {num_max_stuck} steps without changes.'
              f'Process terminated.')
        break

    # Plot intermediate steps through the process
    if (iteration+1) % (max_iter//20) == 0:
        for ii in range(f):
            plt.figure(figsize=(8, 6))
            im = plt.imshow(population_culture[:, :, ii], cmap=catcmap)
            im.set_clim(0, q-1)
            plt.title(f'Population cultural profile (feature {ii+1})'
                      f' after {iteration+1} iterations')
            plt.colorbar(ticks=[jj for jj in range(q)])
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            plt.savefig(f'./tests/{id_test}/population_'
                        f'dim{ii+1}_iter{iteration+1}.png')
            plt.close()

dt = time.time() - t0

# Plot the population at the end of the process
for ii in range(f):
    plt.figure(figsize=(8, 6))
    im = plt.imshow(population_culture[:, :, ii], cmap=catcmap)
    im.set_clim(0, q-1)
    if iteration < max_iter-1:
        plt.title(f'Population cultural profile (feature {ii+1})'
                  f' after {iteration+1} iterations')
    else:
        plt.title(f'Population cultural profile (feature {ii+1})'
                  f' after {max_iter} iterations')
    plt.colorbar(ticks=[jj for jj in range(q)])
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(f'./tests/{id_test}/population_dim{ii+1}_end.png')
    plt.close()


# Plot order parameter during simulation
plt.figure(figsize=(8, 6))
for ii in range(f):
    plt.plot([rr[ii] for rr in rho], label=f'feature {ii+1}')
plt.xlabel('iterations (t)')
plt.ylabel('$\\rho$')
plt.title(f'Order parameter')
plt.legend()
plt.xlim([0, max([len([rr[ii] for rr in rho]) for ii in range(f)])] )
plt.ylim([min([min([rr[ii] for rr in rho]) for ii in range(f)]), 1])
plt.tight_layout()
plt.grid()
plt.savefig(f'./tests/{id_test}/order_evolution.png')
plt.close()

plt.figure(figsize=(8, 6))
for ii in range(f):
    plt.loglog([rr[ii] for rr in rho], label=f'feature {ii+1}')
plt.xlabel('iterations (t)')
plt.ylabel('$\\rho$')
plt.title(f'Order parameter')
plt.legend()
plt.xlim([0, max([len([rr[ii] for rr in rho]) for ii in range(f)])] )
plt.ylim([min([min([rr[ii] for rr in rho]) for ii in range(f)]), 1])
plt.tight_layout()
plt.grid()
plt.savefig(f'./tests/{id_test}/order_evolution_log.png')
plt.close()


# Document the test
with open(f'./tests/{id_test}/doc_test.txt', 'w') as fw:
    fw.write(f'Axelrod test with population shape [{n}, {m}]\n\n')
    fw.write(f'Initial random distribution of cultural profiles with'
            f' {f} attributes which can take {q} different categories each\n\n')
    fw.write(f'Max # of iterations allowed: {max_iter}\n')
    fw.write(f'Stop criteria: no evolution since {num_max_stuck} steps ago\n\n')
    if iteration < max_iter-1:
        fw.write(f'Process finished at iter {iteration}\n\n')
    else:
        fw.write(f'Process stopped due to max iter criteria\n\n')
    fw.write(f'Time employed for running and plotting intermediate '
             f'steps: {dt} s')
