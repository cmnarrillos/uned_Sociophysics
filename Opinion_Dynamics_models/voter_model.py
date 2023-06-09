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

max_iter = 1000000
num_max_stuck = max(200, max_iter//100)
circle = True
radius = 0.55

# Shape of the population (N, M)
n = 200
m = 250


# Initialize population opinion
if circle:
    initial_state = initialize_circle(n, m, radius)
else:
    initial_state = initialize_random_matrix(n, m)


t0 = time.time()

population_opinion = initial_state

no_changes_since = 0

for iteration in range(max_iter):
    # Select a random element of the matrix: ii
    ii_x, ii_y = random_element(population_opinion)

    # Select a random neighbor of this element: jj
    jj_x, jj_y = random_neighbor(ii_x, ii_y, n, m)

    # Update the opinion of the agent ii according to Voter model
    if population_opinion[ii_x, ii_y] == population_opinion[jj_x, jj_y]:
        no_changes_since += 1
    else:
        population_opinion[ii_x, ii_y] = population_opinion[jj_x, jj_y]
        no_changes_since = 0

    # Exit the loop if there are no updates
    if no_changes_since == num_max_stuck:
        print(f'There have been {num_max_stuck} steps without changes.'
              f'Process terminated.')
        break

    # Plot intermediate steps through the process
    if iteration % (max_iter//10) == 0:
        plt.imshow(population_opinion, cmap='gray')
        plt.savefig(f'./tests/{id_test}/population_iter{iteration}.png')

dt = time.time() - t0

# Plot the population at the end of the process
plt.imshow(population_opinion, cmap='gray')
plt.savefig(f'./tests/{id_test}/population_end.png')


with open(f'./tests/{id_test}/doc_test.txt', 'w') as f:
    f.write(f'Voter test with population shape [{n}, {m}]\n\n')
    if circle:
        f.write(f'Initial opinions shape: circle of radius '
                f'{int(min(n,m)*radius)} centered at ({n/2}, {m/2})\n\n')
    else:
        f.write(f'Initial random distribution of 2 opinions\n\n')
    f.write(f'Max # of iterations allowed: {max_iter}\n')
    f.write(f'Stop criteria: no evolution since {num_max_stuck} steps ago\n\n')
    if iteration < max_iter-1:
        f.write(f'Process finished at iter {iteration}\n\n')
    else:
        f.write(f'Process stopped due to max iter criteria\n\n')
    f.write(f'Time employed for running and plotting intermediate '
            f'steps: {dt} s')
