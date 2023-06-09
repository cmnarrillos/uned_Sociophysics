import datetime
import os
import matplotlib.pyplot as plt
from aux_functions import *


# Identify the test (for saving results)
current_time = datetime.datetime.now()
id_test = 'voter_' + current_time.strftime("%Y-%m-%d_%H-%M-%S")
# Create folder for results
if not os.path.exists('./tests/' + id_test):
    os.makedirs('./tests/' + id_test)

max_iter = 100000
num_max_stuck = 1000
circle = True
radius = 0.6

# Shape of the population (N, M)
n = 200
m = 250

# Initialize population opinion
if circle:
    initial_state = initialize_circle(n, m, radius)
else:
    initial_state = initialize_random_matrix(n, m)


# Plot the population at the start of the process
plt.imshow(initial_state, cmap='gray')
plt.savefig(f'./tests/{id_test}/population_initial.png')


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
    if not iteration//(max_iter//10):
        plt.imshow(population_opinion, cmap='gray')
        plt.savefig(f'./tests/{id_test}/population_iter{iteration}.png')


# Plot the population at the end of the process
plt.imshow(population_opinion, cmap='gray')
plt.savefig(f'./tests/{id_test}/population_end.png')
plt.show()
