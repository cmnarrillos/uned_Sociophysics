import numpy as np
import math
import matplotlib.pyplot as plt
import os
from opinion_dyn_functions import *

folder_name = "results"

if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    print(f"Folder '{folder_name}' created successfully.")
else:
    print(f"Folder '{folder_name}' already exists.")

k_values = [0, 0.25, 0.5, 0.75, 1]

# Compute the relation between a_{t+1} and a_t for even groups

# Values of a_t to compute
x = np.linspace(0, 1, 101)

# Compute for different k values for even groups
for k in k_values:
    group_size_even = []
    a_t1_even = []
    for r in range(2, 21, 2):
        y = update_support_even(x, r, k)
        a_t1_even.append(y)
        group_size_even.append(r)

    plot_threshold(x, a_t1_even, group_size_even, k)

# Compute for odd groups
group_size_odd = []
a_t1_odd = []
for r in range(3, 20, 2):
    y = update_support_odd(x, r)
    a_t1_odd.append(y)
    group_size_odd.append(r)

plot_threshold(x, a_t1_odd, group_size_odd)

# Find critical support values
ac = []
for k in k_values:
    ac_i = []
    group_size = []
    for r in range(2, 51, 2):
        ac_i.append(critical_support_even(r, k))
        group_size.append(r)
    ac.append(ac_i)

plt.figure()
for k, ac_i in zip(k_values, ac):
    plt.plot(group_size, ac_i, label=f'$k={k}$')
plt.title('Critical support')
plt.xlabel('Group Size')
plt.ylabel('$a_{c,r}$')
plt.legend()
plt.xlim([0, 50])
plt.ylim([0, 1])
plt.grid('minor')
plt.savefig(f'./{folder_name}/critical_support.png')
plt.close()
