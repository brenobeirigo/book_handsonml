#
#%%

import numpy as np
batch_size = 2
size_prediction_window = 2
n_time_steps = 3
features = 1
shape = (batch_size, n_time_steps + size_prediction_window, features)
shape_Y = (batch_size, n_time_steps, size_prediction_window, features)

shape_batch = (batch_size, n_time_steps, size_prediction_window * features)
# Univariate data
data = np.arange(1, 2*(n_time_steps+size_prediction_window)+1).reshape(shape)
print(data)

# %%
Y = np.zeros(shape_Y)
# %%
print("1 - Populating moving window")
for batch_idx in range(batch_size):
    seq = np.array([data[batch_idx][a+1:a+1+size_prediction_window] for a in range(n_time_steps)])
    print(seq.shape)
    Y[batch_idx,:] = seq
print(np.round(Y,2))

 # %%   
Y = np.zeros(shape_Y)

for step_ahead in range(1, size_prediction_window + 1):
    Y[:,:,step_ahead-1] = data[:,step_ahead:step_ahead+n_time_steps,:]
    print("a =", step_ahead)
    print(np.round(Y.reshape(shape_batch),2))
# %%

# TODO
# Input:
# - Multivariate: many time-dependent waste predictors (e.g., waste in each area, features of the areas)
# Output:
# - Multivariate: waste in each area
# - Multi-step: forecasted for T steps ahead