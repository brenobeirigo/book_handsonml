# %%
import numpy as np
from pprint import pprint
batch_size = 3
time_steps = 4
dimensionality = 1
X = np.random.rand(batch_size, time_steps, dimensionality)
print("Shape X:", X.shape)
pprint(X)
#%%
# Since linear models only accept 2d arrays, we flatten the 3d array as follows:
#%%
X2d = X.reshape(batch_size, time_steps*dimensionality)
print("Shape 2d X:", X2d.shape)
pprint(X2d)

# %%
X.reshape(X.)
# %% [markdown]
# In this case, we were assuming a univariate time series (dimensionality = 1), but multivariate time series could also be considered (e.g., each time step is related to a set of features).

#%%
b1, b2, b3 = X
print(b1, b2, b3