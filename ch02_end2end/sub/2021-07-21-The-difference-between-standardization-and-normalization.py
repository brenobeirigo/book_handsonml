
# ## Feature scaling


# %%
import matplotlib
matplotlib.style.use('ggplot')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import housing_util as hu

housing = hu.load_housing_data()
housing_num = housing.drop(["ocean_proximity"], axis=1)
housing_cols = housing_num.columns.values

# %%
housing_num.info()

# %%
housing_num.describe()
# %%
cols, rows = [int(np.ceil(np.sqrt(housing_num.shape[1])))]*2
fig, ax = plt.subplots(cols,rows, figsize=(3*rows,3*cols))

# [0,0] [0,1] [0,2] (0,1,2)
# [1,0] [1,1] [1,2] (3,4,5)
# [2,0] [2,1] [2,2] (6,7,8)

for i, column in enumerate(housing_cols):
    c = i%cols
    r = int(i/cols)
    housing_num[[column]].plot(kind="box", ax=ax[r][c])

# %%
from sklearn.preprocessing import MinMaxScaler

scaler_normal = MinMaxScaler()
housing_scaled_normal = scaler_normal.fit_transform(housing_num)
df_normalized = pd.DataFrame(housing_scaled_normal, columns=housing_cols)
df_normalized.describe()
# %%
from sklearn.preprocessing import StandardScaler

scaler_std = StandardScaler()
housing_scaled_std = scaler_std.fit_transform(housing_num)

df_standardized = pd.DataFrame(housing_scaled_std, columns=housing_cols)
df_standardized.describe()

# %%[markdown]
# Notice that the resulting distributions will present unit variance:
#%%
np.var(df_standardized)
# %%

fig, ax = plt.subplots(2,1, figsize=(15,10), sharex=True)

ax[0].boxplot(housing_scaled_normal)
ax[0].set_title("Feature scaling - Normalization")

ax[1].boxplot(housing_scaled_std)
ax[1].set_title("Feature scaling - Standardization")

plt.xticks([i for i in range(1,housing_num.shape[1]+1)], housing_num.columns.values)
# %%

# %%

# idx population = 5

df_feature_scaling = pd.DataFrame(
    np.c_[df_normalized.total_bedrooms, df_standardized.total_bedrooms],
    columns=["Normal", "Std"])

df_feature_scaling.describe()

# %%
np.var(df_standardized)

# %%
