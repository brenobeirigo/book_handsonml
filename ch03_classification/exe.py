# %%
from sklearn.compose import ColumnTransformer
import numpy as np
import itertools as it

# %%
count = 20
elements = list(
    map(lambda e:e[0]+str(e[1]),
        it.product(
            ["a", "b", "c", "cat", "f"], range(1,count+1)))
)
elements

# %%
col = np.array(elements).reshape(5,count).T
col
# %%
col = np.insert(col, 2, [range(1, count + 1), range(1,count + 1)], axis=1)
col
# %%
col
# %%
import pandas as pd

X = pd.DataFrame(col, columns=["A", "B", "C", "D", "E", "F", "G"])
X[["C", "D"]] = X[["C", "D"]].astype(np.int8)
X
# %%
X.info()
#%%
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

c = ColumnTransformer(
    [("dropA", "drop", ["A"]),
     ("passB", "passthrough", ["B"]),
     ("normC", MinMaxScaler(), ["C"]),
     ("stdD", StandardScaler(), ["D"]),
     ("1hotE", OneHotEncoder(), ["E"])])

transformed = c.fit_transform(X)

transformed

# %%

