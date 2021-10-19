# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# 

# %%
from operator import index
from types import prepare_class
from pandas.core.frame import DataFrame
from sklearn import linear_model
import housing_util as hu

df = hu.load_housing_data()

df


# %%
df.ocean_proximity.value_counts()

# %% [markdown]
# ## Stratified sampling
# 
# Tthe population is divided into homogeneous subgroups called strata, and the right number of instances are sampled from each stratum to guarantee that the test set is representative of the overall population.
# 
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df["income_cat"] = pd.cut(
    df.median_income,
    bins=[0., 1.5, 3., 4.5, 6., np.inf],
    labels=[1,2,3,4,5]
)

_ = df.income_cat.hist()


# %%
_ = df.hist(bins=50, figsize=(20,15))


# %%
df.describe()


# %%
df.info()


# %%
from zlib import crc32
import numpy as np

def test_set_check(identifier, test_ratio):
        id64 = np.int64(identifier)
        # print(id64)
        checksum = crc32(id64)
        # print(checksum)
        max_test_id = test_ratio * 2**32
        # print(max_test_id)
        masked_checksum32 = checksum & 0xffffffff
        # print(masked_checksum)
        return masked_checksum32 < max_test_id


# %%
def split_train_test_by_id(df, test_ratio, id_column):
    in_test_set = df[id_column].map(lambda id_:test_set_check(id_, test_ratio))
    return df.loc[~in_test_set], df.loc[in_test_set]

# %% [markdown]
# ## Creating ids

# %%
df_id = df.reset_index()
train_set, test_set = split_train_test_by_id(df_id, 0.2, "index")
print(len(train_set), len(test_set), len(df))

# %% [markdown]
# If you use the row index as a unique identifier, you need to make sure that new data gets appended to the end of the dataset and that no row ever gets deleted. If this is not possible, then you can try to use the most stable features to build a unique identifier: 

# %%
df["id"] = df.longitude*1000 + df.latitude
train_set, test_set = split_train_test_by_id(df, 0.2, "id")
print("#train:", len(train_set), "#test:", len(test_set), "#total:", len(df))

# %% [markdown]
# The location information is actually quite coarse, and as a result many districts will have the exact same ID, so they will end up in the same set (test or train). This introduces some unfortunate sampling bias:

# %%
print("N. of ids:", len(set(df["id"])), " - N. of entries:", len(df["id"]))


# %%
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
print("#train:", len(train_set), "#test:", len(test_set), "#total:", len(df))


# %%
train_set.median_income.hist()


# %%
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)
for train_index, test_index in split.split(df, df["income_cat"]):
    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index]


# %%
total_share_cat = df.income_cat.value_counts()/len(df)
strat_share_cat = strat_test_set.income_cat.value_counts()/len(strat_test_set)
random_share_cat = test_set.income_cat.value_counts()/len(test_set)


# %%
shares = np.array([
    total_share_cat.values,
    strat_share_cat.values,
    random_share_cat.values,
    (random_share_cat.values-total_share_cat.values)/total_share_cat.values*100,
    (strat_share_cat.values-total_share_cat.values)/total_share_cat.values*100])

df_sampling_bias_comparison = pd.DataFrame(
    shares.T,
    columns=["Overall", "Stratified", "Random", "Rand. %error", "Strat. %error"],
    index=total_share_cat.index)

df_sampling_bias_comparison.sort_index()

# %% [markdown]
# ## Discover and vizualize the data to gain insights

# %%
df_housing = strat_train_set.copy()
df_housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)


# %%
import folium

def add_circles(point, map, min_pop, max_pop):
    # https://leafletjs.com/reference-1.6.0.html#circle
    folium.Circle(
        radius = (point.population - min_pop)/(max_pop-min_pop)*30,
        weight = 1,
        opacity = 0.4,
        location = [point.latitude, point.longitude],
        color="crimson"
    ).add_to(map)

map = folium.Map(width=600, height=400, zoom_start=2)


#use df.apply(,axis=1) to "iterate" through every row in your dataframe
df_housing.apply(
    add_circles,
    args=(
        map, 
        min(df_housing.population),
        max(df_housing.population)
    ),
    axis=1
)

#Set the zoom to the maximum possible
map.fit_bounds(map.get_bounds())

#Save the map to an HTML file
map.save('html_map_output/housing_scatter.html')

map


# %%
import matplotlib.pyplot as plt
df_housing.plot(
    kind="scatter",
    x="longitude",
    y="latitude",
    figsize=(10,7),
    s=df_housing["population"]/100,
    c=df_housing["median_house_value"],
    cmap=plt.get_cmap("jet"),
    colorbar=True,
    alpha=0.4)

plt.axis()
plt.show()


# %%
corr_matrix = df_housing.corr()
corr_matrix.median_house_value.sort_values(ascending=False)

# %%
from pandas.plotting import scatter_matrix
# Focus on a few promising attributes that seem most correlated with the median house value
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]

# This scatter matrix plots every numerical attribute against every other numerical attribute, plus a histogram of each numerical attribute
_ = scatter_matrix(df_housing[attributes], figsize=(12,8))


# %%
df_housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)

# %% [markdown]
# ## Experimenting with attribute combinations

# %%
df_housing["bedrooms_per_room"] = df_housing.total_bedrooms/df_housing.total_rooms
df_housing["population_per_room"] = df_housing.population/df_housing.total_rooms
df_housing["population_per_household"] = df_housing.population/df_housing.households
df_housing["rooms_per_household"] = df_housing.total_rooms/df_housing.households

df_housing.corr()["median_house_value"]

# %% [markdown]
# The new `bedrooms_per_room` attribute is much more correlated with the median house value than the total number of rooms or bedrooms. Apparently houses with a lower bedroom/room ratio tend to be more expensive.
# 
# The number of rooms per household is also more informative than the total number of rooms in a district — obviously the larger the houses, the more expensive they are.
# %% [markdown]
# ## Prepare the data for ML algorithms

# %%
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()


housing_test = strat_test_set.drop("median_house_value", axis=1)
housing_test_labels = strat_test_set["median_house_value"].copy()

print(len(housing), len(housing_test))
# %% [markdown]
# ## Data cleaning
# 
# Options:
# 
# 1. Get rid of observations featuring missing values:
#    ```python
#    df.dropna(subset=["target_column"])
#    ```
# 2. Get rid of the whole attribute:
#    ```python
#    df.drop("target_column", axis=1)
#    ```
# 3. Set the values for some value (zero, mean, median, etc.):
#    ```python
#    # Remember to save this median value!
#    # You have to use it later to:
#    # - Replace missing values in the test set;
#    # - Replace missing values in new data.
#    median = df["target_column"].median()
#    df["target_column"].fillna(median, inplace=True)
#    ```
# %% [markdown]
# In strategy #3, it is safer to calculate the median for *all* numerical attributes once we cannot be sure that there won't be any missing values in the *new data* after the sytem goes live.

# %%
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")

df_num = housing.drop(["ocean_proximity"], axis=1)

# Save medians for all numerical attributes in inputer.statistics_
imputer.fit(df_num)

# Replace missing values with learned medians
X = imputer.transform(df_num)

# Back to pandas df
df_tr = pd.DataFrame(X, columns=df_num.columns, index=df_num.index)


# %%
with np.printoptions(precision=2, suppress=False):
    print(imputer.statistics_)
    print(df_num.median().values)

# %% [markdown]
# ## Handling text and categorial attributes

# %%
housing_cat = housing[["ocean_proximity"]]
housing_cat.value_counts()

# %% [markdown]
# ### Categorical encoding

# %%
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
print(ordinal_encoder.categories_)
housing_cat_encoded[:10]

# %% [markdown]
# ### One-hot encoding

# %%
from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder()
housing_cat_1hot = onehot_encoder.fit_transform(housing_cat)
print(onehot_encoder.categories_)
housing_cat_1hot[:10]

# %% [markdown]
#  Using up tons of memory mostly to store zeros would be very wasteful, so instead a sparse matrix only stores the location of the nonzero elements. You can use it mostly like a normal 2D array, but if you really want to convert it to a (dense) NumPy array, just call the toarray() method:

# %%
housing_cat_1hot[:10].toarray()

# %% [markdown]
# ## Custom transformers
# %% [markdown]
# More generally, you can add a hyperparameter to gate any data preparation step that you are not 100% sure about. The more you automate these data preparation steps, the more combinations you can automatically try out, making it much more likely that you will find a great combination (and saving you a lot of time).

# %%
from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    
    def fit(self, X, y=None):
        return self  # nothing else to do
    
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]

        else:
            # https://numpy.org/doc/stable/reference/generated/numpy.c_.html
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=True)
housing_extra_attribs = attr_adder.transform(housing.values)

print("HParameters:", attr_adder.get_params())

cols = housing.columns.values
new_cols = np.array(["rooms_per_household", "population_per_household", "bedrooms_per_room"])

pd.DataFrame(housing_extra_attribs, columns=np.concatenate((cols, new_cols)))[:3]

# %% [markdown]
# ## Feature scaling

# %%
cols, rows = 3, 3
fig, ax = plt.subplots(nrows=rows,ncols=cols, figsize=(10,10))

# [0,0] [0,1] [0,2] (0,1,2)
# [1,0] [1,1] [1,2] (3,4,5)
# [2,0] [2,1] [2,2] (6,7,8)

housing_num = housing.drop(["ocean_proximity", "income_cat"], axis=1)
housing_cols = housing_num.columns.values

for i, column in enumerate(housing_cols):
    c = i%cols
    r = int(i/cols)
    housing[[column]].plot(kind="box", ax=ax[r][c])
    ax[r][c].set_title(column)


# %%
from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler_normal = MinMaxScaler()
scaler_std = StandardScaler()

housing_scaled_normal = scaler_normal.fit_transform(housing_num[["population"]])
housing_scaled_std = scaler_std.fit_transform(housing_num[["population"]])

df_feature_scaling = pd.DataFrame(np.c_[housing_scaled_normal, housing_scaled_std], columns=["Normal", "Std"])

df_feature_scaling.describe()


# %%
plt.hist(housing_scaled_normal, bins=50)


# %%
plt.hist(housing_scaled_std, bins=50)


# %%
print(f"Variance: {np.var(housing_scaled_std)} - Mean: {np.mean(housing_scaled_std)}")


# %%
a = np.array([[1,2,3]])
b = np.array([[4,5,6]])
c = 0
np.c_[a, b, c]

# %% [markdown]
# ## Transformation pipelines

# %%

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)


# %%
housing_num_tr

# %% [markdown]
# ### Column transformer

# %%
from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

housing_prepared = full_pipeline.fit_transform(housing)
housing_prepared

# %% [markdown]
# ##### Drop and Passthrough

# %%
data = [["a1", "a2", "a3", "a4"],
    ["b1", "b2", "b3", "b4"],
    [1,2,3,4],
    [1,2,3,4],
    ["cat1", "cat1", "cat2", "cat3"],
    ["f1", "f2", "f3", "f4"]]

df_original = pd.DataFrame(
    np.array(data).T, 
    columns=["A", "B", "C", "D", "E", "F"]
)

print(df_original)

# %%
cat_data = [df_original.E]
print(np.array(cat_data).reshape(-1,1))
print(np.array(cat_data).T)

# %%


print(np.array(cat_data).reshape(-1,1))
print(cat_data)
onehot_enc = OneHotEncoder()
onehot_enc.fit_transform(cat_data)

# %%
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Each transformer must have a unique name (no `__`):
# remove, keep, minmax, standard, 1hot
pipeline_col = ColumnTransformer([
    ("remove", "drop", ["A"]),
    ("keep", "passthrough", ["B"]),
    ("minmax", MinMaxScaler(), ["C"]),
    ("std", StandardScaler(), ["D"]),
    ("1hot", OneHotEncoder(), ["E"])])

df_tr = pipeline_col.fit_transform(df_original)

# Since "F" was not listed, it is dropped by default
print(df_tr)
# pd.DataFrame(df_tr, columns =  ["B", "C(minmax)", "D(std)", "E(1hot)"])


# %%
housing_prepared

# %% [markdown]
# ## Select and train a model
# %% [markdown]
# ### Training and evaluating on the training set
# %% [markdown]
# #### Linear regression

# %%
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)


# %%
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions:", lin_reg.predict(some_data_prepared))
print("     Labels:", np.array(some_labels))


# %%
from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse  = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# %%
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
tree_regression = DecisionTreeRegressor()
tree_regression.fit(housing_prepared, housing_labels)
housing_predictions = tree_regression.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse
# %%

### Cross validation: Decision Tree

from sklearn.model_selection import cross_val_score

def display_scoring(scores):
    scoring_info = (
        f"Scores: {scores}\n"
        f"  Mean: {scores.mean()}\n"
        f"   Std: {scores.std()}")
    return scoring_info


tree_regression = DecisionTreeRegressor()

rmse_scores = cross_val_score(
    tree_regression,
    housing_prepared,
    housing_labels, 
    scoring="neg_mean_squared_error",
    cv=10)
tree_rmse_scores = np.sqrt(-rmse_scores)

print(display_scoring(tree_rmse_scores))



# %%

### Cross validation: Linear Regression

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
lin_model = LinearRegression()
lin_scores = cross_val_score(
    lin_model,
    housing_prepared,
    housing_labels,
    scoring="neg_mean_squared_error",
    cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
print(display_scoring(lin_rmse_scores))


# %%

### Cross validation: Random Forest Regression

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

rforest_regressor = RandomForestRegressor()

rforest_regressor.fit(housing_prepared, housing_labels)
rforest_housing_predictions = rforest_regressor.predict(housing_prepared)
rforest_mse = mean_squared_error(
    housing_labels, 
    rforest_housing_predictions)

rforest_rmse = np.sqrt(rforest_mse)
print(rforest_rmse)

# %%

rforest_scores = cross_val_score(
    rforest_regressor,
    housing_prepared,
    housing_labels,
    scoring="neg_mean_squared_error",
    cv=10)

rforest_rmse_scores = np.sqrt(-rforest_scores)
print(display_scoring(rforest_rmse_scores))

# %%
df_scores = pd.DataFrame(np.c_[lin_rmse_scores, tree_rmse_scores, rforest_rmse_scores],
          columns=["Lin", "Tree", "Forest"])

df_scores.plot()
# %%

import joblib
joblib.dump(lin_model, "models/lin.pkl")
joblib.dump(tree_regression, "models/tree_regression.pkl")
joblib.dump(rforest_scores, "models/rforest.pkl")

#%%[markdown]
# ## Scikit-Learn Grid Search exploration


#%%

# %%[markdown]
# First, let's see how the parameter grid look like:
# %%
from sklearn.model_selection import ParameterGrid

param_grid = [
    {'n_estimators': [3, 10, 30],
     'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False],
     'n_estimators': [3,10],
     'max_features': [2, 3, 4]}
]

list(ParameterGrid(param_grid))

# %%[markdown]
# Now, let's create two grid search objects to try out two cross-validation strategies:
# 1. Random division for train/test sets (labeled `cv`)
# 2. Stratified sampling from train/test sets (labeled `strat`)
# Both will use a Random Forest Regressor estimator since it performed better (see book).
# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Estimator
forest_reg = RandomForestRegressor()

# Generates 5 folds for cross-validation
grid_search_cv = GridSearchCV(
    estimator=forest_reg,
    param_grid=param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    return_train_score=True
)

# %%

# Generates 5 folds for cross-validation (using stratified sampling)
splits = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

# Create new DataFrame out of the prepared data array and 
# reset label indexes. # This is necessary to make sure row indexes
# match each other :`df_housing` belongs to the training set, which was 
# previously sampled.
# Therefore, indices are scattered and not sorted, not matching with the
# sorted indices from the new DataFrame created out of 
# `housing_prepared`.

cv_stratified = splits.split(
    np.zeros(len(df_housing)),
    df_housing.income_cat
)

grid_search_strat = GridSearchCV(
    forest_reg,
    param_grid,
    cv=cv_stratified,
    scoring="neg_mean_squared_error",
    return_train_score=True
)

#%%[markdown]
# As indicated on the GridSearchCV [user guide](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html),
# setting `return_train_score=True`, training scores will also be computed.
# 
# Although computationally expensive — and not strictly required to select the parameters that yield the best generalization performance — 
# computing training scores helps to get insights on how different parameter settings impact the overfitting/underfitting trade-off.
#%%

grid_search_cv.fit(housing_prepared, housing_labels)
best_forest_reg_cv = grid_search_cv.best_estimator_

# %%

grid_search_strat.fit(housing_prepared, housing_labels)
best_forest_reg_strat = grid_search_strat.best_estimator_

# %% [markdown]
# Our grid search is configured as follows:
# Parameter grid: 3*4 + 2*3 = 18 (model hyperparameter combinations)
# Cross-validation folds: 5 (five-fold cross validation)
# Total: 90 rounds of training
# %%
# %%[markdown]
# To get the cross-validation data for both train and test phases, we define the following function:
#%%
def get_train_test_rmse_dfs_from_grid_search(grid_search):
    train_score = {}
    test_score = {}
    for k, v in grid_search.cv_results_.items():
        if k.startswith("split") and k.find("train") >=0:
            train_score[k]=v
        if k.startswith("split") and k.find("test") >=0:
            test_score[k]=v

    df_train = pd.DataFrame.from_dict(train_score)
    df_test = pd.DataFrame.from_dict(test_score)
    df_train = np.sqrt(-df_train)
    df_test = np.sqrt(-df_test)
    
    return df_train, df_test

# %%[markdown]
# And to make it easier to see the difference between training and testing data, we generate the following plot:
#%%
import matplotlib.pyplot as plt
def get_train_test_comparison_plot(df_train, df_test, grid_search):
    
    fig, (ax_train, ax_test) = plt.subplots(2,1, figsize=(20,10), sharex=True)

    df_train.T.boxplot(ax=ax_train)
    ax_train.set_title("Train set")
    ax_train.set_ylabel("RMSE")

    df_test.T.boxplot(ax=ax_test)
    ax_test.set_title("Test set")
    ax_test.set_ylabel("RMSE")
    ax_test.set_xlabel("Hyperparameter combination")

    best_rmse_score = np.sqrt(-grid_search.best_score_)
    annotated_text = (
        f" Score: {best_rmse_score:,.0f}"
        f"\nParams: {grid_search.best_params_}"
    )
    ax_test.annotate(
        annotated_text,
        xy=[grid_search.best_index_+1, best_rmse_score],
        xycoords='data',
        xytext=(10,20), 
        textcoords='offset points',
        size=8,
        color='red',
        arrowprops=dict(
            arrowstyle="->",
            color="red",
            connectionstyle="arc3"))

    plt.xticks(
        ticks=[i for i in range(1,19)],
        labels=[i for i in range(1,19)]
    )

# %%[markdown]
# Now, let's see how the stratified sampling strategy looks like:
# %%

df_train_strat, df_test_strat = get_train_test_rmse_dfs_from_grid_search(grid_search_strat)
get_train_test_comparison_plot(df_train_strat, df_test_strat, grid_search_strat)

# %%

df_train_cv, df_test_cv = get_train_test_rmse_dfs_from_grid_search(grid_search_cv)
get_train_test_comparison_plot(df_train_cv, df_test_cv, grid_search_cv)

# %%
from sklearn.metrics import mean_squared_error

def get_rmse(estimator, X_train, X_test, y_train, y_test):
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    estimator.fit(X_train, y_train)
    scores = estimator.predict(X_test)
    score = mean_squared_error(y_test, scores)
    return np.sqrt(score)

# %%

rmse_cv = get_rmse(
    best_forest_reg_cv,
    housing_prepared,
    housing_prepared,
    housing_labels,
    housing_labels)

# %%

X_test = full_pipeline.transform(housing_test)
 
rmse_strat = get_rmse(
    best_forest_reg_strat,
    housing_prepared,
    X_test,
    housing_labels,
    housing_test_labels)

rmse_cv = get_rmse(
    best_forest_reg_strat,
    housing_prepared,
    X_test,
    housing_labels,
    housing_test_labels)

print(rmse_strat, rmse_cv)
# WHY ARE VALUES DIFFERENT FOR THE SAME ESTIMATOR???
'''
(16512, 17) (4128, 17) (16512,) (4128,)
(16512, 17) (4128, 17) (16512,) (4128,)
46684.56705566629 46423.599782514015
'''
# %%

rmse_strat = get_rmse(
    best_forest_reg_strat,
    housing_prepared,
    housing_prepared,
    housing_labels,
    housing_labels)
# %%
from pprint import pprint
pprint(grid_search.best_params_)
pprint(grid_search.best_estimator_)
pprint(grid_search.best_index_)
pprint(grid_search.best_score_)
pprint(grid_search.cv_results_)

# %%
pprint(grid_search.best_params_)
pprint(grid_search.best_estimator_)
pprint(grid_search.best_index_)
pprint(grid_search.best_score_)

# %%
grid_search.best_estimator_.fit(housing_prepared, housing_labels)
scores = grid_search.best_estimator_.predict(housing_prepared)
score = mean_squared_error(housing_labels, scores)
print(best_rmse_score, np.sqrt(score))
# %%

grid_search.best_estimator_
# %%
grid_search.cv_results_
# %%

# %%

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    housing_prepared,
    housing_labels,test_size=0.2,
    random_state=42)

grid_search.best_estimator_.fit(X_train, y_train)
scores = grid_search.best_estimator_.predict(X_test)
score = mean_squared_error(y_test, scores)
print(best_rmse_score, np.sqrt(score))


# %%

from sklearn.model_selection import StratifiedShuffleSplit

splits = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

df_housing_prepared = pd.DataFrame(housing_prepared)
housing_prepared_labels = housing_labels.reset_index()
for train_index, test_index in splits.split(df_housing_prepared, df_housing["income_cat"]):
    X_train = df_housing_prepared[df_housing_prepared.index.isin(train_index)]
    y_train = housing_prepared_labels[df_housing_prepared.index.isin(train_index)]
    X_test = df_housing_prepared[df_housing_prepared.index.isin(test_index)]
    y_test = housing_prepared_labels[df_housing_prepared.index.isin(test_index)]
 
grid_search.best_estimator_.fit(X_train, y_train)
scores = grid_search.best_estimator_.predict(X_test)
score = mean_squared_error(y_test, scores)
print(best_rmse_score, np.sqrt(score))
 # %%
 
 X_train = full_pipeline.fit_transform(train_set)
 full_pipeline.get_params
 # %%
 train_set.ocean_proximity.value_counts()
 
 # %%
 test_set.ocean_proximity.value_counts()
 
 # %%
 X_test = full_pipeline.fit_transform(test_set)
 X_test.shape
 X_test[0]
 # %%
grid_search.best_estimator_.fit(X_train, y_train)
scores = grid_search.best_estimator_.predict(X_test)
score = mean_squared_error(y_test, scores)
print(best_rmse_score, np.sqrt(score))
# %%
