# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# 

# %%
from math import sqrt
from numpy.core.numeric import full
from scipy.sparse.construct import rand
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
import matplotlib
matplotlib.style.use('ggplot')

df["income_cat"] = pd.cut(
    df.median_income,
    bins=[0., 1.5, 3., 4.5, 6., np.inf],
    labels=[1,2,3,4,5]
)

houses_per_income_cat = df.income_cat.value_counts().sort_index()

houses_per_income_cat.plot(kind="bar", color='crimson', rot=0)
# _ = df.income_cat.hist()

plt.xlabel("Income category")
plt.ylabel("#Houses")


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
diff_sampling = (
    strat_test_set.income_cat.value_counts()
    - test_set.income_cat.value_counts()).sort_index()

diff_sampling.plot(kind='bar', rot=0)
plt.title("")
#sns.histplot(data=strat_test_set, x='income_cat', label='Stratified sampling')
#sns.histplot(data=test_set, x='income_cat', label='Standard sampling')
#plt.legend()
#plt.show()

# %%
total_share_cat = df.income_cat.value_counts()/len(df)
strat_share_cat = strat_test_set.income_cat.value_counts()/len(strat_test_set)
standard_share_cat = test_set.income_cat.value_counts()/len(test_set)

percentage_error_standard = 100*(
    standard_share_cat.values
    - total_share_cat.values
    )/total_share_cat.values

percentage_error_strat = 100*(
    strat_share_cat.values
    - total_share_cat.values
    )/total_share_cat.values

# %%
shares = np.array([
    total_share_cat.values,
    strat_share_cat.values,
    standard_share_cat.values,
    abs(percentage_error_standard),
    abs(percentage_error_strat)])

df_sampling_bias_comparison = pd.DataFrame(
    shares.T,
    columns=[
        "Overall",
        "Stratified",
        "Standard",
        "Abs. stand. %error",
        "Abs. strat. %error"],
    index=total_share_cat.index)

df_sampling_bias_comparison.sort_index()

df_housing = strat_train_set.copy()
# %%[markdown]
# Notice that stratified sampling better represents the original data distribution according with `income category`. 
# %%

df_sampling_bias_comparison[
    ["Abs. stand. %error",
     "Abs. strat. %error"]].plot(kind="bar", rot=0)

plt.title("Stratified vs. standard sampling")
plt.ylabel("Absolute error (%)")
plt.xlabel("Income category")
plt.legend(["Standard", "Stratified"]);

# %% [markdown]
# ## Discover and vizualize the data to gain insights

# %%
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

housing_test = strat_test_set.drop("median_house_value", axis=1)
housing_test_labels = strat_test_set["median_house_value"].copy()

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
new_cols = np.array(
    ["rooms_per_household",
     "population_per_household",
     "bedrooms_per_room"])

pd.DataFrame(
    housing_extra_attribs,
    columns=np.concatenate((cols, new_cols)))[:3]

# %%

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

housing_num = housing.drop(["ocean_proximity", "income_cat"], axis=1)
housing_num_tr = num_pipeline.fit_transform(housing_num)
housing_num_tr

# %%
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

housing_prepared = full_pipeline.fit_transform(housing)
housing_prepared


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
forest_reg = RandomForestRegressor(random_state=42)
# Random Forests includes randomization; for stability we set 
# `random_state`. Whithout setting random_state, consecutive runs
# (i.e., fit, predict) for the same model will result in different
# scores.

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

best_forest_reg_cv
# %%

grid_search_strat.fit(housing_prepared, housing_labels)
best_forest_reg_strat = grid_search_strat.best_estimator_
best_forest_reg_strat

# %% [markdown]
# Our grid search is configured as follows:
# Parameter grid: 3\*4 + 2\*3 = 18 (model hyperparameter combinations)
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
    ax_test.set_title("Validation set")
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

rmse_strat = get_rmse(
    best_forest_reg_strat,
    housing_prepared,
    housing_prepared,
    housing_labels,
    housing_labels)

print("TRAIN - RMSE stratified:", rmse_strat, "- RMSE standard:", rmse_cv)

# %%

def print_pipeline_transformers_fit(pipeline):

    # with np.printoptions(formatter={'all':lambda x:f"{x:.1f}"}):
    # with np.printoptions(precision=1):
    #with np.printoptions(precision=1, threshold=5, suppress=True):
    with np.printoptions(precision=1, threshold=5):
            
        # Medians from SimpleImputer    
        print(pipeline
        .named_transformers_["num"]
        .named_steps["imputer"]
        .statistics_)
        
        # Means from standardization
        print(pipeline
        .named_transformers_["num"]
        .named_steps["std_scaler"]
        .mean_)

print("Fitting to TRAINING data and transforming...")
X_train = full_pipeline.fit_transform(housing)
print_pipeline_transformers_fit(full_pipeline)

print("Only transforming TESTING data...")
X_test = full_pipeline.transform(housing_test)
print_pipeline_transformers_fit(full_pipeline)

# print("WRONG! Fitting to TESTING data and transforming...")
# X_test = full_pipeline.fit_transform(housing_test)
# print_pipeline_transformers_fit(full_pipeline)


rmse_strat = get_rmse(
    best_forest_reg_strat,
    housing_prepared,
    X_test,
    housing_labels,
    housing_test_labels)

rmse_cv = get_rmse(
    best_forest_reg_cv,
    housing_prepared,
    X_test,
    housing_labels,
    housing_test_labels)


print("TEST - RMSE stratified:", rmse_strat, "- RMSE standard:", rmse_cv)


# %%
from scipy import stats

housing_predicted_labels = best_forest_reg_strat.predict(X_test)

# %%
df_target_predicted = pd.DataFrame(
    np.c_[housing_test_labels.values,
          np.round(housing_predicted_labels,1)],
    columns=["target", "predicted"]
)                       

df_target_predicted.to_csv("median_housing_values_and_predictions.csv", index=False)
# %%
plt.boxplot([housing_test_labels, housing_predicted_labels], labels=["Target values","Predicted values"])
plt.axhline(housing_test_labels.mean())

# %%
import seaborn as sns

sns.histplot(data={"Target": housing_test_labels, "Predicted": housing_predicted_labels})
plt.axvline(housing_test_labels.mean())

# %%


squared_errors = (housing_test_labels - housing_predicted_labels)**2

mse = squared_errors.mean()
print("MSE:", mse)

# Your predictions are off by
rmse = np.sqrt(mse)
print("RMSE:", rmse, "(generalization error)")

# How precise is this estimate?
sample = squared_errors
n = len(sample)
print("Sample size:", n)

sample_mean = sample.mean()
print("Sample mean:", sample_mean)

sample_variance = (
    sum((sample-sample_mean)**2)
    /(n-1))
print("Sample variance:", sample_variance)

sample_std = np.sqrt(sample_variance)
print("Sample standard deviation:", sample_std)

sample_std_error = (
    sample_std
    /np.sqrt(n))

print("Sample standard error:", sample_std_error)

z_value_confidence_level = {
    80:	1.28,
    90:	1.645,
    95:	1.96,
    98:	2.33,
    99:	2.58,
}

z_value = z_value_confidence_level[95]

margin_of_error = z_value*sample_std_error
print("Sample margin of error:", margin_of_error)

confidence_interval = [
    sample_mean - margin_of_error,
    sample_mean + margin_of_error]
print("Sample CI:", confidence_interval)

# %%
sns.boxplot(sample, showfliers=False, showmeans=True)
plt.axvline(confidence_interval[0])
plt.axvline(confidence_interval[1])
plt.axvline(sample_mean)

# %%
mask = np.logical_and(
    sample.values > confidence_interval[0],
    sample.values < confidence_interval[1])
# %%
print(len(sample.values[mask])/len(sample.values))
sns.histplot(sample.values[mask])
plt.axvline(sample_mean)
# %%
# You can guarantee (with 95% of confidence) that errors fall between
print("Sample CI squared (because sample was squared):", np.sqrt(confidence_interval))
# https://www.dummies.com/education/math/statistics/how-to-calculate-a-confidence-interval-for-a-population-mean-when-you-know-its-standard-deviation/
# https://en.wikipedia.org/wiki/Standard_score
# https://statisticsbyjim.com/hypothesis-testing/hypothesis-tests-confidence-intervals-levels/

# %%
sns.boxplot(sample)
# %%

print(squared_errors)
confidence = 0.95

print(stats.t.interval(
    confidence,
    len(squared_errors) - 1,
    loc=sample_mean,
    scale=std_error)) # Std. error from the mean (SEM)

# https://www.investopedia.com/ask/answers/042415/what-difference-between-standard-error-means-and-standard-deviation.asp

# %%
best_forest_reg_strat.feature_importances_

# %%
full_pipeline.named_transformers_
# %%
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_encoder.categories_

# %%
new_cols
# %%
cat_one_hot_attributes = cat_encoder.categories_[0]
cat_one_hot_attributes
# %%
np.set_printoptions(precision=3)
attributes = num_attribs + list(new_cols) + list(cat_one_hot_attributes)
attributes

# %%

best_features = zip(attributes, best_forest_reg_cv.feature_importances_)
list(sorted(best_features, key=lambda x:x[1], reverse=True))
# %%
rmse_strat = get_rmse(
    best_forest_reg_strat,
    housing_prepared,
    housing_prepared,
    housing_labels,
    housing_labels)

# %%
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

r_search = RandomizedSearchCV(
    RandomForestRegressor(),
    random_grid,
    n_iter=10,
    scoring="neg_mean_squared_error",
    refit=True,
    random_state=42,
    cv=3
)

r_search.fit(housing_prepared, housing_labels)

# %%

for best, params in zip(r_search.cv_results_["mean_test_score"],
                        r_search.cv_results_["params"]):
    print(best, params)


rmse = get_rmse(
    r_search.best_estimator_,
    housing_prepared,
    X_test,
    housing_labels,
    housing_test_labels)

print("TEST - Random Search RMSE:", rmse, r_search.best_params_)

# %%
# Random search Linear Regression
from sklearn.linear_model import LinearRegression

lin_reg_model = LinearRegression()
print(lin_reg_model.get_params())

# %%
from sklearn.model_selection import RandomizedSearchCV

rand_search = RandomizedSearchCV(
    lin_reg_model,
    param_grid, 
    cv=5,
    random_state=42,
    scoring="neg_mean_square_error")

rand_search.fit(housing_prepared, housing_labels)

rand_search.best_estimator_
# Analyze the best models


# %%
full_pipeline.named_transformers_

