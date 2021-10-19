import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib
matplotlib.style.use('ggplot')

# %%
data = pd.read_csv("median_housing_values_and_predictions.csv")
data.head(5)

# %%
sns.boxplot(data=data, showfliers=True, showmeans=True)
# %%
sns.histplot(data=data)
plt.axvline(data["target"].mean(), color="red")

# %% [markdown]
# ## How precise is this estimate?
# Our sample to calculate the confidence interval will be the squared errors:
 # %%

# WHY NOT ROOTED??? ***
sample =  (data["predicted"] - data["target"])**2
sns.boxplot(data=sample, showfliers=False, showmeans=True)

# %%

sns.displot(data=sample)
# %% [markdown]
# The MSE is the mean ${\textstyle \left({\frac {1}{n}}\sum _{i=1}^{n}\right)}$ of the squared errors ${\displaystyle \left(Y_{i}-{\hat {Y_{i}}}\right)^{2}}$.
# %%%

mse = sample.mean()
print("MSE:", mse)

# Your predictions are off by
rmse = np.sqrt(mse)
print("RMSE:", rmse, "(generalization error)")

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

print("Sample standard error:", sample_std_error, "(SEM = ", stats.sem(sample), ")")

z_value_confidence_level = {
    80:	1.28,
    90:	1.645,
    95:	1.96,
    98:	2.33,
    99:	2.58,
}

confidence = 95
z_value = z_value_confidence_level[confidence]

margin_of_error = z_value*sample_std_error
print("Sample margin of error:", margin_of_error)

confidence_interval = [
    sample_mean - margin_of_error,
    sample_mean + margin_of_error]

confidence_interval_rooted = np.sqrt(confidence_interval)
print("Sample CI:", confidence_interval)
print("Sample CI (rooted):", confidence_interval)

stats.t.interval(confidence, len(sample) - 1,
                          loc=sample.mean(),
                          scale=stats.sem(sample))

# %%
plt.hist(sample, bins=100)
plt.yscale("log")
plt.axvline(np.mean(sample), color="blue")


# %% [markdown]
# As the size of the sample data grows larger, the SEM decreases versus the SD; hence, as the sample size increases, the sample mean estimates the true mean of the population with greater precision. In contrast, increasing the sample size does not make the SD necessarily larger or smaller, it just becomes a more accurate estimate of the population SD.
# %%
increasing_samples = [np.random.choice(sample, s) for s in range(len(sample))]
fig, ax = plt.subplots(figsize=(20,3))
plt.scatter(list(range(len(sample))), [np.std(s) for s in increasing_samples])
plt.scatter(list(range(len(sample))), [stats.sem(s) for s in increasing_samples])
plt.yscale("log")
# %%
sample_size = 600
n_samples = 40000
sub_samples = [np.random.choice(sample, sample_size) for _ in range(n_samples)]
means = [np.mean(s) for s in sub_samples]
true_mean = np.mean(sample)
plt.hist(means, bins=1000)
plt.axvline(true_mean, color="blue")
plt.text(true_mean, 1100, "Population mean")

for n_std in range(1,4):
    plt.axvline(true_mean + n_std*stats.sem(sample), linewidth=0.5, color="black", linestyle="--")
    plt.axvline(true_mean - n_std*stats.sem(sample), linewidth=0.5, color="black", linestyle="--")
# %%
normal_distrib = np.random.normal(sample_mean, sample_std, n)
error_normal_distrib = np.random.normal(sample_mean, sample_std_error, n)
# %%
sns.kdeplot(error_normal_distrib)
sns.kdeplot(sample)
# sns.kdeplot(normal_distrib)

for n_std in range(1,4):
    plt.axvline(sample_mean + n_std*sample_std_error, linewidth=0.5, color="black", linestyle="--")
    plt.axvline(sample_mean - n_std*sample_std_error, linewidth=0.5, color="black", linestyle="--")
# %%

# %

# %%
normal_distrib = np.random.normal(sample_mean, sample_std, n)
sns.kdeplot(normal_distrib)
sns.kdeplot(sample)
plt.axvline(sample_mean, linewidth=0.5, color="red", linestyle="--")
for n_std in range(1,4):
    plt.axvline(sample_mean + n_std*sample_std, linewidth=0.5, color="black", linestyle="--")
    plt.axvline(sample_mean - n_std*sample_std, linewidth=0.5, color="black", linestyle="--")
# %%
rooted_sample = sample
mean = np.mean(rooted_sample)
std = np.std(rooted_sample)
normal_distrib = np.random.normal(mean,  std, n)
print("Mean:", mean, " - Std:", std)

sns.kdeplot(normal_distrib)
# sns.kdeplot(rooted_sample)
# plt.legend(labels=["Normal", "Sample"])
plt.axvline(mean, linewidth=0.5, color="red", linestyle="--")
for n_std in range(1,4):
    r_ci = mean + n_std*std
    l_ci = mean - n_std*std
    
    print(n_std, l_ci, r_ci)
    plt.axvline(r_ci, linewidth=0.5, color="black", linestyle="--")
    plt.axvline(l_ci, linewidth=0.5, color="black", linestyle="--")
    
    print(s)
    s = stats.sem(normal_distrib)
    r_se = mean + n_std*s
    l_se = mean - n_std*s
    print(n_std, l_se, r_se)
    
    #plt.axvline(r_se, linewidth=0.5, color="blue", linestyle="--")
    #plt.axvline(l_se, linewidth=0.5, color="blue", linestyle="--")

# %%
sns.kdeplot(data=(data["predicted"] - data["target"]))
plt.title("Errors")
# %%
root_squared_errors = np.sqrt((data["predicted"] - data["target"])**2)
sns.kdeplot(data=root_squared_errors)
plt.axvline(root_squared_errors.mean())
plt.axvline(confidence_interval[0])
plt.axvline(confidence_interval[1])


# %%
mask = np.logical_and(
    sample.values > confidence_interval[0],
    sample.values < confidence_interval[1])
# %%
print(len(sample.values[mask])/len(sample.values))
sns.histplot(data=sample.values[mask])
plt.axvline(sample_mean)
# %%
# You can guarantee (with 95% of confidence) that errors fall between
print("Sample CI squared (because sample was squared):", np.sqrt(confidence_interval))
# https://www.dummies.com/education/math/statistics/how-to-calculate-a-confidence-interval-for-a-population-mean-when-you-know-its-standard-deviation/
# https://en.wikipedia.org/wiki/Standard_score
# https://statisticsbyjim.com/hypothesis-testing/hypothesis-tests-confidence-intervals-levels/
# https://dfrieds.com/math/confidence-intervals.html (CHECK THIS!!!)
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1255808/ (BEST FOR SE & SD)
# https://machinelearningmastery.com/confidence-intervals-for-machine-learning/
# %%

sample = np.random.choice(rooted_sample, 10000)
len(sample[(sample < confidence_interval[0]) | (sample > confidence_interval[1])])
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
