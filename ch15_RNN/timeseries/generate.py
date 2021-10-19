import numpy as np

# TODO ankinize forecasting a time series

def generate_time_series(batch_size, n_steps):
    # When dealing with time series (and other types of sequences such as sentences), the input features are generally represented as 3D arrays of shape [batch size, time steps, dimensionality], where dimensionality is 1 for univariate time series and more for multivariate time series.
    # np.random.rand = d0, d1, d2, ..., dn -> dimensions of the returned array
    # e.g., 4 arrays with `batch_size` rows and one col
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
    #print(freq1, freq2, offsets1, offsets2)
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))  #   wave 1
    #print(series)
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20)) # + wave 2
    #print(series)
    # n. of rows = batch_size (number of time series)
    # n. of cols = n_steps (what it wave value for each time series and time step)
    series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5)   # + noise
    
    # what if dimensionality were different?
    # E.g., dimensionality = 3:
    #
    # [[ [v1,v2,v3] [v1,v2,v3] [v1,v2,v3]
    # [v1,v2,v3] [v1,v2,v3] [v1,v2,v3]
    # [v1,v2,v3] [v1,v2,v3] [v1,v2,v3]
    # [v1,v2,v3] [v1,v2,v3] [v1,v2,v3]
    # [v1,v2,v3] [v1,v2,v3] [v1,v2,v3]    
    # ]]
    #print("noise:",series)
    
    # The ellipsis (three dots) indicates "as many ':' as needed". (Its name for use in index-fiddling code is Ellipsis, and it's not numpy-specific.) This makes it easy to manipulate only one dimension of an array, letting numpy do array-wise operations over the "unwanted" dimensions. You can only really have one ellipsis in any given indexing expression, or else the expression would be ambiguous about how many ':' should be put in each. 
    series = series[..., np.newaxis].astype(np.float32)
    #print(series)
    return series