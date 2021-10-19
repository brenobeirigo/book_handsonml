

#%%

from timeseries.generate import generate_time_series

# %%
total_steps = 3
series = generate_time_series(5, total_steps + 1)

# %%

total_steps = 50
batch_size = 10000
dimensionality = 1
series = generate_time_series(batch_size, total_steps + 1)
X_train, y_train = series[:7000, :total_steps], series[:7000, -1]
X_valid, y_valid = series[7000:9000, :total_steps], series[7000:9000, -1]
X_test, y_test = series[9000:, :total_steps], series[9000:, -1]

# %%
X_train.shape, y_train.shape
# Return for each day a multivariate time series. Each time step is associated with the total number of passengers requesting a ride at each zone location.
# Input: bin_size

# %%
X_train
# %% [markdown]
# # Baseline metrics

# ## Naive forecasting

# Simply predict the last value in each series:
# %%
y_pred = X_valid[:, -1]
y_pred.shape
#%%
from sklearn.metrics import mean_squared_error
mean_squared_error(y_valid, y_pred)
# %% [markdown]
# ## Use a fully connected network
# Flatten the input such that the combination of the complete time series is used to make the prediction.
# %%
from sklearn.linear_model import LinearRegression

lin = LinearRegression()
# Flatten n_steps and dimensionality: linear models accept 2d arrays only
X_train2d = X_train.reshape(len(X_train), total_steps * dimensionality)
X_valid2d = X_valid.reshape(len(X_valid), total_steps * dimensionality)

lin.fit(X_train2d, y_train)
y_pred = lin.predict(X_valid2d)
y_pred.shape
# %%

mean_squared_error(y_valid, y_pred)

# %%
import tensorflow as tf
from tensorflow import keras

tf.__version__
# %%
keras.__version__
#%% [markdown]
# ## Using a fully connected network
#%%
lin_model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[total_steps, 1]),
    keras.layers.Dense(1)
])

lin_model.summary()
#%%
lin_model.compile(loss="mean_squared_error",
              optimizer="adam")

# %%
# We pass it the input features (X_train) and the target classes (y_train), as well as the number of epochs to train (or else it would default to just 1, which would definitely not be enough to converge to a good solution). We also pass a validation set (this is optional).
# Keras will measure the loss and the extra metrics on this set at the end of each epoch, which is very useful to see how well the model really performs. If the performance on the training set is much better than on the validation set, your model is probably overfitting the training set (or there is a bug, such as a data mismatch between the training set and the validation set).

# And that’s it! The neural network is trained.15 At each epoch during training, Keras displays the number of instances processed so far (along with a progress bar), the mean training time per sample, and the loss and accuracy (or any other extra metrics you asked for) on both the training set and the validation set. You can see that the training loss went down, which is a good sign, and the validation accuracy reached 89.26% after 30 epochs. That’s not too far from the training accuracy, so there does not seem to be much overfitting going on.

# TIP Instead of passing a validation set using the validation_data argument, you could set validation_split to the ratio of the training set that you want Keras to use for validation. For example, validation_split=0.1 tells Keras to use the last 10% of the data (before shuffling) for validation.

lin_model.fit(
    X_train,
    y_train,
    epochs=20,
    validation_data=(X_valid, y_valid))

# %%
y_pred = lin_model.predict(X_valid)
y_pred.shape
# %%

mean_squared_error(y_valid, y_pred)

# %% [markdown]
# ## Implementing a simple RNN
# Simple RNN (1 unit)
# - Uses the hyperbolic tangent activation function
# - Initial state (h_init) = 0
# - h_init is passed to a single recurrent neuron (unit) + x_0 (value of the 1st time step)
# - The neuron computes a weighted sum of these values and applies the hyperbolic tangent activation function to the result, and this gives the first output, y_0.
# - In a simple RNN, this output is also the new state h_0.
# - This new state is passed to the same recurrent neuron along with the next input value, x_1, and the process is repeated until the last time step. Then the layer just outputs the last value, y_49.
# - All of this is performed simultaneously for every time series. 
#%%
model = keras.models.Sequential([
  keras.layers.SimpleRNN(1, input_shape=[None, 1])
])

model.summary()
# %%


model.compile(
    loss="mean_squared_error",
    optimizer="adam",
    metrics=["accuracy"])

# %%
model.fit(
    X_train,
    y_train,
    epochs=20,
    validation_data=(X_valid, y_valid))

y_pred = model.predict(X_valid)
# %%
np.mean(keras.losses.mean_squared_error(y_valid, y_pred))

# %% [markdown]
## Deep RNNs
# %% [markdown]
# By increasing the number of neurons (1 to 5), the MSE decreased from
# 0.136 to 0.0077.
# %%

# X_train (batch_size, time_steps, n_feature)
# y_train (batch_size, 1) -> Predict last value

# Create a sequential mode with 1 layer (simple RNN with 5 units)
model = keras.models.Sequential([
    keras.layers.SimpleRNN(5, input_shape=[None, 1])
])
model.summary()
# %%
# Compile with MSE loss
model.compile(loss="mean_squared_error")
# %%

# Fit using 20 epochs and validation data
model.fit(
    X_train,
    y_train,
    epochs=20,
    validation_data=(X_valid, y_valid))

# Predict
y_pred = model.predict(X_valid)

# Calculate MSE
np.mean(keras.losses.mean_squared_error(y_valid, y_pred))

#%%
# By default, recurrent layers in Keras only return the final output (e.g., for 50 time steps, y_{49}).
# To make them return one output per time step:
#   - Set `return_sequences=True` for all recurrent layers (except the last one, if you only care about the last output).
# If you don’t, they will output a 2D array (1, features) (containing only the output of the last time step (y_{time_steps-1}) instead of a 3D array (1, time_steps, features) (containing outputs for all time steps), and the next recurrent layer will complain that you are not feeding it sequences in the expected 3D format.
model = keras.models.Sequential([
    # 20 units generate `time_steps` h's = 20 x time_steps
    keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    # 20 units generate `time_steps` h's = 20 x time_steps 
    keras.layers.SimpleRNN(20, return_sequences=True),
    # 1 unit (univariate time series -> 1 output value per time step)
    keras.layers.SimpleRNN(1)
])

model.summary()

#%%
""" 

Model: "sequential_7"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
simple_rnn_13 (SimpleRNN)    (None, 1, 20)             440       -> 20x20 + 20 + 20
_________________________________________________________________
simple_rnn_14 (SimpleRNN)    (None, 1, 20)             820       
_________________________________________________________________
simple_rnn_15 (SimpleRNN)    (None, 1)                 22        
=================================================================
Total params: 1,282
Trainable params: 1,282
Non-trainable params: 0

m = 20 (hidden units)
x = (50, 1)
Input: x \in R
Hidden unit: h \in R^20

# Layer 1 (Total = 440)
- Weights for input units: w_x \in R^20 
- Weights for hidden units: w_h \in R^(20x20)
- Bias for hidden units: b_h \in R^20
Total = 440

# Layer 1 (Total = 820)
- Weights for input units: 20x20 (h's from the previous?)
- Weights for hidden units: w_h \in R^(20x20)
- Bias for hidden units: b_h \in R^20
Total = 440

# Layer 2
- Weights for input units: w_x \in R^20 
- Weights for hidden units: w_h \in R^(1x1)
- Bias for hidden units: b_h \in R^1

Weight for input units: 
Bia
Weight for the dense layer: 
Bias for the dense layer: 
"""

# %%
model.compile(loss="mean_squared_error")

# Fit using 20 epochs and validation data
model.fit(
    X_train,
    y_train,
    epochs=20,
    validation_data=(X_valid, y_valid))

# Predict
y_pred = model.predict(X_valid)

# Calculate MSE
np.mean(keras.losses.mean_squared_error(y_valid, y_pred))
# 0.002758452

# %% [markdown]
# The last layer is not ideal: it must have a single unit because we want to forecast a univariate time series, and this means we must have a single output value per time step.

# Why having a single unit is not 
# However, having a single unit means that the hidden state is just a single number.
# That’s really not much, and it’s probably not that useful; presumably, the RNN will mostly use the hidden states of the other recurrent layers to carry over all the information it needs from time step to time step, and it will not use the final layer’s hidden state very much.
# Moreover, since a SimpleRNN layer uses the tanh activation function by default, the predicted values must lie within the range –1 to 1. 
# 
# %%
# But what if you want to use another activation function? For both these reasons, it might be preferable to replace the output layer with a Dense layer: it would run slightly faster, the accuracy would be roughly the same, and it would allow us to choose any output activation function we want. 
# 
# If you make this change, also make sure to remove return_sequences=True from the second (now last) recurrent layer:
# %%
model = keras.models.Sequential([
    keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    # Last recurrent layer with only the 20 values
    # coming from the units (no sequences)
    keras.layers.SimpleRNN(20),
    # Output layer is a dense layer:
    # - sligtly faster
    # - comparable accuracy
    # - allow different activation function
    #   (SimpleRNN uses tanh, therefore values in range -1—1)
    keras.layers.Dense(1)
])

model.compile(loss="mean_squared_error")

model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))

y_pred = model.predict(X_valid)

np.mean(keras.losses.mean_square_error(y_valid, y_pred))
# %%
from sklearn.metrics import mean_squared_error
mean_squared_error(y_valid, y_pred)

# 0.0030976671

# %% [markdown]

# %%
# series = generate_time_series(1, n_steps + 10)
# X_new, Y_new = series[:, :n_steps], series[:, n_steps:]
# X = X_new
# for step_ahead in range(10):
#     y_pred_one = model.predict(X[:, step_ahead:])[:, np.newaxis, :]
#     X = np.concatenate([X, y_pred_one], axis=1)

# Y_pred = X[:, n_steps:]
# # %%

# %%
