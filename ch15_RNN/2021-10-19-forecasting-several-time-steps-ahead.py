from sklearn.metrics import mean_squared_error
from tensorflow import keras
from sklearn.model_selection import train_test_split
from numpy.core.fromnumeric import mean
from timeseries.generate import generate_time_series
# %% [markdown]
# # Multi-step forecasting
# ## Option 1: Use one-step model several times in a row
# - Make the model predict the next value
# - Add value to the inputs (acting as if this predicted value had actually occurred)
# - Use the model again to predict the following value, and so on.

# Problems:
# - The predictions for the next time steps is always more accurate (erros accumulate).
# - Comparison with linear model

# %% [markdown]
# # Chapter 15 - Processing sequences using RNNs and CNNs
# %% [markdown]
# ## Forecasting a time series
# Batch of time series (3D array):
# %%
import matplotlib.pyplot as plt
import numpy as np

n_steps_ahead = 50
total_steps = 200
batch_size = 1000
series = generate_time_series(batch_size, total_steps + n_steps_ahead)
print(series.shape)

# %% [markdown]
# The two first series of the batch:
# %%
_ = plt.plot(series[0], label="Series 1")
_ = plt.plot(series[1], label="Series 2")
plt.title(f"Series (batch size, steps, features) = {series.shape}")
plt.legend()
plt.show()

# %% [markdown]
# ### Splitting the data
# We aim to predict `n_steps_ahead`:
# %%


def get_train_val_test_split(X, Y):

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y,
        train_size=0.8,
        random_state=42)

    X_train, X_valid, Y_train, Y_valid = train_test_split(
        X_train, Y_train,
        train_size=0.9,
        random_state=42)

    return (X_train, X_valid, X_test,
            Y_train, Y_valid, Y_test)


(X_train, X_valid, X_test,
 Y_train, Y_valid, Y_test) = get_train_val_test_split(
    series[:, :total_steps],
    series[:, total_steps:])

print(X_train.shape, X_valid.shape, X_test.shape,
      Y_train.shape, Y_valid.shape, Y_test.shape)


# %%

model = keras.models.Sequential([
    keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.SimpleRNN(20),
    keras.layers.Dense(1)
])

model.compile(loss="mean_squared_error")

# %%

# Fit the model using only the prediction of the next time step
model.fit(
    X_train, Y_train[:, 0],
    epochs=20,
    validation_data=(X_valid, Y_valid[:, 0])
)

y_pred = model.predict(X_test)

# Note that y shape is (batch_size, 1)
print(y_pred.shape)

# %%

# %%
# We can add a new axis to put it back to the original 3D shape using:
# - reshape
# - np.newaxis

# %%
assert np.array_equal(
    y_pred.reshape(len(y_pred), 1, len(y_pred[1])),
    y_pred[:, np.newaxis]
)

# %%
X = X_test.copy()

for step_ahead in range(n_steps_ahead):
    # Shift sequence
    y_pred_one = model.predict(X[:, step_ahead:])
    # Join predicted value to X
    X = np.concatenate([X, y_pred_one[:, np.newaxis]], axis=1)
# %%
y_rnn_plus_next = X[:, -n_steps_ahead:]
print(y_rnn_plus_next.shape, Y_test.shape)
# %%

print("RMSE:", np.sqrt(mean_squared_error(
    np.concatenate(y_rnn_plus_next), np.concatenate(Y_test))))


# %%

lin_model = keras.models.Sequential([
    keras.layers.Flatten(
        input_shape=[total_steps, 1]),
    keras.layers.Dense(n_steps_ahead)
])

lin_model.compile(loss="mean_squared_error")


lin_model.fit(
    X_train, Y_train,
    epochs=20,
    validation_data=(X_valid, Y_valid)
)
# %%

y_linear = lin_model.predict(X_test)


# %%
print("RMSE Linear:", np.sqrt(mean_squared_error(
    np.concatenate(y_linear), np.concatenate(Y_test))))
# %%
# %%[markdown]
# ## Option 2: Predict all next values at once
# Sequence-to-vector RNN: output a vector with next `n_steps_ahead` values at the last time step, i.e., after the complete sequence:

# %%
rnn_s2v_model = keras.models.Sequential([
    keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.SimpleRNN(20),
    keras.layers.Dense(n_steps_ahead)
])

rnn_s2v_model.compile(loss="mean_squared_error")


rnn_s2v_model.fit(
    X_train, Y_train,
    epochs=20,
    validation_data=(X_valid, Y_valid)
)

y_s2v = rnn_s2v_model.predict(X_test)


# %%
print("RMSE s2v RNN:", np.sqrt(mean_squared_error(
    np.concatenate(y_s2v), np.concatenate(Y_test))))
# %% [markdown]

# ## Option 3: Sequence-to-sequence RNN
#
# Sequence-to-sequence RNN: output a vector with next  `n_steps_ahead` values at each time step.
#
# Advantages of S2S over S2V:
# - Many more error gradients flowing through the model = +stability & +training speed:
#    - S2V: Gradients flow only through time;
#    - S2S: Gradients flow through time AND from the output of each time step.

# ### How to adjust the data
# Each target must be a sequence of the same length as the input sequence.
# For example, consider the following univariate time series with 10 time steps and corresponding values:
#
# |    t     |    v     |
# |:--------:|:--------:|
# |    0     |    1     |
# |    1     |    2     |
# |    2     |    3     |
# |    3     |    4     |
# |    4     |    5     |
# |    5     |    6     |
# |    6     |    7     |
# |    7     |    8     |
# |    8     |    9     |
# |    9     |    10    |
#
#
# If the expected number of output time steps is `n_steps_ahead=3`, the training data are framed using an overlapping moving window as follows:
#
# |    t     |   t-1    |    t     |   t+1    |  t+2     |
# |:--------:|:--------:|:--------:|:--------:|:--------:|
# |    0     |  `NaN`   |    1     |    2     |    3     |
# |    1     |    1     |    2     |    3     |    4     |
# |    2     |    2     |    3     |    4     |    5     |
# |    3     |    3     |    4     |    5     |    6     |
# |    4     |    4     |    5     |    6     |    7     |
# |    5     |    5     |    6     |    7     |    8     |
# |    6     |    6     |    7     |    8     |    9     |
# |    7     |    7     |    8     |    9     |    10    |
# |    8     |    8     |    9     |    10    |  `NaN`   |
# |    9     |    9    |     10    |  `NaN`   |  `NaN`   |
#
# The rows with `NaN` can be excluded. Then, we have:
#
#       X_train = [1 2 3 4 5 6]
#       Y_train = [[2 3 4],
#                  [3 4 5],
#                  [4 5 6],
#                  [5 6 7],
#                  [6 7 8],
#                  [7 8 9],
#                  [8 9 10]]
#
# Although there are overlapping values between `X_train` and `Y_train`, at each time step, the model only knows about past time steps — it cannot look ahead (*causal* model).
#

# %%
print(X_train.shape, Y_train.shape, total_steps, n_steps_ahead)
train_batch_size = len(X_train)
v2v_shape = (train_batch_size, total_steps, n_steps_ahead)

# %%
X_train[:, :, :]
# %%
Y_multistep = np.zeros(v2v_shape)
print(Y_multistep.shape)

# Range start from 1 because we want to start the prediction from
# using the first input at the first time step.
#
# In the example below, it corresponds to X=1 and y=[2 3 4]:
#
# |    t     |   t-1    |    t     |   t+1    |  t+2     |
# |:--------:|:--------:|:--------:|:--------:|:--------:|
# |    0     |  `NaN`   |    1     |    2     |    3     |
# |    1     |    1     |    2     |    3     |    4     |
#
for step_ahead in range(1, n_steps_ahead+1):

    # X and y are stacked to create columns with the moving values
    shifted_col = np.concatenate([
        X_train[:, step_ahead:],
        Y_train[:, :step_ahead]], axis=1)

    # Columns are broadcasted to the multistep array
    Y_multistep[:, :, step_ahead-1:] = shifted_col

# %%
print(np.round(Y_multistep[0], 2))
print("First column:")
print(np.round(X_train[0][1:4], 2))
print(np.round(X_train[0][-2:], 2))
print(np.round(Y_train[0][:1], 2))


# %%
v2v_shape = (batch_size, total_steps, n_steps_ahead)
print("Series:", series.shape)
print(v2v_shape)
Y_multistep = np.zeros(v2v_shape)
print(Y_multistep.shape)
# %%
# Range start from 1 because we want to start the prediction from
# using the first input at the first time step.
#
# In the example below, it corresponds to X=1 and y=[2 3 4]:
#
# |    t     |   t-1    |    t     |   t+1    |  t+2     |
# |:--------:|:--------:|:--------:|:--------:|:--------:|
# |    0     |  `NaN`   |    1     |    2     |    3     |
# |    1     |    1     |    2     |    3     |    4     |
#
for step_ahead in range(1, n_steps_ahead+1):
    # X and y are stacked to create columns with the moving values
    shifted_col = series[:, step_ahead: step_ahead + total_steps, :]
    # Columns are broadcasted to the multistep array
    Y_multistep[:, :, step_ahead-1:] = shifted_col

X_train, X_valid, X_test, Y_train, Y_valid, Y_test = get_train_val_test_split(
    series[:, :total_steps, :], Y_multistep)
# %%
# To turn the model into a sequence-to-sequence model, we must set return_sequences=True in all recurrent layers (even the last one), and we must apply the output Dense layer at every time step.
# The TimeDistributed layer:
# - Wraps any layer (e.g., a Dense layer) and
# - Applies the wrapped layer at every time step of its input sequence.
# It does this efficiently, by
# - reshaping the inputs so that each time step is treated as a separate instance:
#     - [batch size, time steps, input dimensions] to [batch size × time steps, input dimensions];
#
# in this example,
# N. of input dimensions = 20 (previous SimpleRNN layer has 20 units)
# then it runs the Dense layer, and finally it reshapes the outputs back to sequences (i.e., it reshapes the outputs from
# Reshaping = [batch size × time steps, output dimensions] to [batch size, time steps, output dimensions];
# N. of output dimensions = 10 (Dense layer has 10 units).


rnn_s2s_model = keras.models.Sequential([
    keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    # To turn the model into a sequence-to-sequence model, we must set return_sequences=True in all recurrent layers
    keras.layers.SimpleRNN(20, return_sequences=True),

    keras.layers.Dense(n_steps_ahead)
    # keras.layers.TimeDistributed(keras.layers.Dense(n_steps_ahead))
])

# %%
# All outputs are needed during training, but only the output at the last time step is useful for predictions and for evaluation.
# So although we will rely on the MSE over all the outputs for training, we will use a custom metric for evaluation, to only compute the MSE over the output at the last time step:


def last_time_step_mse(Y_true, Y_pred):
    return keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])

# Note: Used `sklearn.metrics.mean_squared_error` and got error: NotImplementedError: Cannot convert a symbolic Tensor (strided_slice_1:0) to a numpy array. This error may indicate that you're trying to pass a Tensor to a NumPy call, which is not supported

# Error fix:
# https://github.com/tensorflow/models/issues/9706
# - Go to env/lib/python3.8/site-packages/tensorflow/python/ops/array_ops.py
# - Add `from tensorflow.python.ops.math_ops import reduce_prod`
# - search for `def _constant_if_small` and then replace the entire function to:
# def _constant_if_small(value, shape, dtype, name):
#   try:
#     if reduce_prod(shape) < 1000:
#       return constant(value, shape=shape, dtype=dtype, name=name)
#   except TypeError:
#     # Happens when shape is a Tensor, list with Tensor elements, etc.
#     pass
#   return None


rnn_s2s_model.compile(loss="mse", metrics=[last_time_step_mse])
rnn_s2s_model.summary()
# %%

rnn_s2s_model.fit(
    X_train, Y_train,
    epochs=20,
    validation_data=(X_valid, Y_valid)
)

# %%
y_s2s = rnn_s2s_model.predict(X_test)
print(y_s2s.shape)

# %%
print("RMSE s2s RNN:", np.sqrt(mean_squared_error(
    np.concatenate(y_s2s[:, -1]), np.concatenate(Y_test[:, -1]))))


# %%

batch_id = 4
fig, ax = plt.subplots(figsize=(15, 10))
plt.plot(X_test[batch_id], "b")

x_ticks = range(total_steps, total_steps + n_steps_ahead)

plt.plot(x_ticks, Y_test[batch_id, -1], "b:", label="Actual")
plt.plot(x_ticks, y_linear[batch_id], "g", marker="x", label="Linear")
plt.plot(x_ticks, y_rnn_plus_next[batch_id],
         "r", marker="x", label="RNN(recurring)")
plt.plot(x_ticks, y_s2v[batch_id], "k", label="RNN S2V")
plt.plot(x_ticks, y_s2s[batch_id, -1], "k",  marker="o", label="RNN S2S")
plt.xlim(180)

plt.legend()
plt.show()
# %%

# %%
