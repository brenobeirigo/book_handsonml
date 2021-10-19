from timeseries.generate import generate_time_series
# %% [markdown]
# # Forecasting several time steps ahead
# ## Option 1: Use one-step model several times in a row
# - Make the model predict the next value
# - Add value to the inputs (acting as if this predicted value had actually occurred)
# - Use the model again to predict the following value, and so on.

### Problems:
# - The predictions for the next time steps is always more accurate (erros accumulate).
# - Comparison with linear model
# %%

# %%

""" 

series = [
[1 2 3 | 6 7],
[1 2 3 | 6 7],
]

X_new = [
[1 2 3],
[1 2 3],    
]

y_new = [
    [6 7],
    [6 7]
]

# step_ahead = 0
X = [
[1 2 3],
[1 2 3],    
]

y_pred_one = [
    [p11],
    [p12]
]

# step_ahead = 1
X = [
[2 3 p11],
[2 3 p12],    
]

y_pred_one = [
    [p21],
    [p22]
]

X = [
[ 3 p11 p22],
[ 3 p12 p22],    
]


"""
# %%
import matplotlib.pyplot as plt
import numpy as np
n_steps_ahead = 50
total_steps = 200
series = generate_time_series(1000, total_steps + n_steps_ahead)
_ = plt.plot(series[0], label="Series 1")
_ = plt.plot(series[1], label="Series 2")
plt.legend()
plt.title(f"Series (batch size, steps, features) = {series.shape}")
plt.show()
print(series.shape)

# %%

from tensorflow import keras

model = keras.models.Sequential([
    keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.SimpleRNN(20),
    keras.layers.Dense(1)
])

model.compile(loss="mean_squared_error")



# %% 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    series[:, :total_steps], series[:, total_steps:],
    train_size=0.8,
    random_state=42)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_train, y_train,
    train_size=0.8,
    random_state=42)

X_train.shape, X_valid.shape, X_test.shape, y_train.shape, y_valid.shape, y_test.shape
# %%

# Fit the model using only the prediction of the next time step
model.fit(
    X_train, y_train[:, 0],
    epochs=20,
    validation_data=(X_valid, y_valid[:, 0])
)

y_pred = model.predict(X_test)

# Note that y shape is (batch_size, 1)
print(y_pred.shape)

# %%

# %%
# We can add a new axis to put it back to the original 3d shape using:
# - reshape
# - np.newaxis

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
y_pred_sequence = X[:, -n_steps_ahead:]
print(y_pred_sequence.shape, y_test.shape)
# %%
from sklearn.metrics import mean_squared_error

print("RMSE:", np.sqrt(mean_squared_error(np.concatenate(y_pred_sequence), np.concatenate(y_test))))


# %%

lin_model = keras.models.Sequential([
    keras.layers.Flatten(
        input_shape=[total_steps, 1]),
    keras.layers.Dense(n_steps_ahead)
])

lin_model.compile(loss="mean_squared_error")


lin_model.fit(
    X_train, y_train,
    epochs=20,
    validation_data=(X_valid, y_valid)
)
# %%

lin_y_pred = lin_model.predict(X_test)


# %%
print("RMSE Linear:", np.sqrt(mean_squared_error(np.concatenate(lin_y_pred), np.concatenate(y_test))))
# %%

# %%
rnn_s2v_model = keras.models.Sequential([
    keras.layers.SimpleRNN(20, return_sequences=True,input_shape=[None, 1]),
    keras.layers.SimpleRNN(20),
    keras.layers.Dense(n_steps_ahead)
])

rnn_s2v_model.compile(loss="mean_squared_error")


rnn_s2v_model.fit(
    X_train, y_train,
    epochs=20,
    validation_data=(X_valid, y_valid)
)

rnn_s2v_y_pred = rnn_s2v_model.predict(X_test)


# %%
print("RMSE s2v RNN:", np.sqrt(mean_squared_error(np.concatenate(rnn_s2v_y_pred), np.concatenate(y_test))))
# %% [markdown]

# ## Forecasting steps ahead every time step

# Sequence-to-vector RNN into sequence-to-sequence RNN
# - Loss has a ter for the output of the RNN at each and every time step (not just the output at tlast time step)
# - Many more error gradients flowing through the model = +stability & +training speed:
#    - Before: Gradients flow only through time;
#    - With S2S: Gradients flow through time AND from the output of each time step.

# ### How to adjust the data
# Each target must be a sequence of the same length as the input sequence.
# E.g. first batch:
# ahead = 3
# n_steps = 7 (predict last 3)
# t =  0 1         6        
# X = [1 2 3 4 5 6 7 | 8 9 10]
#| Timestep | Output   |
#|:--------:|:--------:|
#|  $t_0$   | [2 3 4]  |
#|  $t_1$   | [3 4 5]  |
#|  $t_2$   | [4 5 6]  |
#|  $t_3$   | [5 6 7]  |
#|  $t_4$   | [6 7 8]  |
#|  $t_5$   | [7 8 9]  |
#|  $t_6$   | [8 9 10] |
#
# Although there are overlapping values between X_train and Y_train, at each time step, the model only knows about past time steps (it cannot look ahead)
# Y.shape (batch_size, n_steps, ahead)
# Y = [[2 3 4]]

#%%
import numpy as np
batch_s = 2
ahead = 2
n_steps = 3
s = np.arange(1,2*(n_steps+ahead)+1).reshape((batch_s, n_steps + ahead, 1))
s

# %%
Y = np.zeros((batch_s, n_steps, ahead, 1))
print(Y.shape)
np.round(Y,2)
# %%
for b in range(batch_s):
    seq = np.array([s[b][a+1:a+1+ahead] for a in range(n_steps)])
    Y[b,:] = seq
print(np.round(Y,2))

 # %%   
Y = np.zeros((batch_s, n_steps, ahead, 1))
print(Y.shape)
np.round(Y,2)

for a in range(1, ahead + 1):
    Y[:,:,a-1] = s[:,a:a+n_steps,:]
    print("a =", a)
    print(np.round(Y.reshape((batch_s, n_steps, ahead)),2))
# %%
# %%
plt.plot(X_test[0], "b")

x_ticks = range(total_steps, total_steps + n_steps_ahead)

plt.plot(x_ticks, y_test[0], "b:", label="Actual")

# RNN
plt.plot(x_ticks, y_pred_sequence[0], "r", marker="x", label="Forecast RNN")

# Linear
plt.plot(x_ticks, lin_y_pred[0], "g", marker="x", label="Forecast Linear")


# Linear
plt.plot(x_ticks, rnn_s2v_y_pred[0], "k", label="Forecast RNN S2V")

plt.xlim(180)

plt.legend()
plt.show()
# %%