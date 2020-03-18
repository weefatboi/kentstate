
import keras
keras.__version__


import os

data_dir = 'C:/Users/Owner/Desktop/cloudbox/forecasting/jena_climate'
fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')

f = open(fname)
data = f.read()
f.close()

lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]

print(header)
print(len(lines))


import numpy as np

float_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i, :] = values


mean = float_data[:200000].mean(axis=0)
float_data -= mean
std = float_data[:200000].std(axis=0)
float_data /= std


def generator(data, lookback, delay, min_index, max_index,
              shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows),
                           lookback // step,
                           data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets


lookback = 1440
step = 6
delay = 144
batch_size = 128


train_gen = generator(float_data,
                      lookback=lookback,
                      delay=delay,
                      min_index=100001,
                      max_index=200000,
                      shuffle=True,
                      step=step, 
                      batch_size=batch_size)
val_gen = generator(float_data,
                    lookback=lookback,
                    delay=delay,
                    min_index=200001,
                    max_index=300000,
                    step=step,
                    batch_size=batch_size)
test_gen = generator(float_data,
                     lookback=lookback,
                     delay=delay,
                     min_index=300001,
                     max_index=None,
                     step=step,
                     batch_size=batch_size)


val_steps = (300000 - 200001 - lookback) // batch_size

test_steps = (len(float_data) - 300001 - lookback) // batch_size


def evaluate_naive_method():
    batch_maes = []
    for step in range(val_steps):
        samples, targets = next(val_gen)
        preds = samples[:, -1, 1]
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)
    print(np.mean(batch_maes))
    
evaluate_naive_method()



from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model = Sequential()
model.add(layers.GRU(128,
                     dropout=0.1,
                     recurrent_dropout=0.5,
                     return_sequences=True,
                     input_shape=(None, float_data.shape[-1])))
model.add(layers.GRU(128, activation='relu',
                     dropout=0.1, 
                     recurrent_dropout=0.5))
model.add(layers.Dense(1))

# Add batch normalization in combination with output dense layer

dense_model.add(layers.BatchNormalization())

# Add Callbacks list
## stop model after more than 2 epochs of no accuracy improvement
## save best model in terms of validation loss
## send model logs to jena_output folder for exploration with localhost tensorboard command from conda shell

callbacks_list = [
  keras.callbacks.EarlyStopping(
    monitor='acc',
    patience=2,
  ),
  keras.callbacks.ModelCheckpoint(
    filepath='best_forecasting.h5',
    monitor='val_loss',
    save_best_only=True,
  )
  keras.callbacks.TensorBoard(
    log_dir='jena_output',
    histogram_freq=1,
    embeddings_freq=1,
  )
]

# Opted not to include reduce learning rate on plateau callback because it
# might be negated by the previous callback that stops and saves the model after
# validation loss stops improving.
# used a learning rate of 0.015 in the RMSprop call with a decaying rate of 1e-5 over the course of the run
# to reduce learning rate over time

# add accuracy metrics to compile function

model.compile(optimizer=RMSprop(learning_rate=0.015, decay=1e-5), loss='mae', metrics=['acc'])

# pass the callbacks list to the fit function

history = model.fit_generator(train_gen,
                              steps_per_epoch=500,
                              epochs=30,
                              batch_size=128,
                              callbacks=callbacks_list,
                              validation_data=val_gen,
                              validation_steps=val_steps)


# Results:

import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()








