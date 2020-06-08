import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

data_dir = '/opt/data'
file_name = os.path.join(data_dir, 'ground_water/430154100411801.csv')

with open(file_name, 'r', encoding='utf-8') as file:
    data = file.read()

lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1: -1]

print(header)
print(len(lines))

float_data = np.zeros((len(lines), len(header) - 2))
for i, line in enumerate(lines):
    values = [float(x) for x in [line.split(',')[1], line.split(',')[2], line.split(',')[4]]]
    float_data[i, :] = values

train_count = 876

mean = float_data[:train_count].mean(axis=0)
float_data -= mean
std = float_data[:train_count].std(axis=0)
float_data /= std


def generator(_data, _look_back, _delay, min_index, max_index, shuffle=False, _batch_size=15, _step=1):
    if max_index is None:
        max_index = len(_data) - _delay - 1
    index = min_index + _look_back

    while True:
        if shuffle:
            rows = np.random.randint(min_index, max_index, size=_batch_size)
        else:
            if index + _batch_size >= max_index:
                index = min_index + _look_back
            rows = np.arange(index, min(index + _batch_size, max_index))
            index += len(rows)

        samples = np.zeros((len(rows), _look_back // _step, _data.shape[-1]))
        targets = np.zeros((len(rows),))

        for inner_index, row in enumerate(rows):
            indices = range(rows[inner_index] - _look_back, rows[inner_index], _step)
            samples[inner_index] = _data[indices]
            targets[inner_index] = _data[rows[inner_index] + _delay][1]

        yield samples, targets


look_back = 10
step = 1
delay = 1
batch_size = 16

train_generator = generator(float_data,
                            _look_back=look_back,
                            _delay=delay,
                            min_index=0,
                            max_index=875,
                            shuffle=True,
                            _step=step,
                            _batch_size=batch_size)

val_generator = generator(float_data,
                          _look_back=look_back,
                          _delay=delay,
                          min_index=876,
                          max_index=985,
                          _step=step,
                          _batch_size=batch_size)

test_generator = generator(float_data,
                           _look_back=look_back,
                           _delay=delay,
                           min_index=986,
                           max_index=None,
                           _step=step,
                           _batch_size=batch_size)

val_steps = (985 - 876 - look_back) // batch_size
test_step = (len(float_data) - 986 - look_back) // batch_size

model = Sequential()
model.add(layers.GRU(32, input_shape=(None, float_data.shape[-1])))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_generator,
                              steps_per_epoch=54, epochs=10,
                              validation_data=val_generator, validation_steps=val_steps)

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
