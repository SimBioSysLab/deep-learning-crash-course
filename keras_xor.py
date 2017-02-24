#!/usr/bin/env python
"""Example of building a model to solve an XOR problem in Keras.

Running this example:
  pip install keras
  python keras_xor.py
"""

from __future__ import print_function

import keras
import numpy as np

# XOR data.
x = np.array([
        [0, 1],
        [1, 0],
        [0, 0],
        [1, 1],
    ])
y = np.array([
        [1],
        [1],
        [0],
        [0],
    ])

# Builds the model.
input_var = keras.layers.Input(shape=(2,), dtype='float32')
hidden = keras.layers.Dense(5, activation='tanh')(input_var)
hidden = keras.layers.Dense(5, activation='tanh')(hidden)
output_var = keras.layers.Dense(1, activation='sigmoid')(hidden)

# Create the model and compile it.
model = keras.models.Model([input_var], [output_var])
model.compile(loss='mean_squared_error', optimizer='sgd')

# Train the model.
model.fit([x], [y], nb_epoch=10000)

# Show the predictions.
preds = model.predict([x])

print(preds)
