from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, Lambda, MaxPooling2D
from tensorflow.keras.models import Sequential, Model


def lenet(input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]) -> Model:
    num_classes = output_shape[0]

    ##### Your code below (Lab 2)
    
    actual_input_shape = (input_shape[0], input_shape[1], 1)

    model = Sequential()
    print("\n\n", input_shape, "\n\n")
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=actual_input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    
    ##### Your code above (Lab 2)

    return model

