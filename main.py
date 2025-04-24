import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

train_dir = "dataset/train"
test_dir = "dataset/test"

train_gen = ImageDataGenerator(rescale=1./255)
train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=(100, 100),
    class_mode='categorical'
)


test_data = train_gen.flow_from_directory(
    'rockpaperscissors/test',
    target_size=(100, 100),
    class_mode='categorical'
)

#define the model as sequential

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')  # 3 classes: rock, paper, scissors
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(
    train_data,
    epochs=10,
    validation_data=test_data
)

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.legend()
plt.show()

