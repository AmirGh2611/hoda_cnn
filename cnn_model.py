from keras import Sequential, Input
from keras.layers import Conv2D, Flatten, Dense, AveragePooling2D, BatchNormalization

model = Sequential([
    Input(shape=(32, 32, 1)),
    Conv2D(16, kernel_size=(3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    AveragePooling2D(pool_size=(2, 2)),  # 16*16
    Conv2D(16, kernel_size=(3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    AveragePooling2D(pool_size=(2, 2)),  # 8*8
    Flatten(),
    Dense(10)
])
