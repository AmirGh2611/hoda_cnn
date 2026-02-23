from dataset import load_hoda
from keras import Sequential, Input
from keras.layers import Conv2D, Flatten, Dense, AveragePooling2D, BatchNormalization, Dropout
from keras.losses import SparseCategoricalCrossentropy
from keras.callbacks import TensorBoard

# reading data
x_train, y_train, x_test, y_test = load_hoda()
# preprpocess
mean = x_train.mean()
std = x_train.std()
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std
# uint8 to float32
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
# add channel
x_train = x_train.reshape(-1, 32, 32, 1)
x_test = x_test.reshape(-1, 32, 32, 1)
# model
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
# tensorboard
tensorboard = TensorBoard(log_dir="./logs", histogram_freq=1)
# loss and optimizer
model.compile(optimizer="adam", loss=SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
# training
history = model.fit(x_train, y_train, batch_size=64, epochs=25, validation_split=0.2, callbacks=[tensorboard])
# test
loss, acc = model.evaluate(x_test, y_test)
print("Test loss:", loss)
print("Test accuracy:", acc)
# to see the tensorboard results: python -m tensorboard.main --logdir=logs/