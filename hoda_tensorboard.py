from keras.losses import SparseCategoricalCrossentropy
from keras.callbacks import TensorBoard
from cnn_model import model
from preprocess import x_train, y_train

# tensorboard
tensorboard = TensorBoard(log_dir="logs")
# loss and optimizer
model.compile(optimizer="adam", loss=SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
# training
model.fit(x_train, y_train, batch_size=64, epochs=25, validation_split=0.2, callbacks=[tensorboard], verbose=0)
# to see the tensorboard results: tensorboard --logdir logs
